"""Pure CPU tests; synthetic references only, no real data/network/GPU or writes."""
from copy import deepcopy
import math
import unittest

from calibrate_beta import CANDIDATES, PREFIX, calibrate, digest, improve, normalize, validate_reference


def bank(probability=1 / 24):
    logq = [math.log(probability)] + [math.log((1 - probability) / 23)] * 23
    return [list(logq) for _ in range(PREFIX)], [[1] + [0] * 23 for _ in range(PREFIX)]


def reference_fixture():
    records = [{"qid": f"q{i}", "recipe_id": f"r{i}", "presentation_id": f"q{i}::p0123",
                "source_split": "train", "manifest_set": "train_all7558", "technical_status": "ok",
                "display_perm": [0, 1, 2, 3], "target_order": [0, 1, 2, 3], "target_labels": list("ABCD")}
               for i in range(256)]
    config = {"policy": "canonical24_full_sequence_plusEOS_temperature1", "candidate_texts": list(CANDIDATES),
              "selected_records_sha256": digest(records),
              "input": {"min_pixels": 65536, "max_pixels": 262144, "recipe_token_budget": 1536,
                        "max_input_tokens": 4096, "max_completion_tokens": 32, "prompt_version": "recipeqa-order-v1"}}
    signature = digest(config)
    rows = []
    for index, record in enumerate(records):
        token_scores = [[-math.log(24) - .09, -.01] for _ in range(24)]
        logq, q, logz = normalize([sum(values) for values in token_scores])
        rows.append({**{key: value for key, value in record.items() if key not in ("manifest_set", "technical_status")},
                     "index": index, "reference_spec_sha256": signature, "q": q, "logq": logq, "logZ": logz,
                     "token_logps": token_scores, "exact_rewards": [1] + [0] * 23, "gold_candidate_index": 0,
                     "input_audit": {"qid": record["qid"], "recipe_id": record["recipe_id"],
                                     "presentation_id": record["presentation_id"], "input_hash": digest([index])},
                     "candidates": [{"text": text, "token_ids": [100 + i, 1], "token_count": 2, "eos_token_id": 1}
                                    for i, text in enumerate(CANDIDATES)]})
    return rows, config, records


class BetaCalibrationTests(unittest.TestCase):
    def test_uniform_known_improvement(self):
        logq, rewards = bank()
        result = improve(logq[0], rewards[0], math.log(23))
        self.assertAlmostEqual(result["q_beta"][0], .5, places=14)
        self.assertAlmostEqual(result["q_beta"][1], .5 / 23, places=14)
        expected = .5 * math.log(12) + .5 * math.log(12 / 23)
        self.assertAlmostEqual(result["kl"], expected, places=13)

    def test_small_beta_limit_and_KL_monotonicity(self):
        logq, rewards = bank()
        values = [improve(logq[0], rewards[0], t)["kl"] for t in (0, .01, .1, 1, 3, 10, 1000)]
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[-1], math.log(24), places=12)
        self.assertEqual(improve(logq[0], rewards[0], 1000)["q_beta"][0], 1.0)

    def test_positive_root_meets_mean_radius(self):
        logq, rewards = bank()
        result = calibrate(logq, rewards, .02)
        self.assertEqual(result["status"], "calibrated_positive_beta")
        self.assertEqual(result["exit_code"], 0)
        self.assertGreater(result["beta"], 0)
        self.assertAlmostEqual(result["actual_mean_KL"], .02, places=10)
        self.assertLessEqual(abs(result["residual"]), result["absolute_residual_tolerance"])
        self.assertAlmostEqual(result["D_max"], math.log(24), places=12)

    def test_global_beta_with_heterogeneous_reference(self):
        probabilities = [.01, .1, .3, .5, .9, .99, .999, .7] * 8
        logq = [bank(p)[0][0] for p in probabilities]
        rewards = [[1] + [0] * 23 for _ in range(PREFIX)]
        result = calibrate(logq, rewards, .02)
        self.assertAlmostEqual(result["actual_mean_KL"], .02, places=10)
        self.assertAlmostEqual(result["D_max"], sum(-math.log(p) for p in probabilities) / 64, places=12)
        self.assertTrue(all(improve(row, reward, result["inverse_beta"])["exact_expected_reward"] >= p
                            for row, reward, p in zip(logq, rewards, probabilities)))

    def test_boundary_radius_requires_review_and_never_beta_zero(self):
        logq, rewards = bank()
        maximum = math.log(24)
        for delta in (0, -.1, maximum, maximum + 1):
            result = calibrate(logq, rewards, delta)
            self.assertEqual(result["status"], "non_binding_or_boundary_needs_policy_review")
            self.assertEqual(result["exit_code"], 2)
            self.assertIsNone(result["beta"])
            self.assertFalse(result["automatic_beta_zero_fallback"])

    def test_nonfinite_and_missing_gold_fail(self):
        logq, rewards = bank()
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                calibrate(logq, rewards, value)
        bad = deepcopy(rewards)
        bad[0] = [0] * 24
        with self.assertRaises(ValueError):
            calibrate(logq, bad, .02)
        bad = deepcopy(logq)
        bad[0][2] = float("nan")
        with self.assertRaises(ValueError):
            calibrate(bad, rewards, .02)

    def test_logspace_extreme_reference_does_not_take_log_zero(self):
        logits = [-1000.0] + [0.0] * 23
        logq, q, _ = normalize(logits)
        self.assertEqual(q[0], 0.0)  # numeric exp underflow; logq is still finite
        value = improve(logq, [1] + [0] * 23, 1001)
        self.assertTrue(math.isfinite(value["kl"]))
        self.assertGreater(value["q_beta"][0], 0)

    def test_complete_reference_and_raw_EOS_q_validation(self):
        rows, config, records = reference_fixture()
        report = validate_reference(rows, config, records)
        self.assertEqual(report["rows"], 256)
        self.assertEqual(report["canonical_token_count"], 2)
        for mutate in (lambda x: x[0]["exact_rewards"].__setitem__(0, 0),
                       lambda x: x[0].__setitem__("source_split", "val"),
                       lambda x: x[0]["q"].__setitem__(0, .9),
                       lambda x: x[0]["candidates"][0]["token_ids"].__setitem__(-1, 2),
                       lambda x: x[0].__setitem__("gold_candidate_index", 1)):
            bad = deepcopy(rows)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_reference(bad, config, records)
        with self.assertRaises(ValueError):
            validate_reference(rows[:-1], config, records)


if __name__ == "__main__":
    unittest.main()
