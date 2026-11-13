"""Synthetic-only tests of independent endpoint mathematics."""
from copy import deepcopy
import math
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bank_math as m


def candidates():
    return [{"text": text, "token_ids": [i + 1, i + 2, i + 3, i + 4, 99],
             "token_count": 5, "eos_token_id": 99} for i, text in enumerate(m.ACTIONS)]


def policy(weights):
    high = max(weights)
    z = high + math.log(sum(math.exp(x - high) for x in weights))
    return m.reconstruct([[(x - z - .1) / 5] * 5 for x in weights], candidates())


class MathematicsTests(unittest.TestCase):
    def test_EOS_sum_is_not_omitted_or_tempered_by_C(self):
        values = [[-1, -1, -1, -1, -.1 - i * .02] for i in range(24)]
        p = m.reconstruct(values, candidates())
        expected = 1 / sum(math.exp(-i * .02) for i in range(24))
        self.assertAlmostEqual(p["q"][0], expected, places=13)
        self.assertNotAlmostEqual(p["q"][0], 1 / 24, places=5)

    def test_bad_support_EOS_probability_and_shape_rejected(self):
        for change in ("missing", "EOS", "nan", "positive", "mass", "order"):
            c = candidates(); values = [[-1.] * 5 for _ in range(24)]
            if change == "missing": values[0].pop()
            if change == "EOS": c[0]["token_ids"][-1] = 98
            if change == "nan": values[0][0] = float("nan")
            if change == "positive": values[0][0] = .01
            if change == "mass": values = [[-.01] * 5 for _ in range(24)]
            if change == "order": c.reverse()
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.reconstruct(values, c)

    def test_gold_is_derived_from_photo_coordinates(self):
        record = {"display_perm": [3, 1, 0, 2], "target_order": [0, 1, 2, 3],
                  "target_labels": ["C", "B", "D", "A"]}
        self.assertEqual(m.ACTIONS[m.target_index(record)], "C B D A")
        record["target_labels"] = ["A", "B", "C", "D"]
        with self.assertRaises(ValueError): m.target_index(record)

    def test_analytic_odds_and_Jgap_identity(self):
        reference = policy([i * .05 for i in range(24)])
        current = policy([math.sin(i) for i in range(24)])
        gold = 7; star = m.analytic_policy(reference, gold)
        ratio_before = reference["q"][gold] / reference["q"][0]
        ratio_after = star["q"][gold] / star["q"][0]
        self.assertAlmostEqual(ratio_after / ratio_before, math.exp(1 / m.BETA), places=13)
        result = m.metrics(current, reference, gold)
        self.assertAlmostEqual(result["Jstar_minus_J"], m.BETA * result["KL_current_qstar"], places=12)
        self.assertTrue(result["qstar_MAP_not_accuracy_upper_bound"])

    def test_qstar_is_the_per_question_objective_maximizer_not_shared_model_proof(self):
        reference = policy([i * .03 for i in range(24)]); gold = 12
        star = m.analytic_policy(reference, gold)
        # Use normalized star scores as complete-string log probabilities with
        # extra residual mass, without changing its conditional distribution.
        p = policy(star["logq"])
        result = m.metrics(p, reference, gold)
        self.assertAlmostEqual(result["Jstar_minus_J"], 0., places=12)
        self.assertAlmostEqual(result["KL_current_qstar"], 0., places=12)

    def test_three_initial_strata_are_exhaustive_and_gold_dependent(self):
        for weights, gold, expected in (
            ([0.] * 24, 0, "initial_MAP_correct"),
            ([.1] + [0.] * 23, 1, "initial_wrong_qstar_MAP_correct"),
            ([2.] + [0.] * 23, 1, "initial_wrong_qstar_MAP_wrong"),
        ):
            ref = policy(weights)
            self.assertEqual(m.metrics(ref, ref, gold)["stratum"], expected)

    def test_exact_tie_is_lexical_and_near_tie_does_not_change_choice(self):
        logs = [-1.] * 24
        result = m.map_details(logs, 1)
        self.assertEqual(result["chosen_index"], 0)
        self.assertEqual(result["tied_max_count"], 24)
        self.assertEqual((result["gold_best_rank"], result["gold_worst_rank"]), (1, 24))
        self.assertEqual(result["gold_rank_tie_broken"], 2)
        logs[1] += 1e-12
        result = m.map_details(logs, 0)
        self.assertEqual(result["chosen_index"], 1)
        self.assertEqual(result["tied_max_count"], 1)
        self.assertEqual(result["near_max_count_at_1e_minus9_nats"], 24)

    def test_common_score_shift_changes_raw_CE_not_conditional_policy(self):
        ref = policy([i * .01 for i in range(24)])
        changed = m.reconstruct([[(s - 2) / 5] * 5 for s in ref["scores"]], candidates())
        a = m.metrics(ref, ref, 3); b = m.metrics(changed, ref, 3)
        self.assertAlmostEqual(a["conditional_NLL"], b["conditional_NLL"], places=12)
        self.assertAlmostEqual(b["raw_sequence_NLL"] - a["raw_sequence_NLL"], 2., places=12)
        self.assertAlmostEqual(b["common_beta_J"], a["common_beta_J"], places=12)

    def test_higher_J_can_have_lower_gold_probability(self):
        ref = policy([0.] * 24)
        high_gold = policy([4.] + [0.] * 23)
        a = m.metrics(ref, ref, 0); b = m.metrics(high_gold, ref, 0)
        self.assertLess(a["qgold"], b["qgold"])
        self.assertGreater(a["common_beta_J"], b["common_beta_J"])

    def test_existing_greedy_parser_invalids_are_not_filtered(self):
        record = {"display_perm": [3, 2, 1, 0], "target_order": [0, 1, 2, 3]}
        self.assertEqual(m.score_greedy("Answer: D,C,B,A", record)["exact"], 1)
        for text in ("D C B B", "I think D C B A", "D C B A.", ",D C B A"):
            self.assertEqual(m.score_greedy(text, record)["valid"], 0)

    def test_completed_old_history_probabilities_and_gap_identity(self):
        project = Path(__file__).resolve().parents[2]
        relative = Path('exploratory/principled_ordering_20261105/finite_trials_v2/flow44')
        base = project if (project / relative / 'raw_sft/run_signature.json').is_file() else project / '.state/remote_principled_20261105'
        for method in ('raw_sft', 'exact_kl'):
            folder = base / relative / method
            signed = json.loads((folder / 'run_signature.json').read_text())
            done = json.loads((folder / 'train_outcome.json').read_text())
            self.assertEqual((done['status'], done['global_step'], done['actual_prompt_visits']), ('completed', 64, 256))
            cache = Path(signed['payload']['cli']['reference_cache']).relative_to('/root/autodl-tmp/rl_recipeqa')
            refs = [json.loads(x) for x in (base / cache).read_text().splitlines()]
            visits = [json.loads(x) for x in (folder / 'actual_prompt_visits.jsonl').read_text().splitlines()]
            for ref, visit in zip(refs[:4], visits[:4]):
                initial = m.reconstruct(ref['token_logps'], ref['candidates'])
                current = m.reconstruct(visit['token_logps_including_eos'], ref['candidates'])
                for a, b in zip(current['q'], visit['q']): self.assertAlmostEqual(a, b, delta=1e-9)
                result = m.metrics(current, initial, ref['gold_candidate_index'])
                self.assertLess(result['gap_identity_residual'], 1e-9)


if __name__ == "__main__": unittest.main(verbosity=2)
