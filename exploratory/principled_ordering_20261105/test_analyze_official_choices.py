"""CPU-only adversarial fixtures; no real data, GPU, network or file writes."""
import copy
import importlib.util
from itertools import permutations
from pathlib import Path
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("official_choice_bridge", HERE / "analyze_official_choices.py")
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def fixtures(count=200):
    source_rows = []
    display = [2, 0, 3, 1]
    manifest = []
    for index in range(count):
        images = [f"recipe_{index}_{step}_0.jpg" for step in (2, 4, 6, 8)]
        orders = [[0, 1, 2, 3], [1, 0, 2, 3], [3, 2, 1, 0], [1, 0, 2, 3]]
        shift = index % 4
        orders = orders[shift:] + orders[:shift]
        answer = orders.index([0, 1, 2, 3])
        source_rows.append({"task": "visual_ordering", "qid": f"q{index}", "recipe_id": f"r{index}",
                            "choice_list": [[images[i] for i in order] for order in orders], "answer": answer,
                            "context": [{"id": step} for step in range(1, 9)]})
        manifest.append({"qid": f"q{index}", "recipe_id": f"r{index}", "presentation_id": f"q{index}::p2031",
                         "source_split": "val", "manifest_set": "val", "display_perm": display,
                         "target_order": [0, 1, 2, 3], "target_labels": ["B", "D", "A", "C"],
                         "images_chronological": images, "source_step_ids_chronological": [2, 4, 6, 8]})
    source = bridge.audit_source(source_rows)
    config = {"cli": {"limit": 200, "tie_atol": bridge.TIE_ATOL, "microbatch": 1, "run_id": "fixture"},
              "manifest_sha256": bridge.DEV_SHA, "post_hoc_exploration": True,
              "candidate_texts": list(bridge.CANONICALS), "selected_rows_sha256": bridge.digest(manifest),
              "source_sha256": {"exploratory/principled_ordering_20261105/score_permutations.py": bridge.SCORER_SHA,
                                "src/recipeqa/metrics.py": bridge.METRIC_SHA},
              "input": {"min_pixels": 65536, "max_pixels": 262144, "recipe_token_budget": 1536,
                        "max_input_tokens": 4096, "max_completion_tokens": 32, "prompt_version": "recipeqa-order-v1"},
              "weight_identity": {"base": {"weights_sha256": "a" * 64}, "adapter": {"weights_sha256": "b" * 64}}}
    identity = bridge.digest(config["weight_identity"])
    config["model_weights_sha256"] = identity
    signature = bridge.digest(config)
    scored = []
    for expected in manifest:
        gold = " ".join(expected["target_labels"])
        option_texts = [" ".join(chr(65 + display.index(s)) for s in order)
                        for order in source[expected["qid"]]["choice_semantic_permutations"]]
        outside = next(text for text in bridge.CANONICALS if text not in option_texts)
        values = [-1.0 if text == outside else -2.0 if text == gold else -10.0 for text in bridge.CANONICALS]
        namespace = bridge.digest(["visible_fixture", expected["qid"]])
        chosen, _ = bridge.choose(values, bridge.CANONICALS, namespace, "canonical-map-tie-v1")
        candidates = [{"text": text, "token_ids": [100 + i, 1], "token_count": 2, "eos_token_id": 1,
                       "token_logps_including_eos": [value + .1, -.1],
                       "logp_including_eos": value, "logp_without_eos_diagnostic": value + .1}
                      for i, (text, value) in enumerate(zip(bridge.CANONICALS, values))]
        scored.append({**expected, "run_id": "fixture", "configuration_sha256": signature,
                       "model_weights_sha256": identity, "technical_error": None,
                       "input_audit": {"prompt_text_hash": namespace}, "candidates": candidates,
                       "canonical_map": {"map_text": bridge.CANONICALS[chosen], "map_candidate_index": chosen},
                       "canonical_map_score": bridge.score_order(bridge.CANONICALS[chosen], display, [0, 1, 2, 3]),
                       "greedy": {"raw_response": outside, "score": bridge.score_order(outside, display, [0, 1, 2, 3])}})
    return source_rows, source, manifest, config, scored


class ChoiceBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_rows, cls.source, cls.manifest, cls.config, cls.rows = fixtures()

    def validate(self, row, expected=None):
        return bridge.validate_score_row(row, expected or self.manifest[0], self.config,
                                         bridge.digest(self.config), self.config["model_weights_sha256"])

    def test_all200_and_duplicate_wrong_options_retained(self):
        bridge.validate_manifest(self.manifest, self.source)
        self.assertEqual(len(self.source), 200)
        self.assertTrue(all(item["duplicate_incorrect_options"] == 1 for item in self.source.values()))

    def test_576_display_and_option_mappings(self):
        for display in permutations(range(4)):
            for order in permutations(range(4)):
                text = " ".join(chr(65 + display.index(i)) for i in order)
                scored = bridge.score_order(text, list(display), [0, 1, 2, 3])
                self.assertEqual(scored["exact"], int(order == (0, 1, 2, 3)))

    def test_four_choices_can_improve_without_any_model_change(self):
        row = self.rows[0]
        values, index, map_score, greedy_score = self.validate(row)
        prediction = bridge.analyze_row(row, self.source[row["qid"]], values, index, map_score, greedy_score)
        self.assertEqual(prediction["four_choice_exact"], 1)
        self.assertEqual(prediction["canonical24_map_score"]["exact"], 0)
        self.assertEqual(prediction["greedy_score"]["exact"], 0)
        self.assertEqual(len(prediction["official_option_complete_logps"]), 4)

    def test_duplicate_correct_option_rejected(self):
        rows = copy.deepcopy(self.source_rows[:1])
        gold = rows[0]["answer"]
        rows[0]["choice_list"][(gold + 1) % 4] = rows[0]["choice_list"][gold]
        with self.assertRaises(ValueError):
            bridge.audit_source(rows)

    def test_source_images_not_same_set_rejected(self):
        rows = copy.deepcopy(self.source_rows[:1])
        rows[0]["choice_list"][0][0] = "foreign_4_0.jpg"
        with self.assertRaises(ValueError):
            bridge.audit_source(rows)

    def test_incomplete_and_duplicate_manifest_rejected(self):
        with self.assertRaises(ValueError):
            bridge.validate_manifest(self.manifest[:-1], self.source)
        manifest = copy.deepcopy(self.manifest)
        manifest[1] = manifest[0]
        with self.assertRaises(ValueError):
            bridge.validate_manifest(manifest, self.source)

    def test_test_confirmation_train_rejected(self):
        for path in ("data/test.json", "data/confirmation200.jsonl", "results/test/scores.jsonl"):
            with self.assertRaises(ValueError):
                bridge.no_test_or_confirmation(path)
        for split in ("test", "train", "confirmation"):
            row = copy.deepcopy(self.rows[0])
            row["source_split"] = split
            with self.assertRaises(ValueError):
                self.validate(row)

    def test_missing_or_duplicate_canonical_candidate_rejected(self):
        row = copy.deepcopy(self.rows[0])
        row["candidates"].pop()
        with self.assertRaises(ValueError):
            self.validate(row)
        row = copy.deepcopy(self.rows[0])
        row["candidates"][1] = row["candidates"][0]
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_missing_eos_and_inconsistent_logpsum_rejected(self):
        row = copy.deepcopy(self.rows[0])
        row["candidates"][0]["token_ids"][-1] = 2
        with self.assertRaises(ValueError):
            self.validate(row)
        row = copy.deepcopy(self.rows[0])
        row["candidates"][0]["logp_including_eos"] += .2
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_tie_rule_target_independent_and_keeps_duplicates(self):
        scores = [-4.0] * 4
        texts = ["A B C D", "B A C D", "B A C D", "D C B A"]
        chosen, tied = bridge.choose(scores, texts, "namespace", "official-choice-map-tie-v1")
        self.assertEqual(tied, [0, 1, 2, 3])
        self.assertEqual(chosen, bridge.choose(scores, texts, "namespace", "official-choice-map-tie-v1")[0])
        # Rotating the recorded gold index has no access to this decision API.
        self.assertEqual(len({bridge.choose(scores, texts, "namespace", "official-choice-map-tie-v1")[0]
                              for hypothetical_gold in range(4)}), 1)

    def test_micro_macro_are_distinct_in_general(self):
        rows = [{"recipe_id": recipe, "four_choice_exact": exact,
                 "canonical24_map_score": {"exact": exact}, "greedy_score": {"exact": exact, "valid": 1}}
                for recipe, exact in (("rA", 1), ("rA", 1), ("rB", 0))]
        metrics = bridge.aggregate(rows)
        self.assertAlmostEqual(metrics["four_choice_question_micro"], 2 / 3)
        self.assertAlmostEqual(metrics["four_choice_recipe_macro"], .5)
        self.assertIsNone(bridge.aggregate([])["four_choice_question_micro"])

    def test_full_file_validation_and_stratification_without_file_writes(self):
        completion = {"status": "complete", "rows": 200, "expected_rows": 200,
                      "configuration_sha256": bridge.digest(self.config), "output_sha256": "c" * 64}
        with patch.object(bridge, "load", side_effect=lambda path: self.config if str(path).endswith("config.json") else completion), \
             patch.object(bridge, "read_jsonl", return_value=self.rows), \
             patch.object(bridge, "sha", return_value="c" * 64), \
             patch.object(bridge, "receipt", side_effect=lambda path: {"path": str(path), "sha256": "c" * 64}):
            predictions, summary = bridge.analyze_file(Path("scores.jsonl"), self.manifest, self.source)
        self.assertEqual(len(predictions), 200)
        self.assertEqual(summary["metrics"]["four_choice_question_micro"], 1.0)
        self.assertEqual(summary["metrics"]["canonical24_map_question_micro"], 0.0)
        self.assertEqual(summary["duplicate_incorrect_options_strata"]["duplicate_incorrect_options"]["questions"], 200)
        self.assertEqual(summary["duplicate_incorrect_options_strata"]["no_duplicate_incorrect_options"]["questions"], 0)


if __name__ == "__main__":
    unittest.main()
