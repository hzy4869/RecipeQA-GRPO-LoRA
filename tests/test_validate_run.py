"""Scientific completion checks: missingness must never masquerade as accuracy."""

from __future__ import annotations

import copy
import importlib.util
from itertools import permutations
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_run.py"
SPEC = importlib.util.spec_from_file_location("recipeqa_validate_run", SCRIPT)
VALIDATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATE)


def example(qid="q1", recipe="r1", display=(0, 1, 2, 3), presentation="p1"):
    return {"qid": qid, "recipe_id": recipe, "presentation_id": presentation,
            "display_perm": list(display), "target_order": [0, 1, 2, 3],
            "target_labels": ["ABCD"[display.index(semantic)] for semantic in range(4)]}


def prediction(row, sample_index=0, **overrides):
    return {**copy.deepcopy(row), "run_id": "pair", "checkpoint": "/data/checkpoint-8", "seed": 42,
            "sample_index": sample_index, "prompt_kind": "answer",
            "raw_response": " ".join(row["target_labels"]), "technical_error": None,
            "candidate_group_id": f"group:{row['qid']}:{row['presentation_id']}", **overrides}


def issue_codes(report):
    return {issue["code"] for issue in report["issues"]} | {
        issue["code"] for group in report.get("groups", []) for issue in group["issues"]}


class PredictionCoverageTests(unittest.TestCase):
    def setUp(self):
        self.manifest = [example(), example("q2", "r2", (2, 0, 3, 1), "p2")]
        self.predictions = [prediction(row) for row in self.manifest]

    def check_coverage(self, rows=None, **kwargs):
        return VALIDATE.validate_prediction_coverage(
            self.manifest, self.predictions if rows is None else rows,
            expected_run_id="pair", expected_checkpoint="/data/checkpoint-8", expected_seed=42,
            **kwargs)

    def test_complete_expected_data_passes(self):
        report = self.check_coverage(expected_questions=2, expected_recipes=2)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["groups"][0]["n_unique_successful_keys"], 2)

    def test_model_invalid_is_a_successful_inference_and_stays_in_denominator(self):
        rows = copy.deepcopy(self.predictions)
        rows[0]["raw_response"] = "A A B C"
        report = self.check_coverage(rows)
        self.assertTrue(report["passed"])
        self.assertEqual(report["groups"][0]["n_invalid_model_responses"], 1)
        self.assertEqual(report["groups"][0]["n_successful_model_responses"], 2)

    def test_technical_failure_is_not_model_invalid(self):
        rows = copy.deepcopy(self.predictions)
        rows[0].update(raw_response="", technical_error="Image failed to decode")
        report = self.check_coverage(rows)
        self.assertFalse(report["passed"])
        self.assertEqual(report["groups"][0]["n_invalid_model_responses"], 0)
        self.assertEqual(report["groups"][0]["n_technical_failure_rows"], 1)
        self.assertIn("missing_successful_predictions", issue_codes(report))

    def test_one_missing_question_fails(self):
        report = self.check_coverage(self.predictions[:1])
        self.assertFalse(report["passed"])
        self.assertEqual(report["groups"][0]["missing_successful_keys"][0]["qid"], "q2")

    def test_completely_empty_predictions_fail(self):
        report = self.check_coverage([])
        self.assertFalse(report["passed"])
        self.assertIn("no_prediction_groups", issue_codes(report))

    def test_successful_duplicate_cannot_expand_n(self):
        report = self.check_coverage(self.predictions + self.predictions[:1])
        self.assertFalse(report["passed"])
        self.assertIn("duplicate_prediction_key", issue_codes(report))

    def test_failed_then_success_duplicate_requires_compaction(self):
        failure = {**self.predictions[0], "technical_error": "OOM", "raw_response": ""}
        report = self.check_coverage([failure] + self.predictions)
        self.assertFalse(report["passed"])
        self.assertIn("duplicate_prediction_key", issue_codes(report))

    def test_unexpected_question_fails(self):
        unexpected = prediction(example("q3", "r3"))
        report = self.check_coverage(self.predictions + [unexpected])
        self.assertFalse(report["passed"])
        self.assertIn("unexpected_prediction_key", issue_codes(report))

    def test_correct_id_with_wrong_recipe_or_display_is_not_accepted(self):
        for field, value in (("recipe_id", "other"), ("display_perm", [3, 2, 1, 0]),
                             ("target_labels", ["D", "C", "B", "A"]), ("target_order", [3, 2, 1, 0])):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.predictions)
                rows[0][field] = value
                report = self.check_coverage(rows)
                self.assertFalse(report["passed"])
                self.assertIn("prediction_manifest_mismatch", issue_codes(report))

    def test_wrong_run_or_checkpoint_fails_even_with_full_coverage(self):
        for field, value in (("run_id", "sft"), ("checkpoint", "/data/different"), ("seed", 2026)):
            with self.subTest(field=field):
                rows = [{**row, field: value} for row in self.predictions]
                self.assertFalse(self.check_coverage(rows)["passed"])

    def test_expected_dataset_size_is_checked_against_manifest(self):
        report = self.check_coverage(expected_questions=905, expected_recipes=597)
        self.assertFalse(report["passed"])
        self.assertIn("manifest_question_count_mismatch", issue_codes(report))

    def test_missing_seed_fails(self):
        rows = copy.deepcopy(self.predictions)
        del rows[0]["seed"]
        self.assertFalse(self.check_coverage(rows)["passed"])

    def test_answer_and_process_prompts_cannot_be_mixed(self):
        rows = copy.deepcopy(self.predictions)
        rows[0]["prompt_kind"] = "process"
        report = self.check_coverage(rows)
        self.assertFalse(report["passed"])
        self.assertIn("unexpected_prompt_kind", issue_codes(report))

    def test_e4_prose_is_coverage_not_answer_parser(self):
        rows = [{**row, "prompt_kind": "process", "raw_response": "A: step unclear | unclear\nOrder: A B C D"}
                for row in self.predictions]
        report = self.check_coverage(rows, prompt_kind="process")
        self.assertTrue(report["passed"])
        self.assertIsNone(report["groups"][0]["n_invalid_model_responses"])

    def test_g4_requires_each_sample_index(self):
        rows = [prediction(row, index) for row in self.manifest for index in range(4)]
        report = self.check_coverage(rows, samples_per_input=4)
        self.assertTrue(report["passed"], report)
        missing_one = self.check_coverage(rows[:-1], samples_per_input=4)
        self.assertFalse(missing_one["passed"])
        self.assertEqual(missing_one["groups"][0]["missing_successful_keys"][0]["sample_index"], 3)

    def test_candidate_group_cannot_combine_different_questions(self):
        rows = [prediction(row, index, candidate_group_id="one_group") for row in self.manifest for index in range(4)]
        report = self.check_coverage(rows, samples_per_input=4)
        self.assertFalse(report["passed"])
        self.assertIn("candidate_group_mixes_inputs", issue_codes(report))

    def test_entire_absent_orbit_question_is_detected(self):
        manifest = [example(qid, recipe, layout, f"layout-{index}")
                    for qid, recipe in (("q1", "r1"), ("q2", "r2"))
                    for index, layout in enumerate(permutations(range(4)))]
        report = VALIDATE.validate_prediction_coverage(
            manifest, [prediction(row) for row in manifest[:24]], require_orbit=True,
            expected_questions=2, expected_recipes=2)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["groups"][0]["missing_successful_keys"]), 24)

    def test_all_24_layouts_are_required_in_manifest_itself(self):
        manifest = [example(display=layout, presentation=str(i)) for i, layout in enumerate(permutations(range(4)))]
        complete = VALIDATE.validate_prediction_coverage(manifest, [prediction(row) for row in manifest], require_orbit=True)
        self.assertTrue(complete["passed"], complete)
        incomplete = VALIDATE.validate_prediction_coverage(manifest[:-1], [prediction(row) for row in manifest[:-1]], require_orbit=True)
        self.assertFalse(incomplete["passed"])
        self.assertIn("manifest_incomplete_orbit", issue_codes(incomplete))

    def test_bad_manifest_target_does_not_turn_into_model_error(self):
        manifest = copy.deepcopy(self.manifest)
        manifest[0]["target_labels"] = ["D", "C", "B", "A"]
        report = VALIDATE.validate_prediction_coverage(manifest, self.predictions)
        self.assertFalse(report["passed"])
        self.assertIn("manifest_target_labels_mismatch", issue_codes(report))


class TrainingEvidenceTests(unittest.TestCase):
    def test_empty_directory_never_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            report = VALIDATE.validate_training_artifacts(directory)
        self.assertFalse(report["passed"])
        codes = issue_codes(report)
        self.assertIn("missing_required_artifact", codes)
        self.assertIn("no_full_checkpoint_available", codes)
        self.assertIn("no_verified_adapter_weight_change", codes)

    def test_outcome_alone_does_not_prove_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_outcome.json").write_text(json.dumps({
                "status": "completed", "global_step": 20,
                "initial_adapter_fingerprint": "a" * 64, "final_adapter_fingerprint": "b" * 64,
                "visual_weights_unchanged": True,
            }))
            report = VALIDATE.validate_training_artifacts(directory, min_steps=20)
        self.assertFalse(report["passed"])
        self.assertIn("missing_required_artifact", issue_codes(report))

    def test_nonfinite_gradient_is_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train_outcome.json").write_text(json.dumps({"status": "completed", "global_step": 1}))
            (root / "gradient_audit.jsonl").write_text(json.dumps({
                "global_step_before_update": 0, "l2_norm": float("nan"),
                "trainable_gradient_tensors": 10, "nonzero_gradient_tensors": 10,
            }) + "\n")
            report = VALIDATE.validate_training_artifacts(directory)
        self.assertFalse(report["passed"])
        self.assertIn("invalid_gradient_norm", issue_codes(report))


if __name__ == "__main__":
    unittest.main()
