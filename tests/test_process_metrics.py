"""E4 source alignment, independent parsing, and preregistered review controls."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from recipeqa.metrics import parse_order
from recipeqa.process_metrics import (
    gold_display_steps, human_response_score, parse_process, score_process_record,
    summarize_automatic, summarize_human, validate_human_labels,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = "ABCD"


def manifest(qid="q", recipe="r"):
    return {"recipe_id": recipe, "qid": qid, "display_perm": [2, 0, 3, 1],
            "target_order": [0, 1, 2, 3], "source_step_ids_chronological": [0, 4, 8, 12],
            "steps": [{"source_step_id": source, "title": f"Stage {index}", "body": "Visible state"}
                      for index, source in enumerate([0, 2, 4, 8, 12], 1)],
            "image_paths_display": [f"private_source_{source}_1.jpg" for source in [8, 0, 12, 4]]}


def response(step_numbers=(4, 1, 5, 3), order="B D A C"):
    return "\n".join(f"{label}: step {step} | visible state for image {label}" for label, step in zip(LABELS, step_numbers)) + f"\nOrder: {order}"


def scored(text=None, qid="q", recipe="r", **extra):
    item = manifest(qid, recipe)
    prediction = {**item, "raw_response": response() if text is None else text, **extra}
    return score_process_record(prediction, item)


def annotations(blind_id="blind", rater="reviewer_1", F="yes", D="yes", S="yes"):
    return [{"blind_id": blind_id, "image_label": label, "rater_id": rater, "F": F, "D": D, "S": S} for label in LABELS]


class ProcessParserTests(unittest.TestCase):
    def test_main_parser_never_accepts_whole_process(self):
        self.assertIsNone(parse_order(response()))
        parsed = parse_process(response())
        self.assertTrue(parsed["format_valid"])
        self.assertEqual(parsed["order"], [1, 3, 0, 2])

    def test_missing_or_duplicated_image_fields_are_independent_of_order(self):
        missing = response().replace("C: step 5 | visible state for image C\n", "")
        parsed = parse_process(missing)
        self.assertFalse(parsed["format_valid"])
        self.assertEqual(parsed["n_missing_image_fields"], 1)
        self.assertTrue(parsed["order_valid"])
        duplicate = response() + "\nC: step 5 | another observation"
        self.assertTrue(parse_process(duplicate)["images"]["C"]["duplicate"])
        self.assertFalse(parse_process(duplicate)["images"]["C"]["present"])

    def test_duplicate_or_prose_order_cannot_be_extracted(self):
        self.assertFalse(parse_process(response() + "\nOrder: B D A C")["order_valid"])
        self.assertFalse(parse_process(response(order="B D A C because it cooks"))["order_valid"])
        self.assertFalse(parse_process(response().replace("Order:", "The order is"))["order_valid"])

    def test_extra_prose_invalidates_structure_but_unique_order_is_reported(self):
        parsed = parse_process("Here is my reasoning:\n" + response())
        self.assertFalse(parsed["format_valid"])
        self.assertTrue(parsed["order_valid"])

    def test_unclear_is_allowed_format_not_a_numeric_mapping(self):
        item = scored(response(step_numbers=(4, "unclear", 5, 3)))
        self.assertEqual(item["process_metrics"]["format_valid"], 1)
        self.assertEqual(item["process_metrics"]["n_unclear_steps"], 1)
        self.assertEqual(item["process_metrics"]["unique_mapping"], 0)


class SourceAlignmentTests(unittest.TestCase):
    def test_source_zero_and_gaps_map_to_actual_prompt_ordinals(self):
        display, chronological = gold_display_steps(manifest())
        self.assertEqual(display, [4, 1, 5, 3])
        self.assertEqual(chronological, [1, 3, 4, 5])
        values = scored()["process_metrics"]
        self.assertEqual(values["exact"], 1)
        self.assertEqual(values["step_match_mean"], 1)
        self.assertEqual(values["all_steps_match"], 1)
        self.assertEqual(values["mapping_order_consistent"], 1)

    def test_source_ids_must_not_be_assumed_equal_to_prompt_ordinals(self):
        values = scored(response(step_numbers=(8, 0, 12, 4)))["process_metrics"]
        self.assertLess(values["step_match_mean"], 1)
        self.assertEqual(values["unique_mapping"], 0)

    def test_missing_source_context_is_hard_error(self):
        item = manifest()
        item["steps"] = item["steps"][:-1]
        with self.assertRaises(ValueError):
            gold_display_steps(item)

    def test_wrong_mapping_can_be_internally_consistent(self):
        values = scored(response(step_numbers=(1, 2, 3, 4), order="A B C D"))["process_metrics"]
        self.assertEqual(values["mapping_order_consistent"], 1)
        self.assertEqual(values["exact"], 0)
        self.assertLess(values["step_match_mean"], 1)

    def test_duplicate_steps_cannot_count_as_consistent(self):
        values = scored(response(step_numbers=(1, 1, 3, 4)))["process_metrics"]
        self.assertEqual(values["unique_mapping"], 0)
        self.assertEqual(values["mapping_order_comparable"], 0)
        self.assertEqual(values["mapping_order_consistent"], 0)

    def test_missing_observation_does_not_erase_correct_mapping(self):
        text = response().replace("visible state for image A", "")
        values = scored(text)["process_metrics"]
        self.assertEqual(values["format_valid"], 0)
        self.assertEqual(values["step_match_mean"], 1)
        self.assertEqual(values["complete_mapping"], 1)
        self.assertEqual(values["n_missing_observations"], 1)

    def test_answer_only_comparison_uses_same_identity_and_presentation(self):
        item = manifest()
        prediction = {**item, "raw_response": response()}
        changed = score_process_record(prediction, item, {**item, "raw_response": "A B C D"})["process_metrics"]
        self.assertEqual(changed["answer_changed_both_valid"], 1)
        self.assertEqual(changed["exact_change_vs_answer_only"], 1)
        invalid = score_process_record(prediction, item, {**item, "raw_response": "Unknown"})["process_metrics"]
        self.assertIsNone(invalid["answer_changed_both_valid"])
        self.assertEqual(invalid["answer_changed_or_validity_changed"], 1)
        with self.assertRaises(ValueError):
            score_process_record(prediction, item, {**item, "qid": "different", "raw_response": "B D A C"})

    def test_missing_predictions_leave_fixed_denominator_rates_pending(self):
        values = summarize_automatic([scored()], 2)
        self.assertEqual(values["expected_responses"], 2)
        self.assertEqual(values["expected_images"], 8)
        self.assertEqual(values["n_missing_predictions"], 1)
        self.assertIsNone(values["response_rates"]["exact"])
        technical = scored(technical_error="CUDA failure")
        self.assertIsNone(technical["process_metrics"])
        self.assertEqual(summarize_automatic([technical], 1)["status"], "pending_predictions")


class HumanThresholdTests(unittest.TestCase):
    def test_preregistered_two_discriminative_images_suffice(self):
        rows = annotations()
        rows[2]["D"] = rows[3]["D"] = "no"
        by_label = {row["image_label"]: row for row in rows}
        result = human_response_score(scored(), by_label)
        self.assertEqual((result["E"], result["joint"]), (1, 1))
        self.assertEqual(sum(row["Q"] for row in result["images"].values()), 2)

    def test_one_discriminative_or_one_unclear_fact_fails_E(self):
        rows = {row["image_label"]: row for row in annotations()}
        for label in "BCD":
            rows[label]["D"] = "no"
        self.assertEqual(human_response_score(scored(), rows)["E"], 0)
        rows["A"]["F"] = "unclear"
        self.assertEqual(human_response_score(scored(), rows)["E"], 0)

    def test_support_S_is_separate_from_E(self):
        rows = {row["image_label"]: row for row in annotations(S="no")}
        self.assertEqual(human_response_score(scored(), rows)["E"], 1)

    def test_missing_model_observation_effective_unclear_even_if_rater_says_yes(self):
        text = response().replace("visible state for image A", "")
        rows = {row["image_label"]: row for row in annotations()}
        result = human_response_score(scored(text), rows)
        self.assertEqual(result["images"]["A"]["raw"]["F"], "yes")
        self.assertEqual(result["images"]["A"]["effective"]["F"], "unclear")
        self.assertEqual(result["E"], 0)

    def test_no_human_labels_never_creates_fake_E_or_joint(self):
        item = scored()
        item["blind_id"] = "blind"
        result = summarize_human([item], [], expected_n=1)
        self.assertEqual(result["status"], "pending")
        for rater in result["reviewers"].values():
            self.assertIsNone(rater["E_rate"])
            self.assertIsNone(rater["joint_rate"])
            self.assertIsNone(rater["per_response"][0]["E"])
            self.assertIsNone(rater["per_response"][0]["joint"])

    def test_reviewers_stay_separate_and_agreement_uses_actual_pairs(self):
        item = scored()
        item["blind_id"] = "blind"
        rows = annotations() + annotations(rater="reviewer_2", F="no")
        result = summarize_human([item], rows, expected_n=1)
        self.assertEqual(result["reviewers"]["reviewer_1"]["E_rate"], 1)
        self.assertEqual(result["reviewers"]["reviewer_2"]["E_rate"], 0)
        self.assertEqual(result["reviewers"]["reviewer_1"]["expected_images"], 4)
        self.assertEqual(result["raw_agreement"]["F"]["n_disagree"], 4)
        self.assertEqual(result["raw_agreement"]["D"]["n_agree"], 4)

    def test_unknown_ids_duplicate_or_invalid_ratings_rejected(self):
        with self.assertRaises(ValueError):
            validate_human_labels(annotations(), {"different"})
        with self.assertRaises(ValueError):
            validate_human_labels(annotations() + annotations()[:1], {"blind"})
        with self.assertRaises(ValueError):
            validate_human_labels(annotations(F="probably"), {"blind"})


class ProcessCliTests(unittest.TestCase):
    def test_blind_export_pending_then_two_reviewers_and_separate_consensus(self):
        with tempfile.TemporaryDirectory(prefix=".process-smoke-", dir=ROOT) as temporary:
            directory = Path(temporary)
            manifest_file = directory / "manifest.jsonl"
            manifests = [manifest("q1", "secret_recipe_1"), manifest("q2", "secret_recipe_2")]
            manifest_file.write_text("".join(json.dumps(row) + "\n" for row in manifests))
            predictions = directory / "predictions.jsonl"
            answers = directory / "answers.jsonl"
            rows = [{**item, "run_id": method, "raw_response": response(), "prompt_kind": "process"}
                    for method in ("secret_sft_method", "secret_rl_method") for item in manifests]
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
            answers.write_text("".join(json.dumps({**row, "prompt_kind": "answer", "raw_response": "B D A C"}) + "\n" for row in rows))
            command = [sys.executable, str(ROOT / "scripts/summarize_process.py"), "--manifest", str(manifest_file),
                       "--predictions", str(predictions), "--answer-only", str(answers), "--output-dir", str(directory / "out"),
                       "--bootstrap-samples", "20"]
            subprocess.run(command, check=True, capture_output=True, text=True)
            summary = json.loads((directory / "out/summary.json").read_text())
            self.assertEqual(summary["expected_responses_per_method"], 2)
            self.assertTrue(all(group["human_independent"]["status"] == "pending" for group in summary["groups"]))
            public_text = (directory / "out/blinded/review_items.jsonl").read_text()
            for secret in ("secret_sft_method", "secret_rl_method", "secret_recipe_1", "secret_recipe_2", "private_source_"):
                self.assertNotIn(secret, public_text)
            private = directory / "out/blind_mapping.private.json"
            self.assertEqual(private.stat().st_mode & 0o777, 0o600)
            label_files = []
            for reviewer in ("reviewer_1", "reviewer_2"):
                template = directory / f"out/blinded/labels_{reviewer}.template.csv"
                with template.open(newline="") as stream:
                    reader = csv.DictReader(stream)
                    fields = reader.fieldnames
                    labels = list(reader)
                self.assertEqual(len(labels), 16)  # Two methods × two responses × four images.
                for row in labels:
                    row.update({"F": "yes", "D": "yes", "S": "yes"})
                with template.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader(); writer.writerows(labels)
                label_files.append(template)
            preserved = [file.read_bytes() for file in label_files]
            subprocess.run(command + ["--labels", *map(str, label_files)], check=True, capture_output=True, text=True)
            self.assertEqual([file.read_bytes() for file in label_files], preserved)
            summary = json.loads((directory / "out/summary.json").read_text())
            for group in summary["groups"]:
                self.assertEqual(group["automatic"]["response_rates"]["exact"], 1)
                self.assertEqual(group["human_independent"]["status"], "complete")
                self.assertEqual(group["human_consensus_separate"]["status"], "pending")
                for rater in group["human_independent"]["reviewers"].values():
                    self.assertEqual((rater["E_rate"], rater["joint_rate"], rater["image_Q_rate"]), (1, 1, 1))
                    self.assertEqual(rater["image_criterion_rates"]["F"]["yes"], 1)
            comparison = summary["paired_comparisons"][0]
            self.assertEqual(comparison["automatic"]["metrics"]["exact"]["estimate"], 0)
            self.assertEqual(comparison["human"]["independent:reviewer_1"]["metrics"]["joint"]["ci95"], [0, 0])
            self.assertEqual(comparison["human"]["consensus:consensus"]["status"], "pending")
            # A reviewed response must never silently change under the same ID.
            original_private = private.read_bytes()
            rows[0]["raw_response"] += "\nAdditional text changes this reviewed response."
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
            rejected = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("previously blinded response changed", rejected.stderr)
            self.assertEqual(private.read_bytes(), original_private)


if __name__ == "__main__":
    unittest.main()
