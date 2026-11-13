"""Ensure blank controls preserve labels and analytic baselines aren't fabricated."""

from fractions import Fraction
import importlib.util
from itertools import permutations
from pathlib import Path
import sys
import unittest

SPEC = importlib.util.spec_from_file_location("prepare_sanity_controls", Path(__file__).resolve().parents[1] / "scripts" / "prepare_sanity_controls.py")
controls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls)


def record(display=(2, 0, 3, 1)):
    return {"qid": "q", "recipe_id": "r", "presentation_id": "q::p2031",
            "steps": [{"source_step_id": i, "title": str(i), "body": "state"} for i in range(1, 5)],
            "display_perm": list(display), "target_order": [0, 1, 2, 3],
            "target_labels": ["ABCD"[display.index(i)] for i in range(4)],
            "source_step_ids_chronological": [1, 2, 3, 4],
            "image_paths_display": [f"img_{i}.jpg" for i in display],
            "image_paths_chronological": [f"img_{i}.jpg" for i in range(4)]}


class SanityControlTests(unittest.TestCase):
    def test_blank_changes_pixels_only_not_task_labels(self):
        original = record()
        blank = controls.blank_record(original, "_controls/blank.png", "imagehash", "manifesthash")
        for field in ("qid", "recipe_id", "presentation_id", "steps", "display_perm", "target_order", "target_labels"):
            self.assertEqual(blank[field], original[field])
        self.assertEqual(blank["image_paths_display"], ["_controls/blank.png"] * 4)
        self.assertEqual(blank["image_paths_chronological"], ["_controls/blank.png"] * 4)
        self.assertEqual(len(blank["control_origin_image_paths_sha256"]), 64)
        self.assertNotEqual(original["image_paths_display"], blank["image_paths_display"])

    def test_two_layout_equivariance_for_every_input_layout(self):
        for display in permutations(range(4)):
            original = record(display)
            alternative = controls.alternate_layout(original)
            self.assertNotEqual(alternative["display_perm"], original["display_perm"])
            order = [alternative["display_perm"]["ABCD".index(label)] for label in alternative["target_labels"]]
            self.assertEqual(order, [0, 1, 2, 3])
            self.assertEqual(alternative["image_paths_display"], original["image_paths_display"][::-1])

    def test_uniform_expectations_and_fixed_copy_baseline(self):
        rows = [record(display) for display in permutations(range(4))]
        for index, row in enumerate(rows):
            row["qid"], row["recipe_id"] = f"q{index}", f"r{index}"
        result = controls.analytic_controls(rows)
        random_control = result["uniform_random_legal_permutation"]
        fractions = random_control["expected_micro_and_recipe_macro_fractions"]
        self.assertEqual(Fraction(fractions["exact"]), Fraction(1, 24))
        self.assertEqual(Fraction(fractions["pair"]), Fraction(1, 2))
        self.assertEqual(Fraction(fractions["valid"]), 1)
        self.assertEqual(result["copy_display_order"]["micro"]["exact"], 1 / 24)
        self.assertEqual(result["copy_display_order"]["micro"]["pair"], .5)
        self.assertIn("NOT_model", result["copy_display_order"]["kind"])
        self.assertIn("NOT_model_results", random_control["kind"])


if __name__ == "__main__":
    unittest.main()
