"""Scientific invariants for mapping, reward, clustering, and finite orbits.

Run with ``PYTHONPATH=src python -m unittest discover -s tests -p test_metrics.py``.
These tests use analytic controls, never claim results from a trained model.
"""

from collections import Counter
from fractions import Fraction
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from recipeqa.metrics import (
    aggregate_scores, bootstrap_mean_ci, display_to_semantic, dual_reward_groups,
    orbit_metrics, paired_recipe_bootstrap, parse_order, rank_position_matrix,
    score_order, score_record, semantic_to_display,
)

ORDERS = list(permutations(range(4)))
ROOT = Path(__file__).resolve().parents[1]


def record(qid="q", recipe="r", display=(0, 1, 2, 3), semantic=(0, 1, 2, 3), **extra):
    order = semantic_to_display(semantic, display) if semantic is not None else None
    response = " ".join(chr(ord("A") + index) for index in order) if order is not None else "I do not know."
    return score_record({"recipe_id": recipe, "qid": qid, "display_perm": list(display),
                         "raw_response": response, **extra})


def orbit(control, recipe="r", qid="q"):
    return [record(qid, recipe, display, control(display)) for display in ORDERS]


class StrictParserTests(unittest.TestCase):
    def test_allowed_whole_answers(self):
        for text in ("A B C D", "abcd", " A, B, C, D ", "Answer: A B C D", "answer a,b,c,d", "A\nB\nC\nD"):
            with self.subTest(text=text):
                self.assertEqual(parse_order(text), (0, 1, 2, 3))

    def test_reject_prose_duplicates_missing_and_punctuation(self):
        for text in (None, "", "A B C", "A A C D", "A B C D E", "Order: A B C D",
                     "A B C D.", "The answer is A B C D", "A B C D because...",
                     "A B C D\nA B C D", "A,B,C,D,", "[A,B,C,D]"):
            with self.subTest(text=text):
                self.assertIsNone(parse_order(text))

    def test_bad_manifest_is_not_a_model_error(self):
        for bad in ([0, 1, 2], [0, 1, 2, 2], [False, 1, 2, 3], ["0", 1, 2, 3]):
            with self.assertRaises(ValueError):
                score_order("A B C D", bad)


class CoordinateAndRewardTests(unittest.TestCase):
    def test_all_576_presentation_target_mappings(self):
        for display in ORDERS:
            for target in ORDERS:
                # Independent explicit reference, not an inverse calling itself.
                expected_display = tuple(next(p for p, identity in enumerate(display) if identity == semantic) for semantic in target)
                self.assertEqual(semantic_to_display(target, display), expected_display)
                self.assertEqual(display_to_semantic(expected_display, display), target)
                score = score_order(expected_display, display, target)
                self.assertEqual((score["exact"], score["pair"], score["valid"]), (1, 1, 1))

    def test_reward_histogram_and_group_variance_reference(self):
        # Independent inversion enumeration reproduces the finite-space oracle.
        inversion_counts = Counter(sum(order[a] > order[b] for a in range(4) for b in range(a + 1, 4)) for order in ORDERS)
        self.assertEqual([inversion_counts[k] for k in range(7)], [1, 3, 5, 6, 5, 3, 1])
        for order in ORDERS:
            score = score_order(order)
            independent_inversions = sum(order[a] > order[b] for a in range(4) for b in range(a + 1, 4))
            self.assertEqual(score["inversions"], independent_inversions)
            self.assertAlmostEqual(score["pair"], 1 - independent_inversions / 6)
        exact_zero = Fraction(1, 24) ** 4 + Fraction(23, 24) ** 4
        pair_zero = sum(Fraction(count, 24) ** 4 for count in inversion_counts.values())
        self.assertAlmostEqual(float(exact_zero), 0.8434666763117284)
        self.assertEqual(pair_zero, Fraction(1355, 165888))
        self.assertAlmostEqual(float(pair_zero), 0.008168161651234568)

    def test_middle_swap_counterexample(self):
        for display in ORDERS:
            score = score_order(semantic_to_display((0, 2, 1, 3), display), display)
            self.assertEqual(score["exact"], 0)
            self.assertEqual(score["pair"], 5 / 6)
            self.assertEqual(score["middle_pair"], 0)
            self.assertEqual((score["first_rank"], score["last_rank"]), (1, 1))

    def test_invalid_is_zero_but_not_valid_reverse_bin(self):
        invalid = score_order("A A C D")
        reverse = score_order("D C B A")
        self.assertEqual(invalid["pair"], 0)
        self.assertEqual(invalid["rank_correct"], [0, 0, 0, 0])
        self.assertEqual(list(invalid["six_pairs"].values()), [0] * 6)
        self.assertIsNone(invalid["pair_bin"])
        self.assertEqual(reverse["pair_bin"], 0)
        self.assertEqual(reverse["full_reverse"], 1)
        self.assertEqual(invalid["full_reverse"], 0)


class FiniteOrbitTests(unittest.TestCase):
    def test_three_rank_position_analytic_controls(self):
        controls = {
            "perfect": (lambda display: (0, 1, 2, 3), [[1] * 4 for _ in range(4)]),
            "copy_display": (lambda display: display, [[int(rank == position) for position in range(4)] for rank in range(4)]),
            "middle_swap": (lambda display: (0, 2, 1, 3), [[1] * 4, [0] * 4, [0] * 4, [1] * 4]),
        }
        for name, (control, expected) in controls.items():
            with self.subTest(control=name):
                matrix = rank_position_matrix(orbit(control))
                self.assertEqual(matrix["exposure_counts"], [[6] * 4 for _ in range(4)])
                self.assertEqual(matrix["pooled_rates"], expected)
                self.assertEqual(matrix["recipe_macro_rates"], expected)
                self.assertTrue(all(abs(sum(row)) < 1e-12 for row in matrix["pooled_row_centered"]))

    def test_blind_copy_has_one_accidental_success(self):
        result = orbit_metrics(orbit(lambda display: display))
        values = result["per_question"][0]
        self.assertEqual(values["success_count"], 1)
        self.assertEqual(values["mean_exact"], 1 / 24)
        self.assertEqual(values["all24_exact"], 0)
        self.assertEqual(values["modal_consistency"], 1 / 24)

    def test_consistent_wrong_and_invalid_do_not_become_success(self):
        wrong = orbit_metrics(orbit(lambda display: (0, 2, 1, 3)))["per_question"][0]
        invalid_rows = orbit(lambda display: None)
        invalid = orbit_metrics(invalid_rows)["per_question"][0]
        self.assertEqual((wrong["modal_consistency"], wrong["mean_exact"]), (1, 0))
        self.assertEqual((invalid["modal_consistency"], invalid["mean_exact"], invalid["n_valid"]), (0, 0, 0))
        self.assertEqual(rank_position_matrix(invalid_rows)["pooled_rates"], [[0] * 4 for _ in range(4)])

    def test_incomplete_or_duplicate_orbit_rejected(self):
        rows = orbit(lambda display: (0, 1, 2, 3))
        with self.assertRaises(ValueError):
            orbit_metrics(rows[:-1])
        with self.assertRaises(ValueError):
            orbit_metrics(rows + rows[:1])
        partial = orbit_metrics(rows[:-1], require_complete=False)
        self.assertEqual((partial["n_complete_questions"], partial["n_incomplete_questions"]), (0, 1))

    def test_invalid_remains_in_consistency_denominator(self):
        rows = orbit(lambda display: (0, 1, 2, 3))
        rows[0] = record(display=ORDERS[0], semantic=None)
        values = orbit_metrics(rows)["per_question"][0]
        self.assertEqual(values["modal_consistency"], 23 / 24)
        self.assertEqual(values["mean_exact"], 23 / 24)


class RecipeClusterTests(unittest.TestCase):
    def test_macro_micro_and_missingness(self):
        rows = [record("a1", "a"), record("a2", "a"), record("b1", "b", semantic=None),
                score_record({"qid": "c1", "recipe_id": "c", "technical_error": "CUDA failure"})]
        result = aggregate_scores(rows)
        self.assertEqual(result["micro"]["exact"], 2 / 3)
        self.assertEqual(result["recipe_macro"]["exact"], 1 / 2)
        self.assertEqual((result["n_scored"], result["n_technical_errors"], result["n_recipes"]), (3, 1, 2))
        self.assertEqual(result["invalid"]["count"], 1)

    def test_duplicate_resume_rows_rejected(self):
        with self.assertRaises(ValueError):
            aggregate_scores([record(), record()])

    def test_paired_cluster_interval_is_seeded_and_recipe_weighted(self):
        left = [record("a1", "a"), record("a2", "a"), record("b1", "b", semantic=(3, 2, 1, 0))]
        right = [record("a1", "a", semantic=(3, 2, 1, 0)), record("a2", "a", semantic=(3, 2, 1, 0)), record("b1", "b", semantic=(3, 2, 1, 0))]
        first = paired_recipe_bootstrap(left, right, n_boot=100, seed=13)
        self.assertEqual(first, paired_recipe_bootstrap(left, right, n_boot=100, seed=13))
        self.assertEqual((first["estimate"], first["n_recipes"]), (0.5, 2))
        identical = paired_recipe_bootstrap(left, left, n_boot=100, seed=13)
        self.assertEqual(identical["ci95"], [0, 0])

    def test_paired_comparison_refuses_unequal_coverage(self):
        with self.assertRaises(ValueError):
            paired_recipe_bootstrap([record()], [record(qid="other")])
        with self.assertRaises(ValueError):
            paired_recipe_bootstrap([record()], [])
        with self.assertRaises(ValueError):
            paired_recipe_bootstrap([record()], [score_record({"recipe_id": "r", "qid": "q", "display_perm": [0, 1, 2, 3], "target_order": [1, 0, 2, 3], "raw_response": "A B C D"})])


class CandidateBankTests(unittest.TestCase):
    def test_same_candidates_three_categories(self):
        rows = [record(semantic=(0, 1, 2, 3), candidate_group_id="both"),
                record(semantic=(0, 2, 1, 3), candidate_group_id="both"),
                record(semantic=(0, 2, 1, 3), candidate_group_id="pair"),
                record(semantic=(3, 2, 1, 0), candidate_group_id="pair"),
                record(semantic=None, candidate_group_id="neither"),
                record(semantic=(3, 2, 1, 0), candidate_group_id="neither")]
        result = dual_reward_groups(rows)
        self.assertEqual(result["categories"], {"both_vary": 1, "pair_only_varies": 1, "neither_varies": 1})

    def test_candidate_groups_cannot_mix_questions(self):
        with self.assertRaises(ValueError):
            dual_reward_groups([record(qid="a", candidate_group_id="bad"), record(qid="b", candidate_group_id="bad")])

    def test_missing_candidate_is_not_treated_as_zero_variance(self):
        with self.assertRaises(ValueError):
            dual_reward_groups([record(candidate_group_id="bad")], expected_group_size=4)
        with self.assertRaises(ValueError):
            dual_reward_groups([score_record({"recipe_id": "r", "qid": "q", "candidate_group_id": "bad", "technical_error": "failed"})])


class SummaryCliTests(unittest.TestCase):
    def test_complete_orbit_two_group_cli(self):
        # Keep temporary outputs under the checkout (data disk remotely), not /tmp.
        with tempfile.TemporaryDirectory(prefix=".metrics-smoke-", dir=ROOT) as temporary:
            directory = Path(temporary)
            predictions = directory / "predictions.jsonl"
            rows = []
            for run_id, control in (("perfect", lambda display: (0, 1, 2, 3)), ("copy", lambda display: display)):
                for row in orbit(control):
                    row["run_id"] = run_id
                    rows.append(row)
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
            subprocess.run([sys.executable, str(ROOT / "scripts/summarize_predictions.py"), str(predictions),
                            "--output-dir", str(directory / "out"), "--orbit", "--paired", "--bootstrap-samples", "50"],
                           check=True, capture_output=True, text=True)
            report = json.loads((directory / "out/summary.json").read_text())
            self.assertEqual(len(report["groups"]), 2)
            self.assertEqual(report["paired_comparisons"][0]["status"], "computed")
            self.assertTrue((directory / "out/rank_position_matrix.csv").exists())
            self.assertTrue((directory / "out/orbit_per_question.csv").exists())
            self.assertEqual({group["identity"]["run_id"] for group in report["groups"]}, {"perfect", "copy"})


if __name__ == "__main__":
    unittest.main()
