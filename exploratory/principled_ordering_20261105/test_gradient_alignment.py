"""CPU objective and data-boundary checks; no model or torch imports."""
from __future__ import annotations

from copy import deepcopy
import math
import unittest

from gradient_alignment import coefficient_table, validate_train


def logz(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def objectives(scores, gold, pair):
    z = logz(scores)
    q = [math.exp(value - z) for value in scores]
    return {"exact": q[gold], "pair": sum(p * r for p, r in zip(q, pair)),
            "closed_ce_ascent": scores[gold] - z,
            "raw_sft_sequence_ascent": scores[gold], "log_canonical_mass": z}


def train_rows():
    return [{"recipe_id": f"r{i % 1024}", "source_split": "train", "manifest_set": "train_visits"}
            for i in range(2048)]


class ParameterDiagnosticCPUTests(unittest.TestCase):
    def test_all_five_detached_coefficient_formulas_by_finite_difference(self):
        scores = [-5 + index / 31 for index in range(24)]
        gold = 7
        z = logz(scores)
        q = [math.exp(value - z) for value in scores]
        exact = [int(index == gold) for index in range(24)]
        pair = [(index % 7) / 6 for index in range(24)]
        pair[gold] = 1.0
        coefficients = coefficient_table(q, exact, pair)
        epsilon = 1e-5
        for coordinate in range(24):
            high, low = scores.copy(), scores.copy()
            high[coordinate] += epsilon
            low[coordinate] -= epsilon
            high_objectives, low_objectives = objectives(high, gold, pair), objectives(low, gold, pair)
            for name in coefficients:
                finite_difference = (high_objectives[name] - low_objectives[name]) / (2 * epsilon)
                self.assertAlmostEqual(coefficients[name][coordinate], finite_difference, places=9)

    def test_coefficient_identities(self):
        q = [1 / 24] * 24
        exact = [1] + [0] * 23
        pair = [1] + [.5] * 23
        coefficients = coefficient_table(q, exact, pair)
        for coordinate in range(24):
            self.assertAlmostEqual(coefficients["exact"][coordinate], q[0] * coefficients["closed_ce_ascent"][coordinate])
            self.assertAlmostEqual(coefficients["closed_ce_ascent"][coordinate],
                                   coefficients["raw_sft_sequence_ascent"][coordinate] - coefficients["log_canonical_mass"][coordinate])

    def test_reject_missing_or_multiple_exact_targets(self):
        for exact in ([0] * 24, [1, 1] + [0] * 22):
            with self.assertRaises(ValueError):
                coefficient_table([1 / 24] * 24, exact, [.5] * 24)

    def test_original_training_prefix_only(self):
        records = train_rows()
        self.assertEqual(validate_train(records, 1), records[:1])
        self.assertEqual(validate_train(records, 4), records[:4])
        for field, value in (("source_split", "val"), ("source_split", "test"), ("manifest_set", "val")):
            bad = deepcopy(records)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                validate_train(bad, 1)
        for limit in (0, 5):
            with self.assertRaises(ValueError):
                validate_train(records, limit)
        with self.assertRaises(ValueError):
            validate_train(records[:4], 1)

    def test_reject_duplicate_recipe_prefix(self):
        records = train_rows()
        records[1]["recipe_id"] = records[0]["recipe_id"]
        with self.assertRaises(ValueError):
            validate_train(records, 2)


if __name__ == "__main__":
    unittest.main()
