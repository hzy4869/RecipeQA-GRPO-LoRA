"""CPU-only mathematical / leakage-boundary checks; no model imports."""
from __future__ import annotations

from copy import deepcopy
import math
import unittest

from score_permutations import CANDIDATES, candidate_suffixes, probability_summary, validate_records


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        # A tiny prefix-stable literal tokenizer with one EOS sentinel.
        values = []
        while text:
            if text.startswith("<EOS>"):
                values.append(99)
                text = text[5:]
            else:
                values.append(ord(text[0]) + 1000)
                text = text[1:]
        return values

    def decode(self, values, skip_special_tokens=False):
        return "".join("<EOS>" if value == 99 else chr(value - 1000) for value in values)


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        prefix = "USER\nASSISTANT\n"
        if add_generation_prompt:
            return prefix
        return prefix + messages[-1]["content"] + "<EOS>\n"


def dev_rows():
    return [{"recipe_id": f"r{i}", "qid": f"q{i}", "presentation_id": f"p{i}",
             "source_split": "val", "manifest_set": "val", "target_labels": list("ABCD")}
            for i in range(200)]


class DiagnosticsTests(unittest.TestCase):
    def test_candidate_space_and_chat_eos(self):
        values = candidate_suffixes(FakeProcessor(), [{"role": "user", "content": "x"}])
        self.assertEqual(len(values), 24)
        self.assertEqual(len(set(CANDIDATES)), 24)
        for row in values:
            self.assertEqual(row["token_ids"][-1], 99)
            self.assertEqual(row["post_eos_whitespace_token_ids"], [ord("\n") + 1000])

    def test_reject_test_even_if_named_development(self):
        rows = dev_rows()
        validate_records(rows)
        for field, value in (("source_split", "test"), ("source_split", "train"),
                             ("manifest_set", "orbit")):
            bad = deepcopy(rows)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                validate_records(bad)
        bad = deepcopy(rows)
        bad[1]["recipe_id"] = bad[0]["recipe_id"]
        with self.assertRaises(ValueError):
            validate_records(bad)

    def test_uniform_marginals_and_target_independent_tie(self):
        scores = [-10.0] * 24
        result = probability_summary(scores, CANDIDATES[0], "visible-prompt")
        changed_gold = probability_summary(scores, CANDIDATES[-1], "visible-prompt")
        self.assertEqual(result["map_text"], changed_gold["map_text"])
        self.assertEqual(result["gold_rank_best"], 1)
        self.assertEqual(result["gold_rank_worst"], 24)
        self.assertAlmostEqual(result["q24_gold"], 1 / 24)
        self.assertAlmostEqual(result["logZ_canonical_24"], -10 + math.log(24))
        for row in result["ordinal_marginals_display_label_by_rank"]:
            for value in row:
                self.assertAlmostEqual(value, .25)
        for value in result["pair_marginals_display_label_before"].values():
            self.assertAlmostEqual(value, .5)

    def test_map_uses_sum_not_average_or_gold(self):
        scores = [-100.0] * 24
        scores[7] = -2.0
        scores[3] = -3.0
        result = probability_summary(scores, CANDIDATES[3], "x")
        self.assertEqual(result["map_candidate_index"], 7)
        self.assertEqual(result["gold_rank_best"], 2)
        self.assertAlmostEqual(result["top_two_logp_margin"], 1)
        self.assertAlmostEqual(sum(result["q24"]), 1)
        self.assertAlmostEqual(result["q24"][7] / result["q24"][3], math.e)

    def test_exact_expected_reward_gradient_is_scaled_closed_ce(self):
        # Independent finite-difference check over 24 unconstrained sequence logits.
        scores = [i / 17 - 4 for i in range(24)]
        gold = 5
        q = probability_summary(scores, CANDIDATES[gold], "x")["q24"]
        step = 1e-5
        for i in range(24):
            high, low = scores.copy(), scores.copy()
            high[i] += step
            low[i] -= step
            ph = probability_summary(high, CANDIDATES[gold], "x")["q24_gold"]
            pl = probability_summary(low, CANDIDATES[gold], "x")["q24_gold"]
            finite_difference = (ph - pl) / (2 * step)
            expected = q[gold] * ((1 if i == gold else 0) - q[i])
            self.assertAlmostEqual(finite_difference, expected, places=10)


if __name__ == "__main__":
    unittest.main()
