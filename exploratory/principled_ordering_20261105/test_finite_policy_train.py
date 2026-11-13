"""CPU mathematics/boundary tests; optional tiny-Torch integration is separate."""
from __future__ import annotations

from copy import deepcopy
from itertools import product
import math
import random
import unittest

from finite_policy_train import categorical_draws, log_policy, policy_coefficients, validate_budget


def objective(scores, rewards, reference, beta):
    policy = log_policy(scores)
    return sum(q * reward for q, reward in zip(policy["q"], rewards)) - beta * sum(
        q * (current - ref) for q, current, ref in zip(policy["q"], policy["logq"], reference))


def records(count=256):
    return [{"qid": f"q{i}", "presentation_id": f"p{i}", "recipe_id": f"r{i}",
             "source_split": "train", "manifest_set": "train_all7558", "technical_status": "ok",
             "target_labels": list("ABCD")} for i in range(count)]


class FinitePolicyMathTests(unittest.TestCase):
    def test_enumerated_exact_and_kl_match_full_objective_finite_difference(self):
        scores = [-3 + index / 19 for index in range(24)]
        reference = log_policy([-2 + (23 - index) / 31 for index in range(24)])["logq"]
        rewards = [int(index == 6) for index in range(24)]
        policy = log_policy(scores)
        epsilon = 1e-5
        for method, beta in (("exact", 0.0), ("exact_kl", .37)):
            coefficients = policy_coefficients(method, policy, rewards, reference, beta)["coefficients"]
            for coordinate in range(24):
                high, low = scores.copy(), scores.copy()
                high[coordinate] += epsilon
                low[coordinate] -= epsilon
                numeric = (objective(high, rewards, reference, beta) - objective(low, rewards, reference, beta)) / (2 * epsilon)
                self.assertAlmostEqual(coefficients[coordinate], numeric, places=9)

    def test_rloo_exact_expectation_matches_same_regularized_objective(self):
        # Three supported actions, other21numericallyzero, all24logq finite.
        policy = {"q": [.2, .3, .5] + [0.0] * 21,
                  "logq": [math.log(.2), math.log(.3), math.log(.5)] + [-1000.0] * 21}
        reference = [math.log(.1), math.log(.4), math.log(.5)] + [-1000.0] * 21
        rewards = [1] + [0] * 23
        beta = .17
        target = policy_coefficients("exact_kl", policy, rewards, reference, beta)["coefficients"]
        for k in (4, 8):
            expected = [0.0] * 24
            for samples in product(range(3), repeat=k):
                probability = math.prod(policy["q"][action] for action in samples)
                actual = policy_coefficients("rloo_kl", policy, rewards, reference, beta, list(samples))["coefficients"]
                for i, value in enumerate(actual):
                    expected[i] += probability * value
            for actual, wanted in zip(expected, target):
                self.assertAlmostEqual(actual, wanted, places=12)

    def test_repeated_samples_preserve_multiplicity(self):
        policy = log_policy([-.1 * index for index in range(24)])
        rewards = [1] + [0] * 23
        samples = [0, 0, 0, 1, 1, 2, 2, 2]
        terms = policy_coefficients("rloo_kl", policy, rewards, policy["logq"], 0, samples)
        for action in range(3):
            expected = sum(advantage / 8 for sampled, advantage in zip(samples, terms["advantages"]) if sampled == action)
            self.assertAlmostEqual(terms["coefficients"][action], expected)
        self.assertAlmostEqual(sum(terms["coefficients"]), 0)
        self.assertEqual(samples.count(0), 3)

    def test_constant_wrong_or_correct_group_stays_exactly_inactive(self):
        policy = log_policy([-index for index in range(24)])
        for action in (0, 7):
            terms = policy_coefficients("rloo_kl", policy, [1] + [0] * 23, policy["logq"], .3, [action] * 8)
            self.assertEqual(terms["advantages"], [0.0] * 8)
            self.assertEqual(terms["coefficients"], [0.0] * 24)

    def test_raw_sft_has_only_gold_and_no_hidden_kl(self):
        policy = log_policy([-index for index in range(24)])
        exact = [int(index == 5) for index in range(24)]
        terms = policy_coefficients("raw_sft", policy, exact, [-1.0] * 24, 0)
        self.assertEqual(terms["coefficients"], exact)
        self.assertEqual(terms["active_beta"], 0)

    def test_categorical_sampling_is_reproducible_and_not_gold_conditioned(self):
        q = [.2, .3, .5] + [0.0] * 21
        first = categorical_draws(q, 8, random.Random(44))
        self.assertEqual(first, categorical_draws(q, 8, random.Random(44)))
        self.assertEqual(len(first[0]), 8)
        counts = [0] * 3
        rng = random.Random(123)
        for _ in range(12500):
            indices, uniforms = categorical_draws(q, 8, rng)
            for index in indices:
                counts[index] += 1
        for observed, expected in zip(counts, q[:3]):
            self.assertLess(abs(observed / 100000 - expected), .005)

    def test_budget_is_fixed_prefix_no_heldout_or_partial_accumulation(self):
        source = records()
        self.assertEqual(validate_budget(source, 256, 64), source)
        self.assertEqual(validate_budget(source, 4, 1), source[:4])
        for limit, steps in ((3, None), (5, None), (260, 65), (256, 63), (4, 0)):
            with self.assertRaises(ValueError):
                validate_budget(source, limit, steps)
        for split in ("val", "test"):
            bad = deepcopy(source)
            bad[0]["source_split"] = split
            with self.assertRaises(ValueError):
                validate_budget(bad, 4, 1)

    def test_bad_q_and_incomplete_actions_fail(self):
        with self.assertRaises(ValueError):
            log_policy([0.0] * 23)
        with self.assertRaises(ValueError):
            categorical_draws([.1] * 24, 8, random.Random(1))
        policy = log_policy([0.0] * 24)
        with self.assertRaises(ValueError):
            policy_coefficients("exact_kl", policy, [0] * 24, policy["logq"], .1)


if __name__ == "__main__":
    unittest.main()
