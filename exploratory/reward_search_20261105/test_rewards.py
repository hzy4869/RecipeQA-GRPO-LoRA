"""CPU scientific checks for rewards, equivariance, grouping and audit records."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace
import unittest

from rewards import (
    LoggedSearchReward, REWARD_NAMES, completion_text, group_reward_vectors,
    reference_advantages, score_order, transformed_reward,
)


class RewardChecks(unittest.TestCase):
    def test_all_24_orders_under_all_24_displays(self):
        """Reward depends on semantic order, never displayed letters or filenames."""
        permutations = list(itertools.permutations(range(4)))
        for semantic in permutations:
            reference = score_order(semantic)
            for display in permutations:
                letters = " ".join("ABCD"[display.index(value)] for value in semantic)
                actual = score_order(letters, display)
                self.assertEqual(actual, {**reference, "parsed_order": [display.index(v) for v in semantic]})
                for name in ("exact", "pair", "exp_inv"):
                    self.assertEqual(transformed_reward(actual, name), transformed_reward(reference, name))

    def test_invalid_is_zero_not_valid_reverse(self):
        invalids = ("", "A B C A", "A B C D because", "Order: A B C D", "A B C", None)
        for response in invalids:
            score = score_order(response)
            for name in ("exact", "pair", "exp_inv"):
                self.assertEqual(transformed_reward(score, name), 0)
        reverse = score_order("D C B A")
        self.assertAlmostEqual(transformed_reward(reverse, "exp_inv"), math.exp(-6))
        self.assertEqual(transformed_reward(reverse, "pair"), 0)
        self.assertEqual(transformed_reward(score_order("A B C D"), "exp_inv"), 1)

    def test_exponential_has_larger_final_step_gap(self):
        values = [math.exp(-inversions) for inversions in range(7)]
        self.assertGreater(values[0] - values[1], values[1] - values[2])
        for inversions in range(6):
            self.assertGreater(values[inversions], values[inversions + 1])

    def test_success_gate_uses_exact_only_if_success_present(self):
        wrong = [score_order(text) for text in ("A B D C", "A D B C", "D C B A", "invalid")]
        vectors, rule = group_reward_vectors(wrong)
        self.assertEqual(rule, "pair_when_no_success")
        self.assertEqual(vectors["success_gate_pair"], vectors["pair"])
        with_success = [score_order("A B C D"), *wrong[:3]]
        vectors, rule = group_reward_vectors(with_success)
        self.assertEqual(rule, "exact_when_success_present")
        self.assertEqual(vectors["success_gate_pair"], [1, 0, 0, 0])
        self.assertNotEqual(vectors["success_gate_pair"], vectors["pair"])

    def test_scaling_and_constant_groups(self):
        for rewards in ([0, 0, 0, 0], [1, 1, 1, 1]):
            for scale in ("group", "none"):
                self.assertEqual(reference_advantages(rewards, scale), [0, 0, 0, 0])
        rewards = [0, .5, 1, 0]
        centered = [r - statistics.mean(rewards) for r in rewards]
        self.assertEqual(reference_advantages(rewards, "none"), centered)
        self.assertAlmostEqual(sum(reference_advantages(rewards, "group")), 0)
        # Positive rescaling nearly cancels, but epsilon means not EXACTLY equal.
        baseline = reference_advantages(rewards, "group")
        scaled = reference_advantages([10 * r for r in rewards], "group")
        self.assertLess(max(abs(a - b) for a, b in zip(baseline, scaled)), .001)
        self.assertNotEqual(baseline, scaled)

    def test_conversational_completion_strictness(self):
        self.assertEqual(completion_text([{"role": "assistant", "content": "ABCD"}]), "ABCD")
        for bad in ([], [{"role": "user", "content": "ABCD"}], [{"role": "assistant", "content": []}]):
            with self.assertRaises(TypeError):
                completion_text(bad)

    def metadata(self, groups=2):
        count = 4 * groups
        return {
            "display_perm": [[0, 1, 2, 3] for _ in range(count)],
            "target_order": [[0, 1, 2, 3] for _ in range(count)],
            "qid": [f"q{i // 4}" for i in range(count)],
            "recipe_id": [f"r{i // 4}" for i in range(count)],
            "presentation_id": [f"p{i // 4}" for i in range(count)],
            "visit_id": [f"v{i // 4}" for i in range(count)],
            "input_hash": [f"h{i // 4}" for i in range(count)],
            "input_tokens": [1234] * count,
            "trainer_state": SimpleNamespace(global_step=3),
        }

    def test_returned_rewards_equal_saved_rewards_and_gates(self):
        # Temp is explicitly inside exploratory directory, never remote /tmp.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            reward = LoggedSearchReward(directory, "success_gate_pair", 4)
            responses = ["ABCD", "ABDC", "ADBC", "bad", "ABDC", "ADBC", "DCBA", "bad"]
            returned = reward(responses, **self.metadata())
            groups = [json.loads(line) for line in (Path(directory) / "reward_groups.jsonl").read_text().splitlines()]
            rollouts = [json.loads(line) for line in (Path(directory) / "rollouts.jsonl").read_text().splitlines()]
            self.assertEqual(returned, [row["reward"] for row in rollouts])
            self.assertEqual(returned, sum([row["selected_rewards"] for row in groups], []))
            self.assertEqual(groups[0]["reward_rule"], "exact_when_success_present")
            self.assertEqual(groups[1]["reward_rule"], "pair_when_no_success")
            self.assertEqual(returned[:4], [1, 0, 0, 0])
            self.assertAlmostEqual(returned[4], 5 / 6)
            self.assertEqual([row["sample_index"] for row in rollouts], [0, 1, 2, 3] * 2)
            self.assertEqual(len({row["group_id"] for row in groups}), 2)
            self.assertTrue(all(set(row["component_rewards"]) == set(REWARD_NAMES) for row in rollouts))

    def test_mixed_later_group_fails_without_partial_log(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            reward = LoggedSearchReward(directory, "exp_inv", 4)
            metadata = self.metadata()
            metadata["target_order"][-1] = [3, 2, 1, 0]
            with self.assertRaises(RuntimeError):
                reward(["ABCD"] * 8, **metadata)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_missing_input_identity_fails(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            reward = LoggedSearchReward(directory, "pair", 4)
            metadata = self.metadata(1)
            del metadata["input_hash"]
            with self.assertRaises(RuntimeError):
                reward(["ABCD"] * 4, **metadata)

    def test_all_rewards_logged_as_returned(self):
        for name in REWARD_NAMES:
            with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
                reward = LoggedSearchReward(directory, name, 4, "none")
                returned = reward(["ABCD", "ABDC", "DCBA", "bad"], **self.metadata(1))
                group = json.loads((Path(directory) / "reward_groups.jsonl").read_text())
                self.assertEqual(group["selected_rewards"], returned)
                self.assertEqual(group["reference_advantages"], reference_advantages(returned, "none"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
