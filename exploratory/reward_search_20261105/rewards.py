"""Strict, auditable reward transforms for the post-hoc RecipeQA reward search.

This module does not change the original experiment or its evaluator.  Every
transform starts from the frozen strict four-label ordering score.  No prose,
length, hidden dataset identity, or model-dependent judgment earns reward.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from recipeqa.metrics import score_order


REWARD_FORMULAS = {
    "exact": "e",
    "pair": "q",
    "exp_inv": "exp(-inversions) if valid else 0",
    "success_gate_pair": "Exact if any candidate in group is exactly correct; otherwise Pair",
}
REWARD_NAMES = tuple(REWARD_FORMULAS)
EPSILON = 1e-4


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if (isinstance(completion, list) and len(completion) == 1
            and isinstance(completion[0], dict)
            and completion[0].get("role") == "assistant"
            and isinstance(completion[0].get("content"), str)):
        return completion[0]["content"]
    raise TypeError(f"Unexpected native completion format: {type(completion).__name__}")


def transformed_reward(score: dict, name: str) -> float:
    """All valid transforms lie in [0,1]; malformed answers always receive 0."""
    if name not in REWARD_FORMULAS or name == "success_gate_pair":
        raise ValueError(f"Unknown scalar reward (group gating needs the whole group): {name}")
    if not score["valid"]:
        return 0.0
    q, e = float(score["pair"]), float(score["exact"])
    value = {
        "exact": e,
        "pair": q,
        "exp_inv": math.exp(-score["inversions"]),
    }[name]
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Out-of-range reward {value} for {score}")
    return value


def group_reward_vectors(scores: list[dict]) -> tuple[dict[str, list[float]], str]:
    if len(scores) < 2:
        raise ValueError("At least two candidates are required")
    vectors = {name: [transformed_reward(score, name) for score in scores]
               for name in ("exact", "pair", "exp_inv")}
    has_success = any(score["exact"] for score in scores)
    rule = "exact_when_success_present" if has_success else "pair_when_no_success"
    vectors["success_gate_pair"] = list(vectors["exact"] if has_success else vectors["pair"])
    return vectors, rule


def reference_advantages(rewards: list[float], scale_rewards: str) -> list[float]:
    """Offline audit only; TRL owns actual optimization and numerical behavior."""
    if len(rewards) < 2:
        raise ValueError("At least two candidates are required")
    if scale_rewards not in ("group", "none"):
        raise ValueError(f"Unsupported scaling: {scale_rewards}")
    center = statistics.mean(rewards)
    denominator = statistics.stdev(rewards) + EPSILON if scale_rewards == "group" else 1.0
    return [(value - center) / denominator for value in rewards]


class LoggedSearchReward:
    """Validate entire reward callback batch before appending any result.

    In addition to the original Exact and Pair diagnostics, the returned reward
    and every alternate transform are recorded.  Group membership includes the
    recipe, target, visit and preprocessed input identity, so accidentally mixed
    groups fail rather than silently generate misleading feedback.
    """

    def __init__(self, output_dir, reward_name, num_generations,
                 scale_rewards="group", attempt=None):
        if reward_name not in REWARD_NAMES:
            raise ValueError(f"Unknown reward: {reward_name}")
        if type(num_generations) is not int or num_generations < 2:
            raise ValueError("num_generations must be an integer >= 2")
        if scale_rewards not in ("group", "none"):
            raise ValueError(f"Unsupported scaling: {scale_rewards}")
        self.output_dir = Path(output_dir)
        self.selected_reward = reward_name
        self.num_generations = num_generations
        self.scale_rewards = scale_rewards
        self.attempt = attempt
        self.batch_index = 0
        self.__name__ = f"exploratory_recipeqa_{reward_name}_{scale_rewards}"

    def _append(self, filename, row):
        if self.attempt is not None:
            self.attempt.append(filename, row)
            return
        with (self.output_dir / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def __call__(self, completions, display_perm, qid, recipe_id, presentation_id,
                 target_order=None, completion_ids=None, trainer_state=None,
                 visit_id=None, input_hash=None, input_tokens=None, **kwargs):
        count = len(completions)
        if not count or count % self.num_generations:
            raise RuntimeError("Reward batch must contain complete nonempty G-sized groups")
        if target_order is None:
            raise RuntimeError("Exploratory rewards require explicit target_order metadata")
        columns = {
            "display_perm": display_perm, "qid": qid, "recipe_id": recipe_id,
            "presentation_id": presentation_id, "target_order": target_order,
            "visit_id": visit_id, "input_hash": input_hash, "input_tokens": input_tokens,
        }
        if any(values is None or len(values) != count for values in columns.values()):
            raise RuntimeError("Reward metadata is missing or does not align with completions")
        for name in ("qid", "recipe_id", "presentation_id", "visit_id", "input_hash"):
            if any(value is None or str(value) == "" for value in columns[name]):
                raise RuntimeError(f"Reward metadata contains empty {name}")
        if any(type(value) is not int or value <= 0 for value in input_tokens):
            raise RuntimeError("Reward metadata requires positive integer input-token counts")
        if completion_ids is not None and len(completion_ids) != count:
            raise RuntimeError("Completion token metadata does not align")
        raw = [completion_text(value) for value in completions]
        scores = [score_order(text, perm, target_order=target)
                  for text, perm, target in zip(raw, display_perm, target_order)]
        rewards, components = [], []
        summaries, rollout_records = [], []
        state_step = getattr(trainer_state, "global_step", None)
        for start in range(0, count, self.num_generations):
            end = start + self.num_generations
            group_keys = {
                (str(recipe_id[i]), str(qid[i]), str(presentation_id[i]),
                 tuple(display_perm[i]), tuple(target_order[i]), str(visit_id[i]),
                 str(input_hash[i]), input_tokens[i])
                for i in range(start, end)
            }
            if len(group_keys) != 1:
                raise RuntimeError(f"GRPO group mixes distinct inputs: {group_keys}")
            group_scores = scores[start:end]
            vectors, gate_rule = group_reward_vectors(group_scores)
            selected = vectors[self.selected_reward]
            rewards.extend(selected)
            components.extend([{name: vectors[name][offset] for name in REWARD_NAMES}
                               for offset in range(self.num_generations)])
            attempt_id = self.attempt.attempt_id if self.attempt else "cpu_smoke"
            group_id = f"{attempt_id}:{state_step}:{self.batch_index}:{start // self.num_generations}"
            summaries.append({
                "global_step_before_update": state_step, "batch_index": self.batch_index,
                "group_id": group_id, "qid": qid[start], "recipe_id": recipe_id[start],
                "presentation_id": presentation_id[start], "display_perm": display_perm[start],
                "target_order": target_order[start], "visit_id": visit_id[start],
                "input_hash": input_hash[start], "input_tokens": input_tokens[start],
                "selected_reward": self.selected_reward,
                "reward_formula": REWARD_FORMULAS[self.selected_reward],
                "reward_rule": gate_rule if self.selected_reward == "success_gate_pair" else self.selected_reward,
                "group_has_success": any(score["exact"] for score in group_scores),
                "scale_rewards": self.scale_rewards, "G": self.num_generations,
                "valid_count": sum(score["valid"] for score in group_scores),
                "exact_rewards": [score["exact"] for score in group_scores],
                "pair_rewards": [score["pair"] for score in group_scores],
                "selected_rewards": selected,
                "component_rewards": vectors,
                "exact_zero_variance": len({score["exact"] for score in group_scores}) == 1,
                "pair_zero_variance": len({score["pair"] for score in group_scores}) == 1,
                "selected_zero_variance": len(set(selected)) == 1,
                "selected_mean": statistics.mean(selected),
                "selected_sample_std": statistics.stdev(selected),
                "reference_advantages": reference_advantages(selected, self.scale_rewards),
                "advantage_note": "Offline sample-std audit; trainer owns actual gradients",
            })
            for i in range(start, end):
                rollout_records.append({
                    "global_step_before_update": state_step, "group_id": group_id,
                    "sample_index": i - start, "qid": qid[i], "recipe_id": recipe_id[i],
                    "presentation_id": presentation_id[i], "display_perm": display_perm[i],
                    "target_order": target_order[i], "visit_id": visit_id[i],
                    "input_hash": input_hash[i], "input_tokens": input_tokens[i],
                    "raw_response": raw[i], "selected_reward": self.selected_reward,
                    "reward_formula": REWARD_FORMULAS[self.selected_reward],
                    "reward": rewards[i], "score": scores[i],
                    "component_rewards": components[i],
                    "completion_tokens": len(completion_ids[i]) if completion_ids is not None else None,
                })
        for summary in summaries:
            self._append("reward_groups.jsonl", summary)
        for record in rollout_records:
            self._append("rollouts.jsonl", record)
        log_metric = kwargs.get("log_metric")
        if log_metric is not None:
            for metric in ("valid", "exact", "pair"):
                log_metric(f"recipeqa/{metric}", statistics.mean(score[metric] for score in scores))
            log_metric("recipeqa/selected_reward", statistics.mean(rewards))
            log_metric("recipeqa/zero_variance_group_fraction",
                       statistics.mean(row["selected_zero_variance"] for row in summaries))
        self.batch_index += 1
        return rewards


def reward_table() -> list[dict]:
    rows = [{"valid": False, "correct_pairs": None,
             **{name: 0.0 for name in REWARD_NAMES}}]
    for correct in range(7):
        score = {"valid": 1, "pair": correct / 6, "exact": int(correct == 6),
                 "inversions": 6 - correct}
        rows.append({"valid": True, "correct_pairs": correct,
                     **{name: transformed_reward(score, name) for name in ("exact", "pair", "exp_inv")},
                     "success_gate_pair_if_success_present": score["exact"],
                     "success_gate_pair_if_no_success": score["pair"] if correct < 6 else None})
    return rows
