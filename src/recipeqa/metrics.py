"""Strict four-image scoring and recipe-clustered research summaries.

All permutations use zero-based integers. ``display_perm[position]`` is the
hidden semantic identity shown at that position; ``target_order`` is a sequence
of semantic identities. Source step numbers never enter this coordinate system.
This module deliberately has no model, GPU, NumPy, or data-download dependency.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import permutations
import math
import random
import re
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

IDENTITY = (0, 1, 2, 3)
ALL_PERMUTATIONS = frozenset(permutations(IDENTITY))
PAIR_NAMES = tuple(f"{a + 1}_{b + 1}" for a in range(4) for b in range(a + 1, 4))
SCALAR_METRICS = (
    "exact", "pair", "valid", "first_rank", "last_rank", "middle_pair",
    "full_reverse", *tuple(f"pair_{name}" for name in PAIR_NAMES),
    *tuple(f"rank_{rank + 1}" for rank in range(4)),
)


def _permutation(value: Sequence[int], name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must be a four-integer permutation of 0..3")
    if any(type(item) is not int for item in value) or set(value) != set(IDENTITY):
        raise ValueError(f"{name} must be a four-integer permutation of 0..3")
    return tuple(value)


def parse_order(text: str | None) -> tuple[int, ...] | None:
    """Parse only a whole short answer; never extract an answer from prose.

    Accepts case-insensitive A-D, optional ``Answer``/``Answer:``, and spaces or
    commas between labels (compact ``ABCD`` is also valid). Extra prose,
    punctuation, repeated/missing labels, and trailing commas are invalid.
    """
    if not isinstance(text, str):
        return None
    candidate = re.sub(r"^answer\s*:?\s*", "", text.strip(), count=1, flags=re.I)
    if re.fullmatch(r"[A-D](?:[\s,]*[A-D]){3}", candidate, flags=re.I) is None:
        return None
    labels = re.sub(r"[\s,]", "", candidate).upper()
    if len(set(labels)) != 4:
        return None
    return tuple(ord(label) - ord("A") for label in labels)


def display_to_semantic(
    display_order: Sequence[int], display_perm: Sequence[int]
) -> tuple[int, ...]:
    """Map predicted display positions back to their hidden identities."""
    order = _permutation(display_order, "display_order")
    presentation = _permutation(display_perm, "display_perm")
    return tuple(presentation[position] for position in order)


def semantic_to_display(
    semantic_order: Sequence[int], display_perm: Sequence[int]
) -> tuple[int, ...]:
    order = _permutation(semantic_order, "semantic_order")
    presentation = _permutation(display_perm, "display_perm")
    return tuple(presentation.index(semantic) for semantic in order)


def score_order(
    prediction: str | Sequence[int] | None,
    display_perm: Sequence[int] = IDENTITY,
    target_order: Sequence[int] = IDENTITY,
) -> dict[str, Any]:
    """Score strict parsing, six directed relations, and target-rank recovery.

    Malformed predictions score zero on every accuracy metric. A malformed
    manifest permutation raises an error instead of silently becoming a model
    error. ``pair_bin`` counts correct relations (0..6), and is None for invalid
    output so that invalids remain separate from valid complete reversals.
    """
    presentation = _permutation(display_perm, "display_perm")
    target = _permutation(target_order, "target_order")
    if isinstance(prediction, str) or prediction is None:
        parsed = parse_order(prediction)
    else:
        try:
            parsed = _permutation(prediction, "prediction")
        except ValueError:
            parsed = None
    semantic = display_to_semantic(parsed, presentation) if parsed is not None else None
    ranks = {semantic_id: rank for rank, semantic_id in enumerate(semantic or ())}
    relations = {
        f"{a + 1}_{b + 1}": int(semantic is not None and ranks[target[a]] < ranks[target[b]])
        for a in range(4) for b in range(a + 1, 4)
    }
    rank_correct = [int(semantic is not None and semantic[rank] == target[rank]) for rank in range(4)]
    n_correct = sum(relations.values())
    result: dict[str, Any] = {
        "parsed_order": list(parsed) if parsed is not None else None,
        "semantic_order": list(semantic) if semantic is not None else None,
        "valid": int(semantic is not None),
        "exact": int(semantic == target),
        "pair": n_correct / 6,
        "inversions": 6 - n_correct if semantic is not None else None,
        "pair_bin": n_correct if semantic is not None else None,
        "six_pairs": relations,
        "rank_correct": rank_correct,
        "first_rank": rank_correct[0],
        "last_rank": rank_correct[3],
        "middle_pair": relations["2_3"],
        "full_reverse": int(semantic == target[::-1]),
    }
    result.update({f"pair_{name}": value for name, value in relations.items()})
    result.update({f"rank_{rank + 1}": value for rank, value in enumerate(rank_correct)})
    return result


def score_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute from raw text; never trust precomputed rewards in a JSONL."""
    result = dict(record)
    for field in ("recipe_id", "qid"):
        if field not in result or result[field] is None or str(result[field]) == "":
            raise ValueError(f"Prediction missing {field}")
        result[field] = str(result[field])
    if result.get("technical_error") or result.get("status") == "technical_error":
        result["metrics"] = None
        result["technical_error"] = result.get("technical_error") or "unspecified technical error"
        return result
    if "raw_response" not in result:
        raise ValueError("Prediction missing raw_response; use technical_error for inference failures")
    if not isinstance(result["raw_response"], str):
        raise ValueError("raw_response must be a string (including empty string for invalid output)")
    if "display_perm" not in result:
        raise ValueError("Prediction missing display_perm")
    result["display_perm"] = list(_permutation(result["display_perm"], "display_perm"))
    result["target_order"] = list(_permutation(result.get("target_order", IDENTITY), "target_order"))
    result["metrics"] = score_order(result["raw_response"], result["display_perm"], result["target_order"])
    return result


def _usable(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("metrics") is not None]


def _prediction_key(row: Mapping[str, Any]) -> tuple[str, str, tuple[int, ...]]:
    return str(row["recipe_id"]), str(row["qid"]), tuple(row["display_perm"])


def validate_unique_predictions(rows: Iterable[Mapping[str, Any]]) -> None:
    seen = set()
    for row in _usable(rows):
        key = _prediction_key(row)
        if key in seen:
            raise ValueError(f"Duplicate prediction within one model group: {key}")
        seen.add(key)


def per_recipe_metrics(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_recipe: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _usable(rows):
        by_recipe[str(row["recipe_id"])].append(row)
    return [
        {"recipe_id": recipe, "n_predictions": len(group), **{
            metric: mean(row["metrics"][metric] for row in group) for metric in SCALAR_METRICS
        }}
        for recipe, group in sorted(by_recipe.items())
    ]


def aggregate_scores(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep question-micro and recipe-macro estimates separately labeled."""
    records = list(rows)
    usable = _usable(records)
    validate_unique_predictions(usable)
    recipe_rows = per_recipe_metrics(usable)
    counts = Counter(row["metrics"]["pair_bin"] for row in usable)
    n = len(usable)
    return {
        "n_records": len(records), "n_scored": n, "n_recipes": len(recipe_rows),
        "n_technical_errors": len(records) - n,
        "technical_errors": [{"recipe_id": row["recipe_id"], "qid": row["qid"],
                              "error": row.get("technical_error")} for row in records if row.get("metrics") is None],
        "micro": {metric: mean(row["metrics"][metric] for row in usable) if n else None for metric in SCALAR_METRICS},
        "recipe_macro": {metric: mean(row[metric] for row in recipe_rows) if recipe_rows else None for metric in SCALAR_METRICS},
        "pair_bins": {str(k): {"count": counts[k], "rate_all_scored": counts[k] / n if n else None} for k in range(7)},
        "invalid": {"count": counts[None], "rate_all_scored": counts[None] / n if n else None},
        "denominator_note": "Invalid model responses remain in scored denominators; technical inference failures are listed separately, not silently treated as invalid.",
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Percentile bootstrap of independent recipe-level values, not questions."""
    if not values:
        raise ValueError("Cannot bootstrap an empty recipe sample")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    rng = random.Random(seed)
    n = len(values)
    samples = [sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)]
    return {"estimate": mean(values), "ci95": [_quantile(samples, 0.025), _quantile(samples, 0.975)],
            "n_recipes": n, "n_boot": n_boot, "seed": seed,
            "method": "recipe-cluster percentile bootstrap; does not quantify training-seed uncertainty"}


def recipe_bootstrap_ci(
    rows: Iterable[Mapping[str, Any]], metrics: Sequence[str] = ("exact", "pair"),
    n_boot: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    recipes = per_recipe_metrics(rows)
    return {metric: bootstrap_mean_ci([row[metric] for row in recipes], n_boot, seed) for metric in metrics}


def paired_recipe_bootstrap(
    left: Iterable[Mapping[str, Any]], right: Iterable[Mapping[str, Any]],
    metric: str = "exact", n_boot: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """Left minus right on exactly matched questions/presentations and recipes.

    Reject unequal coverage instead of silently intersecting surviving outputs.
    The same recipe is resampled on both sides, preserving all nested questions.
    """
    left_rows, right_rows = _usable(left), _usable(right)
    validate_unique_predictions(left_rows)
    validate_unique_predictions(right_rows)
    left_map = {_prediction_key(row): row for row in left_rows}
    right_map = {_prediction_key(row): row for row in right_rows}
    if left_map.keys() != right_map.keys():
        raise ValueError("Paired comparison requires identical scored recipe/qid/display_perm coverage; technical missingness must be repaired or explicitly analyzed")
    if not left_map:
        raise ValueError("Paired comparison has no predictions")
    for key in left_map:
        if left_map[key]["target_order"] != right_map[key]["target_order"]:
            raise ValueError(f"Paired comparison target mismatch: {key}")
    left_recipes = {row["recipe_id"]: row for row in per_recipe_metrics(left_rows)}
    right_recipes = {row["recipe_id"]: row for row in per_recipe_metrics(right_rows)}
    differences = [left_recipes[recipe][metric] - right_recipes[recipe][metric] for recipe in sorted(left_recipes)]
    result = bootstrap_mean_ci(differences, n_boot, seed)
    result.update({"metric": metric, "direction": "left_minus_right",
                   "left_estimate": mean(row[metric] for row in left_recipes.values()),
                   "right_estimate": mean(row[metric] for row in right_recipes.values())})
    return result


def rank_position_matrix(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Target rank (row) × displayed input position (column), invalids wrong."""
    counts = [[0] * 4 for _ in range(4)]
    correct = [[0] * 4 for _ in range(4)]
    by_recipe: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in _usable(rows):
        by_recipe[str(row["recipe_id"])].append(row)
        for target_rank, semantic in enumerate(row["target_order"]):
            position = row["display_perm"].index(semantic)
            counts[target_rank][position] += 1
            correct[target_rank][position] += row["metrics"]["rank_correct"][target_rank]
    rates = [[correct[t][p] / counts[t][p] if counts[t][p] else None for p in range(4)] for t in range(4)]
    centered = [[value - mean(row) if value is not None and all(v is not None for v in row) else None for value in row] for row in rates]
    # Macro is identical to pooled rates for the planned one-complete-orbit per
    # recipe design; keep explicit macro output for any unequal future design.
    per_recipe_rates = []
    for group in by_recipe.values():
        local_n = [[0] * 4 for _ in range(4)]
        local_c = [[0] * 4 for _ in range(4)]
        for row in group:
            for rank, semantic in enumerate(row["target_order"]):
                position = row["display_perm"].index(semantic)
                local_n[rank][position] += 1
                local_c[rank][position] += row["metrics"]["rank_correct"][rank]
        per_recipe_rates.append([[local_c[t][p] / local_n[t][p] if local_n[t][p] else None for p in range(4)] for t in range(4)])
    macro = [[mean(matrix[t][p] for matrix in per_recipe_rates) if per_recipe_rates and all(matrix[t][p] is not None for matrix in per_recipe_rates) else None for p in range(4)] for t in range(4)]
    return {"correct_counts": correct, "exposure_counts": counts, "pooled_rates": rates,
            "pooled_row_centered": centered, "recipe_macro_rates": macro,
            "recipe_macro_row_centered": [[value - mean(row) if value is not None and all(v is not None for v in row) else None for value in row] for row in macro],
            "row_axis": "selected target rank 1..4", "column_axis": "display position A..D",
            "independent_unit": "recipe; individual matrix observations are not independent samples"}


def orbit_metrics(rows: Iterable[Mapping[str, Any]], require_complete: bool = True) -> dict[str, Any]:
    """Summarize full 24-display orbits, refusing incomplete/duplicate designs.

    Incomplete orbits may be listed for diagnostics with require_complete=False,
    but no 24-display statistic is computed for them. Invalid responses count
    toward the fixed denominator and never become a modal semantic candidate.
    """
    records = list(rows)
    usable = _usable(records)
    validate_unique_predictions(usable)
    by_question: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        by_question[str(row["recipe_id"]), str(row["qid"])].append(row)
    incomplete = []
    questions = []
    for (recipe, qid), group in sorted(by_question.items()):
        presentations = {tuple(row["display_perm"]) for row in group}
        if presentations != ALL_PERMUTATIONS or len(group) != 24:
            incomplete.append({"recipe_id": recipe, "qid": qid, "n_scored": len(group), "n_unique_displays": len(presentations)})
            continue
        if len({tuple(row["target_order"]) for row in group}) != 1:
            raise ValueError(f"Target semantic order changes within one orbit: {recipe}/{qid}")
        success = sum(row["metrics"]["exact"] for row in group)
        semantic_counts = Counter(tuple(row["metrics"]["semantic_order"]) for row in group if row["metrics"]["valid"])
        questions.append({"recipe_id": recipe, "qid": qid, "n_displays": 24,
                          "success_count": success, "mean_exact": success / 24,
                          "all24_exact": int(success == 24),
                          "modal_consistency": max(semantic_counts.values(), default=0) / 24,
                          "n_valid": sum(row["metrics"]["valid"] for row in group)})
    # Include all-technically-failed questions as incomplete rather than losing
    # their identities when filtering usable rows.
    usable_keys = set(by_question)
    failed_keys = {(str(row["recipe_id"]), str(row["qid"])) for row in records} - usable_keys
    incomplete.extend({"recipe_id": recipe, "qid": qid, "n_scored": 0, "n_unique_displays": 0} for recipe, qid in sorted(failed_keys))
    if incomplete and require_complete:
        raise ValueError(f"Incomplete 24-display orbit(s): {incomplete[:3]} ({len(incomplete)} total)")
    by_recipe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        by_recipe[question["recipe_id"]].append(question)
    names = ("mean_exact", "all24_exact", "modal_consistency", "success_count")
    macro = {name: mean(mean(question[name] for question in group) for group in by_recipe.values()) if by_recipe else None for name in names}
    return {"n_complete_questions": len(questions), "n_complete_recipes": len(by_recipe),
            "n_incomplete_questions": len(incomplete), "incomplete_questions": incomplete,
            "recipe_macro": macro, "per_question": questions,
            "success_count_histogram": {str(k): sum(question["success_count"] == k for question in questions) for k in range(25)}}


def dual_reward_groups(
    rows: Iterable[Mapping[str, Any]], group_key: str = "candidate_group_id",
    expected_group_size: int | None = None,
) -> dict[str, Any]:
    """Score an identical candidate bank under both rewards (E1 diagnostic)."""
    records = list(rows)
    if any(row.get("metrics") is None for row in records):
        raise ValueError("Candidate bank has technical failures; repair missing candidates before computing group variance")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if group_key not in row:
            raise ValueError(f"Candidate-bank row missing {group_key}")
        groups[str(row[group_key])].append(row)
    results = []
    categories = Counter()
    for key, group in sorted(groups.items()):
        if expected_group_size is not None and len(group) != expected_group_size:
            raise ValueError(f"Candidate group {key} has {len(group)} responses, expected {expected_group_size}")
        inputs = {(row["recipe_id"], row["qid"], tuple(row["display_perm"]), tuple(row["target_order"])) for row in group}
        if len(inputs) != 1:
            raise ValueError(f"Candidate group {key} mixes different inputs")
        exact = [row["metrics"]["exact"] for row in group]
        pair = [row["metrics"]["pair"] for row in group]
        exact_var = len(set(exact)) > 1
        pair_var = len(set(pair)) > 1
        if exact_var and not pair_var:
            raise AssertionError("Exact-only variation is impossible under the frozen rewards")
        category = "both_vary" if exact_var else "pair_only_varies" if pair_var else "neither_varies"
        categories[category] += 1
        results.append({"candidate_group_id": key, "n_candidates": len(group),
                        "category": category, "exact_mean": mean(exact), "pair_mean": mean(pair),
                        "exact_std_population": math.sqrt(mean((x - mean(exact)) ** 2 for x in exact)),
                        "pair_std_population": math.sqrt(mean((x - mean(pair)) ** 2 for x in pair))})
    return {"n_groups": len(results), "categories": {name: categories[name] for name in ("both_vary", "pair_only_varies", "neither_varies")},
            "per_group": results, "note": "Descriptive same-candidate comparison; availability is not evidence of downstream improvement."}
