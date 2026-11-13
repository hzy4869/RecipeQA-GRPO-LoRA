"""Independent E4 parser, source-step alignment, and blinded-human summaries.

The only use of the main answer parser is on the payload of a unique anchored
``Order:`` field. A process response is never passed wholesale to that parser.
Human evidence labels measure explicit visible support, not hidden-CoT fidelity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from recipeqa.metrics import parse_order, score_order

LABELS = ("A", "B", "C", "D")
CRITERIA = ("F", "D", "S")
HUMAN_VALUES = {"yes", "no", "unclear"}


def process_key(record: Mapping[str, Any]) -> tuple[str, str, tuple[int, ...]]:
    return str(record["recipe_id"]), str(record["qid"]), tuple(record["display_perm"])


def parse_process(text: str | None) -> dict[str, Any]:
    """Parse exactly five nonblank fields, retaining independent failure flags.

    A response can have a valid final Order but missing image fields. Final
    answer validity and complete-format validity are deliberately reported
    separately. Duplicated image fields are unusable; duplicated Order fields
    invalidate the final answer. No Markdown/prose repair or substring search.
    """
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()] if isinstance(text, str) else []
    fields: dict[str, list[dict[str, Any]]] = defaultdict(list)
    orders = []
    observed_fields = []
    errors = []
    for line in lines:
        image = re.fullmatch(r"([A-D]):\s*step\s+([1-9]\d*|unclear)\s*\|\s*(.*)", line, flags=re.I)
        order = re.fullmatch(r"Order:\s*(.*)", line, flags=re.I)
        if image:
            label, step, observation = image.groups()
            label = label.upper()
            fields[label].append({"step": int(step) if step.lower() != "unclear" else None,
                                  "step_unclear": step.lower() == "unclear",
                                  "observation": observation.strip()})
            observed_fields.append(label)
        elif order:
            orders.append(order.group(1))
            observed_fields.append("Order")
        else:
            errors.append({"type": "unrecognized_line", "line": line})
    images = {}
    for label in LABELS:
        candidates = fields[label]
        if len(candidates) == 1:
            field = dict(candidates[0])
            field.update({"present": True, "complete": bool(field["observation"]), "duplicate": False})
            if not field["observation"]:
                errors.append({"type": "missing_observation", "label": label})
        else:
            field = {"step": None, "step_unclear": False, "observation": "", "present": False,
                     "complete": False, "duplicate": len(candidates) > 1}
            errors.append({"type": "duplicate_image_field" if candidates else "missing_image_field", "label": label})
        images[label] = field
    parsed_order = parse_order(orders[0]) if len(orders) == 1 else None
    if len(orders) != 1:
        errors.append({"type": "duplicate_order_field" if orders else "missing_order_field"})
    elif parsed_order is None:
        errors.append({"type": "invalid_order_payload"})
    if observed_fields != [*LABELS, "Order"]:
        errors.append({"type": "field_order_or_count"})
    return {"images": images, "order": list(parsed_order) if parsed_order is not None else None,
            "order_valid": parsed_order is not None, "format_valid": not errors,
            "n_missing_image_fields": sum(not field["present"] for field in images.values()),
            "n_missing_observations": sum(not field["observation"] for field in images.values()),
            "n_unclear_steps": sum(field["step_unclear"] for field in images.values()),
            "errors": errors}


def gold_display_steps(manifest: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    """Convert original source IDs to the actual 1-based prompt ordinals.

    ``modeling._render_recipe`` numbers context steps by enumerate(..., 1).
    Original annotation IDs may start at zero or have gaps, so comparing a
    predicted step directly with a source ID would be an off-by-one/data error.
    """
    steps = manifest.get("steps")
    source = manifest.get("source_step_ids_chronological")
    if not isinstance(steps, list) or not steps or not isinstance(source, (list, tuple)) or len(source) != 4:
        raise ValueError("E4 needs manifest steps and four source_step_ids_chronological")
    source_ids = [step["source_step_id"] for step in steps]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Duplicate context source step IDs")
    ordinal = {source_id: index + 1 for index, source_id in enumerate(source_ids)}
    if len(set(source)) != 4 or not set(source).issubset(ordinal):
        raise ValueError("Four image source IDs must be distinct and present in context")
    display = tuple(manifest["display_perm"])
    # Reuse the validated permutation boundary without any model output.
    score_order(None, display, manifest.get("target_order", (0, 1, 2, 3)))
    chronological_ordinals = [ordinal[source_id] for source_id in source]
    return [chronological_ordinals[semantic] for semantic in display], chronological_ordinals


def score_process_record(
    prediction: Mapping[str, Any], manifest: Mapping[str, Any],
    answer_only: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if process_key(prediction) != process_key(manifest):
        raise ValueError("Process prediction does not match manifest recipe/qid/display")
    target = tuple(manifest.get("target_order", (0, 1, 2, 3)))
    if tuple(prediction.get("target_order", target)) != target:
        raise ValueError("Process prediction target differs from manifest")
    gold_display, gold_chronological = gold_display_steps(manifest)
    if prediction.get("source_step_ids_chronological") is not None and list(prediction["source_step_ids_chronological"]) != list(manifest["source_step_ids_chronological"]):
        raise ValueError("Process prediction source step IDs differ from manifest")
    technical = prediction.get("technical_error") or ("unspecified technical error" if prediction.get("status") == "technical_error" else None)
    result = dict(prediction)
    result["gold_display_step_ordinals"] = gold_display
    result["gold_chronological_step_ordinals"] = gold_chronological
    if technical:
        result.update({"technical_error": technical, "process_metrics": None, "process_parsed": None})
        return result
    if not isinstance(prediction.get("raw_response"), str):
        raise ValueError("Nontechnical E4 prediction needs raw_response string")
    parsed = parse_process(prediction["raw_response"])
    order_score = score_order(parsed["order"], manifest["display_perm"], target)
    fields = [parsed["images"][label] for label in LABELS]
    predicted_steps = [field["step"] for field in fields]
    matches = [int(field["present"] and field["step"] == gold) for field, gold in zip(fields, gold_display)]
    in_range = [field["step"] is not None and 1 <= field["step"] <= len(manifest["steps"]) for field in fields]
    complete_mapping = all(field["present"] for field in fields) and all(in_range)
    unique_mapping = complete_mapping and len(set(predicted_steps)) == 4
    mapping_sorted = tuple(sorted(range(4), key=lambda display: predicted_steps[display])) if unique_mapping else None
    metrics: dict[str, Any] = {
        "format_valid": int(parsed["format_valid"]), "order_valid": order_score["valid"],
        "exact": order_score["exact"], "pair": order_score["pair"],
        "semantic_order": order_score["semantic_order"],
        "step_match_per_image": matches, "step_match_mean": mean(matches),
        "all_steps_match": int(all(matches)),
        "complete_mapping": int(complete_mapping), "unique_mapping": int(unique_mapping),
        "mapping_order_comparable": int(unique_mapping and parsed["order_valid"]),
        "mapping_order_consistent": int(unique_mapping and tuple(parsed["order"] or ()) == mapping_sorted),
        "n_missing_image_fields": parsed["n_missing_image_fields"],
        "n_missing_observations": parsed["n_missing_observations"],
        "n_unclear_steps": parsed["n_unclear_steps"],
        "n_out_of_range_steps": sum(field["step"] is not None and not valid for field, valid in zip(fields, in_range)),
        "answer_only_available": False, "answer_changed_both_valid": None,
        "answer_changed_or_validity_changed": None, "answer_only_exact": None,
    }
    if answer_only is not None:
        if process_key(answer_only) != process_key(manifest):
            raise ValueError("Answer-only comparison does not match process manifest")
        if tuple(answer_only.get("target_order", target)) != target:
            raise ValueError("Answer-only target differs from process manifest")
        if not answer_only.get("technical_error") and answer_only.get("status") != "technical_error":
            if not isinstance(answer_only.get("raw_response"), str):
                raise ValueError("Answer-only comparison missing raw_response")
            original = score_order(answer_only["raw_response"], manifest["display_perm"], target)
            both_valid = bool(order_score["valid"] and original["valid"])
            metrics.update({"answer_only_available": True, "answer_only_exact": original["exact"],
                            "answer_only_order_valid": original["valid"],
                            "answer_both_valid": int(both_valid),
                            "answer_changed_both_valid": int(order_score["semantic_order"] != original["semantic_order"]) if both_valid else None,
                            "answer_changed_or_validity_changed": int(order_score["semantic_order"] != original["semantic_order"]),
                            "exact_change_vs_answer_only": order_score["exact"] - original["exact"]})
    result.update({"process_parsed": parsed, "process_metrics": metrics})
    return result


def summarize_automatic(records: Sequence[Mapping[str, Any]], expected_n: int) -> dict[str, Any]:
    if expected_n <= 0 or len(records) > expected_n:
        raise ValueError("Expected E4 denominator must be positive and at least record count")
    observed = [record for record in records if record.get("process_metrics") is not None]
    complete = len(observed) == expected_n
    names = ("format_valid", "order_valid", "exact", "pair", "step_match_mean", "all_steps_match",
             "complete_mapping", "unique_mapping", "mapping_order_consistent")
    rates = {name: sum(record["process_metrics"][name] for record in observed) / expected_n if complete else None for name in names}
    comparable = [record["process_metrics"] for record in observed if record["process_metrics"]["mapping_order_comparable"]]
    compared_answers = [record["process_metrics"] for record in observed if record["process_metrics"]["answer_only_available"]]
    both_valid = [item for item in compared_answers if item["answer_changed_both_valid"] is not None]
    return {"status": "complete" if complete else "pending_predictions", "expected_responses": expected_n,
            "expected_images": 4 * expected_n, "n_scored_responses": len(observed),
            "n_technical_errors": sum(record.get("process_metrics") is None for record in records),
            "n_missing_predictions": expected_n - len(records), "response_rates": rates,
            "observed_numerators": {name: sum(record["process_metrics"][name] for record in observed) for name in names},
            "image_counts": {name: sum(record["process_metrics"][name] for record in observed) for name in (
                "n_missing_image_fields", "n_missing_observations", "n_unclear_steps", "n_out_of_range_steps")},
            "mapping_order_comparable_n": len(comparable),
            "mapping_order_consistent_given_comparable": mean(item["mapping_order_consistent"] for item in comparable) if comparable else None,
            "answer_only_comparison": {"n_available": len(compared_answers), "n_both_valid": len(both_valid),
                "changed_rate_both_valid": mean(item["answer_changed_both_valid"] for item in both_valid) if both_valid else None,
                "changed_or_validity_changed_rate_available": mean(item["answer_changed_or_validity_changed"] for item in compared_answers) if compared_answers else None,
                "exact_change_fixed_denominator": sum(item["exact_change_vs_answer_only"] for item in compared_answers) / expected_n if complete and len(compared_answers) == expected_n else None},
            "interpretation": "Source-step agreement and explicit formatting are behavioral diagnostics. Order is parsed independently of complete-format validity; these rates are not answer-only-format accuracy or hidden-reasoning faithfulness."}


def validate_human_labels(rows: Iterable[Mapping[str, Any]], allowed_blind_ids: set[str]) -> list[dict[str, Any]]:
    normalized = []
    seen = set()
    for original in rows:
        row = dict(original)
        blind_id = str(row.get("blind_id", "")).strip()
        label = str(row.get("image_label", "")).strip().upper()
        rater = str(row.get("rater_id", "")).strip()
        if blind_id not in allowed_blind_ids or label not in LABELS or not rater:
            raise ValueError("Human labels need a known blind_id, A-D image_label, and rater_id")
        key = blind_id, label, rater
        if key in seen:
            raise ValueError(f"Duplicate human label: {key}")
        seen.add(key)
        row.update({"blind_id": blind_id, "image_label": label, "rater_id": rater})
        for criterion in CRITERIA:
            raw_value = row.get(criterion)
            value = str(raw_value).strip().lower() if raw_value is not None else ""
            if value and value not in HUMAN_VALUES:
                raise ValueError(f"Human {criterion} must be yes/no/unclear or empty pending: {key}")
            row[criterion] = value or None
        normalized.append(row)
    return normalized


def human_response_score(process_record: Mapping[str, Any], image_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply predeclared F/D/S thresholds; preserve raw vs effective missingness."""
    values = {}
    n_unfilled = 0
    n_model_missing = 0
    for label in LABELS:
        annotation = image_rows.get(label, {})
        raw = {criterion: annotation.get(criterion) for criterion in CRITERIA}
        n_unfilled += sum(value is None for value in raw.values())
        field = process_record["process_parsed"]["images"][label] if process_record.get("process_parsed") else None
        model_missing = field is None or not field["complete"]
        n_model_missing += model_missing
        effective = {criterion: "unclear" if model_missing else (raw[criterion] or "pending") for criterion in CRITERIA}
        values[label] = {"raw": raw, "effective": effective, "model_field_missing": model_missing,
                         "Q": None if "pending" in (effective["F"], effective["D"]) else int(effective["F"] == "yes" and effective["D"] == "yes")}
    evidence = int(all(values[label]["effective"]["F"] == "yes" for label in LABELS)
                   and sum(values[label]["effective"]["D"] == "yes" for label in LABELS) >= 2)
    ready = n_unfilled == 0
    joint = int(evidence and bool(process_record["process_metrics"]["exact"])) if process_record.get("process_metrics") else None
    return {"E": evidence if ready else None, "joint": joint if ready else None,
            "n_unfilled_human_values": n_unfilled, "n_model_missing_image_fields": n_model_missing,
            "annotation_complete": n_unfilled == 0, "images": values}


def summarize_human(
    records: Sequence[Mapping[str, Any]], label_rows: Sequence[Mapping[str, Any]],
    expected_n: int, raters: Sequence[str] = ("reviewer_1", "reviewer_2"),
) -> dict[str, Any]:
    """Separate each raw reviewer from consensus; do not pool them as samples."""
    by_id = {record["blind_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate blinded response ID")
    normalized = validate_human_labels(label_rows, set(by_id))
    requested_raters = list(raters)
    unknown = {row["rater_id"] for row in normalized} - set(requested_raters)
    if unknown:
        raise ValueError(f"Unexpected rater IDs: {sorted(unknown)}")
    lookup = {(row["rater_id"], row["blind_id"], row["image_label"]): row for row in normalized}
    results = {}
    for rater in requested_raters:
        response_rows = []
        counts = {criterion: Counter() for criterion in CRITERIA}
        raw_counts = {criterion: Counter() for criterion in CRITERIA}
        for blind_id, process in by_id.items():
            scored = human_response_score(process, {label: lookup.get((rater, blind_id, label), {}) for label in LABELS})
            response_rows.append({"blind_id": blind_id, **scored})
            for value in scored["images"].values():
                for criterion in CRITERIA:
                    counts[criterion][value["effective"][criterion]] += 1
                    raw_counts[criterion][value["raw"][criterion] or "missing"] += 1
        n_complete = sum(row["annotation_complete"] for row in response_rows)
        ready = len(records) == expected_n and n_complete == expected_n and all(record.get("process_metrics") is not None for record in records)
        e_sum = sum(row["E"] or 0 for row in response_rows)
        joint_sum = sum(row["joint"] or 0 for row in response_rows)
        q_sum = sum(image["Q"] or 0 for response in response_rows for image in response["images"].values())
        results[rater] = {"status": "complete" if ready else "pending", "expected_responses": expected_n,
                         "expected_images": 4 * expected_n, "n_fully_annotated_responses": n_complete,
                         "E_rate": e_sum / expected_n if ready else None,
                         "joint_rate": joint_sum / expected_n if ready else None,
                         "image_Q_rate": q_sum / (4 * expected_n) if ready else None,
                         "image_S_yes_rate": counts["S"]["yes"] / (4 * expected_n) if ready else None,
                         "image_criterion_rates": {criterion: {value: counts[criterion][value] / (4 * expected_n) if ready else None for value in sorted(HUMAN_VALUES)} for criterion in CRITERIA},
                         "raw_label_counts": {criterion: {value: raw_counts[criterion][value] for value in (*sorted(HUMAN_VALUES), "missing")} for criterion in CRITERIA},
                         "effective_label_counts_model_missing_as_unclear": {criterion: {value: counts[criterion][value] for value in (*sorted(HUMAN_VALUES), "pending")} for criterion in CRITERIA},
                         "per_response": response_rows}
    agreement = {}
    if len(requested_raters) == 2:
        left, right = requested_raters
        for criterion in CRITERIA:
            paired = []
            for blind_id in by_id:
                for label in LABELS:
                    a = lookup.get((left, blind_id, label), {}).get(criterion)
                    b = lookup.get((right, blind_id, label), {}).get(criterion)
                    if a is not None and b is not None:
                        paired.append((a, b))
            n_agree = sum(a == b for a, b in paired)
            agreement[criterion] = {"n_paired_image_judgments": len(paired), "n_expected_image_pairs": 4 * expected_n,
                                    "n_agree": n_agree, "n_disagree": len(paired) - n_agree,
                                    "raw_agreement_available_pairs": n_agree / len(paired) if paired else None}
    return {"status": "complete" if results and all(row["status"] == "complete" for row in results.values()) else "pending",
            "reviewers": results, "raw_agreement": agreement,
            "raw_labels": normalized, "note": "Each method denominator is expected N responses or 4N images. Reviewers remain separate; consensus must be imported separately. Missing human labels are pending, never fabricated negative judgments."}
