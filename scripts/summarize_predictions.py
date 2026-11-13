#!/usr/bin/env python3
"""Recompute auditable E0-E3/E1 summaries from saved prediction JSONL files.

Examples (all remote output paths must be on the data disk):
    python scripts/summarize_predictions.py predictions.jsonl --output-dir results/e0
    python scripts/summarize_predictions.py orbit.jsonl --orbit --paired --output-dir results/e3
    python scripts/summarize_predictions.py bank.jsonl --candidate-bank --output-dir results/e1

Required prediction fields: recipe_id, qid, display_perm, raw_response.
Optional: target_order (default semantic [0,1,2,3]), presentation_id, run_id,
model_id, checkpoint, seed, technical_error. Candidate banks additionally need
candidate_group_id (one identical input per group). Technical failures may omit
raw_response, but must retain recipe_id/qid and a nonempty technical_error.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path
import sys

# Works directly from a checkout before editable package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recipeqa.metrics import (  # noqa: E402
    SCALAR_METRICS, aggregate_scores, bootstrap_mean_ci,
    dual_reward_groups, orbit_metrics, paired_recipe_bootstrap,
    per_recipe_metrics, rank_position_matrix, recipe_bootstrap_ci, score_record,
)


def read_jsonl(paths):
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise ValueError("each JSONL record must be an object")
                    yield score_record(raw)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{path}:{lineno}: {error}") from error


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    fields = fieldnames or sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-by", default="run_id,model_id,checkpoint,seed", help="Comma-separated model identity fields; missing fields become null")
    parser.add_argument("--orbit", action="store_true", help="Require complete 24-display orbits and compute E3 matrices")
    parser.add_argument("--allow-incomplete-orbits", action="store_true", help="Diagnostics only: list incomplete orbits, exclude them from orbit statistics")
    parser.add_argument("--candidate-bank", action="store_true", help="E1 same-candidate dual scoring instead of unique-prediction E0/E3 summaries")
    parser.add_argument("--candidate-group-size", type=int, default=4, help="Required number of responses per E1 candidate group (default frozen G=4)")
    parser.add_argument("--paired", action="store_true", help="Compute pairwise group differences only with exactly matched scored coverage")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261107)
    args = parser.parse_args(argv)
    if args.candidate_bank and (args.orbit or args.paired):
        parser.error("--candidate-bank cannot be combined with --orbit or --paired")
    if args.allow_incomplete_orbits and not args.orbit:
        parser.error("--allow-incomplete-orbits requires --orbit")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    group_fields = [field.strip() for field in args.group_by.split(",") if field.strip()]
    if not group_fields:
        parser.error("--group-by must specify at least one identity field")
    records = list(read_jsonl(args.predictions))
    if not records:
        parser.error("No prediction records found")
    groups = defaultdict(list)
    for row in records:
        key = tuple(json.dumps(row.get(field), sort_keys=True, ensure_ascii=False) for field in group_fields)
        groups[key].append(row)
    report = {"schema_version": 1, "source_files": [str(path.resolve()) for path in args.predictions],
              "group_fields": group_fields, "groups": [], "paired_comparisons": [],
              "uncertainty_note": "Bootstrap resamples independent recipes; it does not quantify training-seed uncertainty. No p-values or automatic significance claims are produced."}
    per_record = []
    per_recipe = []
    per_question_orbit = []
    matrices = []
    dual_groups = []
    group_lookup = []
    for index, (key, rows) in enumerate(sorted(groups.items()), 1):
        group_id = f"group_{index:03d}"
        identity = {field: json.loads(value) for field, value in zip(group_fields, key)}
        result = {"group_id": group_id, "identity": identity}
        if args.candidate_bank:
            result["candidate_bank"] = dual_reward_groups(rows, expected_group_size=args.candidate_group_size)
            result["n_technical_errors"] = sum(row["metrics"] is None for row in rows)
            dual_groups.extend({"group_id": group_id, **item} for item in result["candidate_bank"]["per_group"])
        else:
            result["scores"] = aggregate_scores(rows)
            if result["scores"]["n_scored"]:
                result["recipe_macro_ci95"] = recipe_bootstrap_ci(rows, ("exact", "pair", "middle_pair", "first_rank", "last_rank", "valid"), args.bootstrap_samples, args.bootstrap_seed)
            per_recipe.extend({"group_id": group_id, **item} for item in per_recipe_metrics(rows))
        if args.orbit:
            orbit = orbit_metrics(rows, require_complete=not args.allow_incomplete_orbits)
            result["orbit"] = orbit
            per_question_orbit.extend({"group_id": group_id, **item} for item in orbit["per_question"])
            complete_keys = {(item["recipe_id"], item["qid"]) for item in orbit["per_question"]}
            complete_rows = [row for row in rows if (row["recipe_id"], row["qid"]) in complete_keys]
            result["rank_position"] = rank_position_matrix(complete_rows)
            orbit_by_recipe = defaultdict(list)
            for item in orbit["per_question"]:
                orbit_by_recipe[item["recipe_id"]].append(item)
            if orbit_by_recipe:
                result["orbit_recipe_macro_ci95"] = {
                    metric: bootstrap_mean_ci([sum(item[metric] for item in items) / len(items) for items in orbit_by_recipe.values()], args.bootstrap_samples, args.bootstrap_seed)
                    for metric in ("mean_exact", "all24_exact", "modal_consistency")
                }
            for target_rank in range(4):
                for position in range(4):
                    matrix = result["rank_position"]
                    matrices.append({"group_id": group_id, "target_rank": target_rank + 1,
                                     "display_position": chr(ord("A") + position),
                                     "exposure_count": matrix["exposure_counts"][target_rank][position],
                                     "correct_count": matrix["correct_counts"][target_rank][position],
                                     "pooled_rate": matrix["pooled_rates"][target_rank][position],
                                     "pooled_row_centered": matrix["pooled_row_centered"][target_rank][position],
                                     "recipe_macro_rate": matrix["recipe_macro_rates"][target_rank][position],
                                     "recipe_macro_row_centered": matrix["recipe_macro_row_centered"][target_rank][position]})
        report["groups"].append(result)
        group_lookup.append((group_id, rows, result))
        for row in rows:
            flat = {"group_id": group_id, "recipe_id": row["recipe_id"], "qid": row["qid"],
                    "presentation_id": row.get("presentation_id"), "display_perm": row.get("display_perm"),
                    "target_order": row.get("target_order"), "raw_response": row.get("raw_response"),
                    "technical_error": row.get("technical_error")}
            if row["metrics"] is not None:
                flat.update({metric: row["metrics"][metric] for metric in SCALAR_METRICS})
                flat.update({name: row["metrics"][name] for name in ("parsed_order", "semantic_order", "inversions", "pair_bin")})
            per_record.append(flat)
    if args.paired:
        for (left_id, left, left_result), (right_id, right, right_result) in combinations(group_lookup, 2):
            comparison = {"left_group": left_id, "right_group": right_id, "direction": "left_minus_right"}
            try:
                comparison["metrics"] = {metric: paired_recipe_bootstrap(left, right, metric, args.bootstrap_samples, args.bootstrap_seed)
                                         for metric in ("exact", "pair", "middle_pair", "first_rank", "last_rank", "valid")}
                if args.orbit:
                    left_orbits = {(item["recipe_id"], item["qid"]): item for item in left_result["orbit"]["per_question"]}
                    right_orbits = {(item["recipe_id"], item["qid"]): item for item in right_result["orbit"]["per_question"]}
                    if left_orbits.keys() != right_orbits.keys() or not left_orbits:
                        raise ValueError("Paired orbit comparison requires identical complete-orbit coverage")
                    comparison["orbit_metrics"] = {}
                    for metric in ("mean_exact", "all24_exact", "modal_consistency"):
                        differences = defaultdict(list)
                        for key in left_orbits:
                            differences[key[0]].append(left_orbits[key][metric] - right_orbits[key][metric])
                        comparison["orbit_metrics"][metric] = bootstrap_mean_ci([sum(values) / len(values) for values in differences.values()], args.bootstrap_samples, args.bootstrap_seed)
                comparison["status"] = "computed"
            except ValueError as error:
                comparison["status"] = "not_computed"
                comparison["reason"] = str(error)
            report["paired_comparisons"].append(comparison)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "per_record.csv", per_record)
    write_csv(args.output_dir / "per_recipe.csv", per_recipe, ["group_id", "recipe_id", "n_predictions", *SCALAR_METRICS])
    if args.orbit:
        write_csv(args.output_dir / "orbit_per_question.csv", per_question_orbit)
        write_csv(args.output_dir / "rank_position_matrix.csv", matrices)
    if args.candidate_bank:
        write_csv(args.output_dir / "dual_reward_groups.csv", dual_groups)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "n_records": len(records), "n_groups": len(groups),
                      "paired_not_computed": sum(item["status"] != "computed" for item in report["paired_comparisons"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
