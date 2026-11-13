#!/usr/bin/env python3
"""E4 automatic diagnostics and blinded F/D/S review, without fabricated labels.

Example:
  python scripts/summarize_process.py --manifest manifests/process.jsonl \
    --predictions runs/sft/process.jsonl runs/pair/process.jsonl \
    --answer-only runs/sft/test.jsonl runs/pair/test.jsonl --output-dir results/e4

Import reviews with --labels blinded/labels_reviewer_1.csv ... and optionally
--consensus-labels consensus.csv. Only share output-dir/blinded with reviewers;
blind_mapping.private.json and statistical outputs disclose model identities.
Public JSON uses anonymous image asset IDs. --embed-images --image-root DIR
embeds actual image bytes without exposing source filenames. No website is made.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import csv
import hashlib
from itertools import combinations
import json
import mimetypes
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recipeqa.metrics import bootstrap_mean_ci  # noqa: E402
from recipeqa.process_metrics import (  # noqa: E402
    CRITERIA, LABELS, process_key, score_process_record, summarize_automatic,
    summarize_human, validate_human_labels,
)


def read_jsonl(paths):
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("JSONL row is not an object")
                    rows.append(row)
                except ValueError as error:
                    raise ValueError(f"{path}:{lineno}: {error}") from error
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict, tuple)) else value for key, value in row.items()})


def group_key(row, fields):
    return tuple(json.dumps(row.get(field), ensure_ascii=False, sort_keys=True) for field in fields)


def read_labels(paths, consensus=False):
    rows = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if consensus:
                    row["rater_id"] = "consensus"
                rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--answer-only", type=Path, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, nargs="*", default=[])
    parser.add_argument("--consensus-labels", type=Path, nargs="*", default=[])
    parser.add_argument("--raters", default="reviewer_1,reviewer_2")
    parser.add_argument("--group-by", default="run_id,model_id,checkpoint,seed")
    parser.add_argument("--blind-seed", type=int, default=20261108)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261107)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--embed-images", action="store_true")
    args = parser.parse_args(argv)
    if args.embed_images and args.image_root is None:
        parser.error("--embed-images requires --image-root")
    raters = [rater.strip() for rater in args.raters.split(",") if rater.strip()]
    if len(raters) != 2 or len(set(raters)) != 2 or "consensus" in raters:
        parser.error("--raters requires exactly two distinct reviewer IDs, separate from consensus")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    fields = [field.strip() for field in args.group_by.split(",") if field.strip()]
    if not fields:
        parser.error("--group-by cannot be empty")
    manifest_rows = read_jsonl([args.manifest])
    manifest = {process_key(row): row for row in manifest_rows}
    if not manifest or len(manifest) != len(manifest_rows):
        parser.error("Manifest must be nonempty with no duplicate recipe/qid/display")
    if len({str(row["recipe_id"]) for row in manifest_rows}) != len(manifest_rows):
        parser.error("E4 manifest requires exactly one question per independent recipe")
    expected_n = len(manifest_rows)
    predictions = read_jsonl(args.predictions)
    if not predictions:
        parser.error("No process predictions found")
    groups = defaultdict(dict)
    attempts = []
    for row in predictions:
        key = process_key(row)
        if key not in manifest:
            raise ValueError(f"Prediction outside fixed process manifest: {key}")
        if row.get("prompt_kind") not in (None, "process"):
            raise ValueError("--predictions includes an answer-only prompt; use --answer-only")
        model = group_key(row, fields)
        previous = groups[model].get(key)
        if previous is not None:
            if not previous.get("technical_error") and not row.get("technical_error"):
                raise ValueError(f"Duplicate successful process prediction: {key}")
            attempts.append({"identity": model, "key": key, "technical_error": previous.get("technical_error") or row.get("technical_error")})
            if not previous.get("technical_error"):
                continue
        groups[model][key] = row
    answer_groups = defaultdict(dict)
    for row in read_jsonl(args.answer_only):
        key = process_key(row)
        if key not in manifest:
            continue  # Full E0 files include many questions outside the fixed E4 subset.
        if row.get("prompt_kind") not in (None, "answer"):
            raise ValueError("--answer-only includes a process prompt")
        model = group_key(row, fields)
        previous = answer_groups[model].get(key)
        if previous is not None and not previous.get("technical_error") and not row.get("technical_error"):
            raise ValueError(f"Duplicate successful answer-only prediction: {key}")
        if previous is None or previous.get("technical_error"):
            answer_groups[model][key] = row
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public_dir = args.output_dir / "blinded"
    public_dir.mkdir(exist_ok=True)
    private_path = args.output_dir / "blind_mapping.private.json"
    previous_mapping = json.loads(private_path.read_text()) if private_path.exists() else {"entries": []}
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if previous_mapping.get("source_manifest_sha256") not in (None, manifest_hash):
        raise ValueError("Blinded manifest changed; use a new output directory to protect existing annotations")
    mapping_by_key = {entry["private_key"]: entry for entry in previous_mapping["entries"]}
    rng = random.Random(args.blind_seed)
    used_ids = {entry["blind_id"] for entry in previous_mapping["entries"]}
    scored_groups = []
    public = []
    private = []
    for index, (model, items) in enumerate(sorted(groups.items()), 1):
        group_id = f"group_{index:03d}"
        identity = {field: json.loads(value) for field, value in zip(fields, model)}
        scored = []
        for key, prediction in sorted(items.items()):
            row = score_process_record(prediction, manifest[key], answer_groups[model].get(key))
            private_key = json.dumps([list(model), key], ensure_ascii=False)
            response_hash = hashlib.sha256(prediction.get("raw_response", "").encode()).hexdigest()
            existing = mapping_by_key.get(private_key)
            if existing and existing["response_sha256"] != response_hash:
                raise ValueError("A previously blinded response changed; use a new run_id/output directory to avoid reusing old annotations")
            blind_id = existing["blind_id"] if existing else f"E4-{rng.getrandbits(64):016x}"
            while blind_id in used_ids and not existing:
                blind_id = f"E4-{rng.getrandbits(64):016x}"
            used_ids.add(blind_id)
            row["blind_id"] = blind_id
            row["group_id"] = group_id
            scored.append(row)
            recipe = manifest[key]
            assets = []
            for label, relative_path in zip(LABELS, recipe.get("image_paths_display", [])):
                asset = {"image_label": label, "asset_id": f"{blind_id}-{label}"}
                if args.embed_images:
                    root = args.image_root.resolve()
                    path = (root / relative_path).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError("Image path escapes --image-root")
                    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    asset["data_url"] = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
                assets.append(asset)
            public.append({"blind_id": blind_id, "recipe_text": "\n\n".join(
                f"Step {ordinal}: {step.get('title', '')}\n{step.get('body', '')}" for ordinal, step in enumerate(recipe["steps"], 1)),
                "images": assets, "raw_response": prediction.get("raw_response", ""),
                "image_assets_status": "embedded" if args.embed_images else "opaque IDs only; image paths are in the private mapping for blind asset preparation"})
            private.append({"private_key": private_key, "blind_id": blind_id, "group_id": group_id,
                            "identity": identity, "recipe_id": key[0], "qid": key[1],
                            "display_perm": key[2], "response_sha256": response_hash,
                            "image_paths_display": recipe.get("image_paths_display", []),
                            "gold_display_step_ordinals": row["gold_display_step_ordinals"]})
        scored_groups.append((group_id, identity, scored))
    all_ids = {entry["blind_id"] for entry in private}
    if set(mapping_by_key) - {entry["private_key"] for entry in private}:
        raise ValueError("Previously blinded responses disappeared; provide all original prediction groups or use a new output directory")
    raw_labels = validate_human_labels(read_labels(args.labels), all_ids)
    raw_consensus = validate_human_labels(read_labels(args.consensus_labels, consensus=True), all_ids)
    report = {"schema_version": 1, "expected_responses_per_method": expected_n,
              "expected_images_per_method": 4 * expected_n, "groups": [], "prior_technical_attempts": attempts,
              "paired_comparisons": [],
              "source_manifest": str(args.manifest.resolve()),
              "source_manifest_sha256": manifest_hash,
              "human_rubric": {"F": "fact visible and accurate", "D": "step-discriminative evidence", "S": "supports claimed mapping/order",
                  "image_Q": "F=yes and D=yes", "response_E": "all four F=yes and at least two D=yes",
                  "joint": "response E and process Order correct", "missing_model_fields": "effective F/D/S=unclear; raw labels retained",
                  "missing_human_labels": "pending, never invented", "scope": "explicit output quality; not hidden reasoning faithfulness"}}
    human_per_response = []
    for group_id, identity, scored in scored_groups:
        ids = {row["blind_id"] for row in scored}
        automatic = summarize_automatic(scored, expected_n)
        human = summarize_human(scored, [row for row in raw_labels if row["blind_id"] in ids], expected_n, raters)
        consensus = summarize_human(scored, [row for row in raw_consensus if row["blind_id"] in ids], expected_n, ("consensus",))
        result = {"group_id": group_id, "identity": identity, "automatic": automatic,
                  "human_independent": human, "human_consensus_separate": consensus}
        if automatic["status"] == "complete":
            result["automatic_recipe_ci95"] = {metric: bootstrap_mean_ci([row["process_metrics"][metric] for row in scored], args.bootstrap_samples, args.bootstrap_seed)
                                                for metric in ("exact", "step_match_mean", "all_steps_match", "mapping_order_consistent")}
        for category, summary in (("independent", human), ("consensus", consensus)):
            for rater, data in summary["reviewers"].items():
                if data["status"] == "complete":
                    data["recipe_ci95"] = {metric: bootstrap_mean_ci([row[metric] for row in data["per_response"]], args.bootstrap_samples, args.bootstrap_seed) for metric in ("E", "joint")}
                for item in data["per_response"]:
                    human_per_response.append({"group_id": group_id, "category": category, "rater_id": rater,
                                               "blind_id": item["blind_id"], "annotation_complete": item["annotation_complete"],
                                               "E": item["E"] if item["annotation_complete"] else None,
                                               "joint": item["joint"] if item["annotation_complete"] else None,
                                               "n_unfilled_human_values": item["n_unfilled_human_values"],
                                               "n_model_missing_image_fields": item["n_model_missing_image_fields"]})
        report["groups"].append(result)
    for (left_id, _, left_rows), (right_id, _, right_rows) in combinations(scored_groups, 2):
        comparison = {"left_group": left_id, "right_group": right_id, "direction": "left_minus_right",
                      "automatic": {"status": "pending_predictions"}, "human": {}}
        left_result = next(group for group in report["groups"] if group["group_id"] == left_id)
        right_result = next(group for group in report["groups"] if group["group_id"] == right_id)
        left_by_key = {process_key(row): row for row in left_rows}
        right_by_key = {process_key(row): row for row in right_rows}
        if left_result["automatic"]["status"] == "complete" and right_result["automatic"]["status"] == "complete":
            keys = sorted(manifest)
            comparison["automatic"] = {"status": "complete", "metrics": {
                metric: bootstrap_mean_ci([left_by_key[key]["process_metrics"][metric] - right_by_key[key]["process_metrics"][metric] for key in keys],
                                          args.bootstrap_samples, args.bootstrap_seed)
                for metric in ("exact", "step_match_mean", "all_steps_match", "mapping_order_consistent")}}
        for category, name in (("independent", "human_independent"), ("consensus", "human_consensus_separate")):
            for rater in left_result[name]["reviewers"]:
                left_human = left_result[name]["reviewers"][rater]
                right_human = right_result[name]["reviewers"][rater]
                item = {"status": "pending"}
                if left_human["status"] == "complete" and right_human["status"] == "complete":
                    left_values = {row["blind_id"]: row for row in left_human["per_response"]}
                    right_values = {row["blind_id"]: row for row in right_human["per_response"]}
                    item = {"status": "complete", "metrics": {metric: bootstrap_mean_ci([
                        left_values[left_by_key[key]["blind_id"]][metric] - right_values[right_by_key[key]["blind_id"]][metric]
                        for key in sorted(manifest)], args.bootstrap_samples, args.bootstrap_seed) for metric in ("E", "joint")}}
                comparison["human"][f"{category}:{rater}"] = item
        report["paired_comparisons"].append(comparison)
    write_json(private_path, {"warning": "PRIVATE: do not share with reviewers", "source_manifest_sha256": manifest_hash, "entries": private})
    os.chmod(private_path, 0o600)
    rng.shuffle(public)
    (public_dir / "review_items.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public), encoding="utf-8")
    template_fields = ["blind_id", "image_label", "rater_id", *CRITERIA, "unsupported_types", "notes"]
    for rater in (*raters, "consensus"):
        # Never overwrite filled review sheets on a repeated analysis run.
        template = public_dir / f"labels_{rater}.template.csv"
        items_to_export = public
        if template.exists():
            with template.open(encoding="utf-8-sig", newline="") as stream:
                existing_ids = {row["blind_id"] for row in csv.DictReader(stream)}
            items_to_export = [item for item in public if item["blind_id"] not in existing_ids]
            suffix = hashlib.sha256(json.dumps(sorted(item["blind_id"] for item in items_to_export)).encode()).hexdigest()[:8]
            template = public_dir / f"labels_{rater}.additional-{suffix}.template.csv"
        if items_to_export and not template.exists():
            write_csv(template, [{"blind_id": item["blind_id"], "image_label": label, "rater_id": rater,
                                  **{criterion: "" for criterion in CRITERIA}, "unsupported_types": "", "notes": ""}
                                 for item in items_to_export for label in LABELS], template_fields)
    write_json(args.output_dir / "summary.json", report)
    (args.output_dir / "scored_process.private.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for _, _, rows in scored_groups for row in rows), encoding="utf-8")
    write_csv(args.output_dir / "human_per_response.csv", human_per_response,
              ["group_id", "category", "rater_id", "blind_id", "annotation_complete", "E", "joint", "n_unfilled_human_values", "n_model_missing_image_fields"])
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "n_methods": len(scored_groups),
                      "expected_n_per_method": expected_n, "blinded_responses": len(public),
                      "human_status": "complete" if all(group["human_independent"]["status"] == "complete" for group in report["groups"]) else "pending"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
