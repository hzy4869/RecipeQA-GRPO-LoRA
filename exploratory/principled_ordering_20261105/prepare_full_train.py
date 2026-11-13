#!/usr/bin/env python3
"""Prepare every official training ordering question; no GPU, test access or training.

Default output: full_train_data/train_all7558.jsonl alongside immutable audit
receipts. --check-source is read-only and does not index/decode/hash image files.
This scales training data; it is not evidence for an algorithmic improvement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_SHA = "331f5fd153a01fad0c834976f2ff79665ffdffbc7cbe44a7a1c36c25ae1476f2"
DATA_CODE_SHA = "7df1c7a162d10f952e76c47a06f161a8af79e96dc2a1aca881bcb4caa2e34418"
FROZEN_SHA = "9089f48c5f3b571dfa5fddd31b843a0ae2c22296d9ca4a4d03ee05ab7a75f18c"
OLD_TRAIN_SHA = "d2da936d697212dce5e1fef2fde13597eee504d356049d421dbb8f06579e0f7f"
ORDER_SEED = 20261114
PRESENTATION_SEED = 20261115
PRESENTATION_NAMESPACE = "principled_full_train_all7558"
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def inside(value):
    path = Path(value).resolve()
    require(path.is_relative_to(ROOT), f"Path escaped project data root: {path}")
    return path


def independent_choice_audit(rows):
    """Check raw options independently of canonicalize_row; retain every issue."""
    audit, problems, seen = [], [], set()
    for source_index, row in enumerate(rows):
        if row.get("task") != "visual_ordering":
            continue
        item = {"source_row_index": source_index, "source_ordering_index": len(audit),
                "qid": str(row.get("qid", "")), "recipe_id": str(row.get("recipe_id", "")),
                "official_choices_images": row.get("choice_list"), "official_answer_index": row.get("answer"),
                "backend_only_not_model_visible": True}
        try:
            require(item["qid"] and item["recipe_id"] and item["qid"] not in seen,
                    "Missing identity or duplicate question ID")
            seen.add(item["qid"])
            choices, answer = row["choice_list"], row["answer"]
            require(isinstance(choices, list) and len(choices) == 4 and type(answer) is int and 0 <= answer < 4,
                    "Expected four original options and an integer answer index")
            require(all(isinstance(option, list) and len(option) == 4
                        and all(isinstance(name, str) and Path(name).name == name and "\\" not in name for name in option)
                        and len(set(option)) == 4 for option in choices), "Malformed option images")
            names = set(choices[0])
            require(all(set(option) == names for option in choices), "Options use different images")
            # Filename step extraction is independent of frozen source_image_step.
            step_by_name = {name: int(Path(name).stem.rsplit("_", 2)[1]) for name in names}
            require(len(set(step_by_name.values())) == 4, "Photos do not come from four distinct steps")
            chronological = sorted(names, key=step_by_name.get)
            permutations = [[chronological.index(name) for name in option] for option in choices]
            require(all(sorted(option) == [0, 1, 2, 3] for option in permutations), "An option is not a permutation")
            require(permutations[answer] == [0, 1, 2, 3]
                    and sum(option == [0, 1, 2, 3] for option in permutations) == 1,
                    "Official answer is not the unique chronological option")
            context_ids = [int(step["id"]) for step in row["context"]]
            require(context_ids == sorted(set(context_ids)) and set(step_by_name.values()) <= set(context_ids),
                    "Full recipe context is unordered or misses source steps")
            item.update(passed=True, images_chronological=chronological,
                        source_step_ids_chronological=[step_by_name[name] for name in chronological],
                        choice_semantic_permutations=permutations,
                        duplicate_incorrect_options=4 - len({tuple(option) for option in permutations}))
        except (ValueError, TypeError, KeyError, IndexError) as error:
            item.update(passed=False, error=f"{type(error).__name__}: {error}")
            problems.append({key: item[key] for key in ("source_row_index", "qid", "recipe_id", "error")})
        audit.append(item)
    return audit, problems


def inventory(source, old_manifest):
    require(source.name == "train.json", "Only the official train.json source may be opened")
    require(sha(source) == SOURCE_SHA, "Training source differs from the audited official bytes")
    require(sha(ROOT / "src/recipeqa/data.py") == DATA_CODE_SHA, "Frozen data utilities changed")
    require(sha(ROOT / "configs/protocol_lock.json") == FROZEN_SHA, "Frozen protocol metadata changed")
    require(sha(old_manifest) == OLD_TRAIN_SHA, "Original 1,024-question training pool changed")
    raw = json.loads(source.read_bytes())
    require(isinstance(raw, dict) and isinstance(raw.get("data"), list), "Official source is not a data-list object")
    audit, problems = independent_choice_audit(raw["data"])
    from recipeqa.data import load_source
    # Independent issues are reported before asking the frozen loader to fail on
    # its first invalid row. No damaged/ambiguous source item is ever dropped.
    records, frozen_report = load_source(source, "train") if not problems else (None, None)
    old = jsonl(old_manifest)
    require(len(old) == len({row["qid"] for row in old}) == len({row["recipe_id"] for row in old}) == 1024,
            "Original pool is not 1,024 distinct questions/recipes")
    all_qids = {item["qid"] for item in audit}
    all_recipes = {item["recipe_id"] for item in audit}
    old_qids = {str(row["qid"]) for row in old}
    old_recipes = {str(row["recipe_id"]) for row in old}
    require(len(audit) == len(all_qids) == 7558 and len(all_recipes) == 5300,
            "Official full training denominator changed")
    require(old_qids <= all_qids and old_recipes <= all_recipes, "Original pool is not contained in official training")
    report = {"status": "source_metadata_verified" if not problems else "failed_source_metadata_retained",
              "scope": "Only official training JSON and existing training manifest; no validation/test JSON read",
              "source": {"path": str(source), "bytes": source.stat().st_size, "sha256": SOURCE_SHA,
                         "all_task_questions": len(raw["data"]), "ordering_questions": len(audit),
                         "ordering_recipes": len(all_recipes)},
              "frozen_loader_report": frozen_report, "source_metadata_issue_count": len(problems),
              "source_metadata_issues": problems,
              "old_train": {"path": str(old_manifest), "sha256": OLD_TRAIN_SHA, "questions": 1024, "recipes": 1024},
              "overlap": {"old_question_overlap": len(old_qids & all_qids), "old_recipe_overlap": len(old_recipes & all_recipes),
                          "additional_questions": len(all_qids - old_qids), "additional_recipes": len(all_recipes - old_recipes),
                          "new_questions_in_old_recipes": sum(item["qid"] not in old_qids and item["recipe_id"] in old_recipes
                                                             for item in audit)},
              "duplicate_incorrect_options_questions": sum(item.get("duplicate_incorrect_options", 0) > 0 for item in audit),
              "test_access": False, "training_started": False, "semantic_filtering": False,
              "omitted_ordering_questions": 0, "image_indexing_performed": False,
              "image_decoding_performed": False, "image_hashing_performed": False}
    if records is not None:
        frozen_by_qid = {row["qid"]: row for row in records}
        for item in audit:
            record = frozen_by_qid[item["qid"]]
            require(record["recipe_id"] == item["recipe_id"]
                    and record["images_chronological"] == item["images_chronological"]
                    and record["source_step_ids_chronological"] == item["source_step_ids_chronological"],
                    "Independent raw choice reconstruction disagrees with frozen loader")
    return records, audit, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/train.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/images")
    parser.add_argument("--old-train-manifest", type=Path, default=ROOT / "data/manifests/train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=HERE / "full_train_data")
    parser.add_argument("--check-source", action="store_true", help="Read-only metadata inventory; do not inspect image files")
    args = parser.parse_args()
    require(ROOT.is_relative_to("/root/autodl-tmp") and ROOT.stat().st_dev != Path("/").stat().st_dev,
            "All preparation inputs/outputs must remain on the separately mounted project data disk")
    source, image_root, old_manifest, output = map(inside, (args.source, args.image_root, args.old_train_manifest, args.output_dir))
    sys.path.insert(0, str(ROOT / "src"))
    records, audit, source_report = inventory(source, old_manifest)
    if args.check_source:
        print(json.dumps(source_report, ensure_ascii=False, indent=2))
        require(not source_report["source_metadata_issues"], "Source metadata failed; no item was filtered")
        return
    require(not output.exists(), "Output directory exists; inspect it, never overwrite/retry automatically")
    os.environ.update({key: "4" for key in THREADS})
    require("TMPDIR" in os.environ and Path(os.environ["TMPDIR"]).resolve().is_relative_to(ROOT),
            "Source the existing data-disk env.sh first; TMPDIR must be inside this project")
    from recipeqa.data import balanced_presentations, inspect_images, make_presentation, write_jsonl
    from recipeqa.metrics import score_order
    output.mkdir(parents=True)
    started = time.monotonic()
    state = {"status": "preparing", "pid": os.getpid(), "started_utc": now(),
             "test_access": False, "training_started": False, "automatic_restart": False}

    def update(**changes):
        state.update(changes, updated_utc=now(), elapsed_seconds=time.monotonic() - started)
        write(output / "state.json", state)

    def receipt(path):
        return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}

    try:
        update(phase="source_metadata")
        write(output / "source_inventory.json", source_report)
        write_jsonl(output / "official_choice_audit.jsonl", audit)
        require(not source_report["source_metadata_issues"], "Source metadata issues retained; no training manifest promoted")
        require(records is not None, "Frozen source loading did not complete")
        update(phase="image_index_and_rgb_decode", questions=len(records))
        image_info, image_report = inspect_images(
            (name for record in records for name in record["images_chronological"]), image_root, verify=True)
        image_report["scope"] = "Every distinct photograph used by all 7,558 official training ordering questions"
        write(output / "image_report.json", image_report)
        # Hash selected bytes and audit the frozen resolver's concrete paths.
        update(phase="image_byte_hashing", unique_images=len(image_info))
        hashes, path_issues = [], []
        for name, info in sorted(image_info.items()):
            entry = {"filename": name, "path": info.get("path"), "status": info["status"], "decoded": info.get("decoded", False)}
            if info.get("path") and info["status"] not in ("missing", "ambiguous_basename"):
                path = (image_root / info["path"]).resolve()
                try:
                    require(path.is_relative_to(image_root) and path.name == name, "Resolved image path escaped or changed basename")
                    entry.update(bytes=path.stat().st_size, sha256=sha(path))
                    require(entry["bytes"] == info["bytes"], "Image size changed after RGB decode")
                    if info.get("content_sha256"):
                        require(entry["sha256"] == info["content_sha256"], "Duplicate-copy resolution hash changed")
                except (ValueError, OSError) as error:
                    entry["hash_or_path_error"] = f"{type(error).__name__}: {error}"
                    path_issues.append(entry)
            hashes.append(entry)
        image_hash_report = write_jsonl(output / "image_bytes_sha256.jsonl", hashes)
        write(output / "image_index_report.json", {
            "image_root": str(image_root), "indexed_image_files": image_report["indexed_image_files"],
            "selected_unique_basenames": len(image_info), "resolver": "frozen inspect_images; identical duplicate bytes only",
            "resolved_image_report": receipt(output / "image_report.json"), "image_byte_hash_report": image_hash_report,
            "hash_or_path_issues": path_issues, "test_json_access": False})
        update(phase="all_questions_presentations")
        maps = balanced_presentations(records, PRESENTATION_SEED, PRESENTATION_NAMESPACE)
        ordered = sorted(records, key=lambda row: row["qid"])
        random.Random(ORDER_SEED).shuffle(ordered)
        by_qid = {item["qid"]: item for item in audit}
        visits = []
        for index, record in enumerate(ordered):
            display = maps[record["qid"]]
            visit = make_presentation(record, display, manifest_set="train_all7558", image_info=image_info)
            official = by_qid[record["qid"]]
            label_choices = [[chr(ord("A") + display.index(semantic)) for semantic in option]
                             for option in official["choice_semantic_permutations"]]
            require(label_choices[official["official_answer_index"]] == visit["target_labels"],
                    "Official gold-to-display mapping disagrees with free-order target")
            require(score_order(" ".join(visit["target_labels"]), visit["display_perm"], visit["target_order"])["exact"] == 1,
                    "Independent exact scorer rejects gold target")
            visit.update(visit_index=index, pass_index=0,
                         visit_id=f"all7558:order{ORDER_SEED}:presentation{PRESENTATION_SEED}:index{index:06d}",
                         data_order_seed=ORDER_SEED, presentation_seed=PRESENTATION_SEED,
                         official_four_choice_metadata={
                             "backend_only_not_model_visible": True,
                             "source_row_index": official["source_row_index"],
                             "source_ordering_index": official["source_ordering_index"],
                             "answer_index": official["official_answer_index"],
                             "choice_images": official["official_choices_images"],
                             "choice_semantic_permutations": official["choice_semantic_permutations"],
                             "choice_display_labels": label_choices})
            visits.append(visit)
        manifest_path = output / "train_all7558.jsonl"
        manifest = write_jsonl(manifest_path, visits)
        reloaded = jsonl(manifest_path)
        require(len(reloaded) == len({row["qid"] for row in reloaded}) == len({row["visit_id"] for row in reloaded}) == 7558,
                "Full training output lost/duplicated a question or visit")
        require({row["qid"] for row in reloaded} == {row["qid"] for row in records}
                and len({row["recipe_id"] for row in reloaded}) == 5300, "Full source coverage changed")
        require(reloaded == visits, "Written JSONL does not round-trip to all source records")
        presentation_counts = Counter("".join(map(str, row["display_perm"])) for row in reloaded)
        require(len(presentation_counts) == 24 and max(presentation_counts.values()) - min(presentation_counts.values()) <= 1,
                "All-24 presentation balancing failed")
        issue_questions = [row["qid"] for row in reloaded if row["technical_status"] != "ok"]
        issues = image_report["issues"] or path_issues or issue_questions
        require(sha(source) == SOURCE_SHA and sha(ROOT / "src/recipeqa/data.py") == DATA_CODE_SHA
                and sha(old_manifest) == OLD_TRAIN_SHA, "Bound source changed during preparation")
        report = {"status": "failed_technical_issues_retained" if issues else "ready_full_training_data_no_training_started",
                  "created_utc": now(), "task": "RecipeQA-derived ordered-text four-photo free ordering",
                  "manifest": {**manifest, "path": str(manifest_path), "questions": 7558, "recipes": 5300,
                               "passes": 1, "unique_visit_ids": 7558, "technical_issue_questions": issue_questions,
                               "presentation_counts": dict(sorted(presentation_counts.items()))},
                  "seeds": {"question_order": ORDER_SEED, "presentation": PRESENTATION_SEED},
                  "question_order_algorithm": "sort qid lexicographically; Python random.Random(20261114).shuffle once",
                  "presentation_algorithm": "frozen balanced_presentations with SHA256-namespaced seed; all24 count difference <=1",
                  "presentation_namespace": PRESENTATION_NAMESPACE,
                  "source_inventory": receipt(output / "source_inventory.json"),
                  "official_choices_audit": receipt(output / "official_choice_audit.jsonl"),
                  "image_report": receipt(output / "image_report.json"),
                  "image_index_report": receipt(output / "image_index_report.json"),
                  "image_bytes_sha256": receipt(output / "image_bytes_sha256.jsonl"),
                  "source_sha256": {"train.json": SOURCE_SHA, "src/recipeqa/data.py": DATA_CODE_SHA,
                                    "configs/protocol_lock.json": FROZEN_SHA,
                                    str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__)},
                  "old_train_overlap": source_report["overlap"], "semantic_filtering": False,
                  "omitted_ordering_questions": 0, "technical_issues_policy": "All rows retained; any technical issue blocks training-readiness",
                  "text_policy": "Full cleaned ordered recipe preserved; model input clipping remains a separate fixed downstream policy",
                  "backend_metadata_policy": "Original choice mappings/targets/filenames/source IDs never enter prompt text",
                  "inference_or_training_started": False, "test_json_access": False,
                  "disclosure": "Expanded data is a resource intervention, not an algorithmic gain; future comparisons need a common dataset/start/budget",
                  "future_use": "Data only; no hyperparameters selected, training scheduled, or frozen study changed"}
        write(output / "manifest_report.json", report)
        require(not issues, "Technical issues retained in full manifest and reports; training readiness FAILED, no rows removed")
        update(status="ready_full_training_data_no_training_started", phase="all_7558_verified", manifest=manifest,
               finished_utc=now(), manifest_report=receipt(output / "manifest_report.json"))
        print(json.dumps({"status": state["status"], "output_dir": str(output), "manifest": manifest,
                          "questions": 7558, "recipes": 5300, "overlap": source_report["overlap"],
                          "test_access": False, "training_started": False}, indent=2))
    except BaseException as error:
        update(status="failed_preserved_no_retry", error=f"{type(error).__name__}: {error}", finished_utc=now())
        raise


if __name__ == "__main__":
    main()
