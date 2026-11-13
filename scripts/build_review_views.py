#!/usr/bin/env python3
"""Create model-visible-text REVIEW VIEWS without changing frozen model inputs.

CPU only. Reuses the frozen renderer and exact uniform-body-cap search. The
unexpanded input plus a conservative visual-token bound must stay below4096;
otherwise this text-only tool fails rather than guessing a multimodal cap.

Example:
  python scripts/build_review_views.py --root /root/autodl-tmp/rl_recipeqa \
      --build-input-packet

Predict/coverage continue to use the original frozen manifests. Only human
review rendering and E4 summaries use these derived text views.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile


DATA_DISK = Path("/root/autodl-tmp")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def confined(path: Path, root: Path = DATA_DISK) -> Path:
    path, root = path.expanduser().resolve(), root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path is outside the data disk: {path}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = confined(path.with_name(path.name + ".part"))
    partial.write_text(value, encoding="utf-8")
    partial.replace(path)


def write_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def protected_record(row: dict) -> dict:
    """Every original field except body must survive byte-for-byte JSON equality."""
    result = deepcopy(row)
    result.pop("review_text_view", None)
    for step in result["steps"]:
        step.pop("body", None)
    return result


def derive_record(record, source_sha, processor, config, visual_allowance, render_recipe, build_messages):
    tokenizer = processor.tokenizer
    steps = record["steps"]
    raw_recipe = render_recipe(steps, tokenizer, None)
    titles = render_recipe(steps, tokenizer, 0)
    if len(tokenizer.encode(titles, add_special_tokens=False)) > config.recipe_token_budget:
        raise RuntimeError(f"All titles exceed the frozen budget: {record['qid']}")
    lengths = [len(tokenizer.encode(str(step.get("body", "")), add_special_tokens=False)) for step in steps]
    low, high, best = 0, max(lengths), None
    largest_accepted_bound = {"answer": 0, "process": 0}
    while low <= high:
        cap = (low + high) // 2
        recipe = render_recipe(steps, tokenizer, cap)
        recipe_tokens = len(tokenizer.encode(recipe, add_special_tokens=False))
        if recipe_tokens > config.recipe_token_budget:
            high = cap - 1
            continue
        # This preserves prepare_record's input-budget check. The bound includes
        # each original image placeholder PLUS all possible visual tokens, so it
        # deliberately overcounts. Prove nonbinding for EVERY accepted candidate.
        bounds = {}
        for prompt_kind, process in (("answer", False), ("process", True)):
            template = processor.apply_chat_template(build_messages(recipe, process=process), tokenize=False, add_generation_prompt=True)
            bounds[prompt_kind] = len(tokenizer.encode(template, add_special_tokens=False)) + visual_allowance
            largest_accepted_bound[prompt_kind] = max(largest_accepted_bound[prompt_kind], bounds[prompt_kind])
            if bounds[prompt_kind] > config.max_input_tokens:
                raise RuntimeError(f"Cannot prove exact text-only cap for {record['qid']}: {prompt_kind} bound={bounds[prompt_kind]}; use a separate actual processor audit")
        best = cap, recipe, recipe_tokens, bounds
        low = cap + 1
    if best is None:
        raise RuntimeError(f"No feasible text cap: {record['qid']}")
    cap, expected_recipe, recipe_tokens, bounds = best
    result = deepcopy(record)
    for step in result["steps"]:
        body_tokens = tokenizer.encode(str(step.get("body", "")).strip(), add_special_tokens=False)
        step["body"] = tokenizer.decode(body_tokens[:cap], skip_special_tokens=True).strip()
    if protected_record(result) != protected_record(record):
        raise AssertionError("Derived review changed protected metadata, titles, IDs, layout, targets or images")
    rendered_roundtrip = render_recipe(result["steps"], tokenizer, None)
    if rendered_roundtrip != expected_recipe:
        raise AssertionError(f"Derived step bodies do not roundtrip to actual model recipe text: {record['qid']}")
    clipped_ids = [step.get("source_step_id", index + 1) for index, (step, length) in enumerate(zip(steps, lengths)) if length > cap]
    provenance = {
        "kind": "model_visible_recipe_text_review_only", "parent_manifest_sha256": source_sha,
        "uniform_body_cap": cap, "recipe_tokens_raw": len(tokenizer.encode(raw_recipe, add_special_tokens=False)),
        "recipe_tokens_used": recipe_tokens, "body_clipped_step_ids": clipped_ids,
        "rendered_recipe_sha256": hashlib.sha256(expected_recipe.encode()).hexdigest(),
        "protected_record_sha256": hash_json(protected_record(record)),
        "input_upper_bound": bounds, "largest_feasible_candidate_input_upper_bound": largest_accepted_bound,
        "roundtrip_text_equal": True, "all_non_body_fields_equal": True,
        "human_labels": "pending; no judgments generated",
    }
    result["review_text_view"] = provenance
    return result


def build_input_packet(root: Path, audit_view: Path, output_dir: Path, view_report: Path) -> dict:
    builder = root / "scripts/build_audit_packet.py"
    command = [sys.executable, str(builder), "--manifest", str(audit_view),
               "--image-root", str(root / "data/images"), "--output-dir", str(output_dir)]
    subprocess.run(command, check=True, stdout=sys.stderr)
    replacements = {
        "<title>RecipeQA 输入可判定性独立审核</title>": "<title>RecipeQA 模型可见正文独立审核</title>",
        "<h1>RecipeQA 输入可判定性独立审核</h1>": "<h1>RecipeQA 模型可见正文独立审核</h1>",
        "<summary>按原顺序提供的完整食谱</summary>": "<summary>模型可见的有序食谱（全部标题保留，正文按固定预算裁剪）</summary>",
    }
    for reviewer in ("reviewer_1", "reviewer_2"):
        path = confined(output_dir / f"{reviewer}.html")
        page = path.read_text()
        for original, revised in replacements.items():
            if original not in page:
                raise RuntimeError(f"Frozen renderer heading changed; inspect before annotating new HTML: {original}")
            page = page.replace(original, revised)
        page = page.replace('<div class="notice">', '<div class="notice"><strong>本v3只展示模型实际可见的正文。所有步骤与标题保留；请勿用未展示的源全文补全信息。脚本不预填人工判断。</strong><br>', 1)
        atomic_text(path, page)
    readme = output_dir / "README.txt"
    atomic_text(readme, "本v3为模型可见正文审核包，不是源食谱全文审核。\n标题/步骤/图片/目标不变；正文与冻结模型输入的统一token cap严格一致。\n人类标签仍pending；请独立审核。v2完整源文本包保留作历史，不再作为模型可见性审核入口。\n\n" + readme.read_text())
    metadata = json.loads((output_dir / "packet_manifest.json").read_text())
    metadata["review_text_view"] = {"manifest": str(audit_view), "sha256": sha256_file(audit_view),
                                    "validation_report": str(view_report), "human_labels": "pending"}
    write_json(output_dir / "packet_manifest.json", metadata)
    archive = confined(output_dir / "PUBLIC_input_audit.zip")
    partial = confined(archive.with_name(archive.name + ".part"))
    public_files = [output_dir / name for name in ("reviewer_1.html", "reviewer_2.html", "reviewer_1_blank.csv", "reviewer_2_blank.csv", "README.txt")]
    public_files.extend(sorted((output_dir / "assets").glob("*.jpg")))
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in public_files:
            zipped.write(path, path.relative_to(output_dir).as_posix())
    partial.replace(archive)
    with zipfile.ZipFile(archive) as zipped:
        if zipped.testzip() is not None or any("PRIVATE" in name for name in zipped.namelist()):
            raise AssertionError("Public package failed CRC/privacy check")
    return {"directory": str(output_dir), "public_zip": str(archive), "public_zip_sha256": sha256_file(archive),
            "human_labels": "pending", "private_mapping_excluded": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/rl_recipeqa"))
    parser.add_argument("--build-input-packet", action="store_true")
    parser.add_argument("--include-calibration5", action="store_true", help="Also derive the fixed monitor-first-five calibration view without materializing its prediction manifest")
    args = parser.parse_args()
    root = confined(args.root)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in THREAD_VARS:
        os.environ[key] = "4"
    sys.path.insert(0, str(root / "src"))
    import torch
    from recipeqa.modeling import InputConfig, _render_recipe, build_messages, load_processor, hash_json as frozen_hash_json

    torch.set_num_threads(4)
    if torch.cuda.is_available() or torch.cuda.is_initialized():
        raise RuntimeError("This review-view generator must run without any GPU")
    protocol_path = root / "configs/protocol_lock.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen":
        raise RuntimeError("Use the existing frozen protocol; no training settings are chosen here")
    frozen_hashes = {name: sha256_file(root / name) for name in protocol["file_sha256"]}
    if frozen_hashes != protocol["file_sha256"]:
        raise RuntimeError("A frozen source/data/config file differs from the original lock")
    processor_auxiliary = {item["name"]: sha256_file(Path(protocol["base"]["directory"]) / item["name"])
                           for item in protocol["base"]["auxiliary_files"]}
    if processor_auxiliary != {item["name"]: item["sha256"] for item in protocol["base"]["auxiliary_files"]}:
        raise RuntimeError("Frozen tokenizer/processor auxiliary contents changed")
    scientific_sources = ("src/recipeqa/modeling.py", "scripts/build_audit_packet.py")
    for name in scientific_sources:
        if sha256_file(root / name) != protocol["file_sha256"][name]:
            raise RuntimeError(f"Frozen renderer/preprocessor changed: {name}")
    inp = protocol["candidate"]["input"]
    config = InputConfig(inp["min_pixels"], inp["max_pixels"], inp["recipe_token_budget"], inp["max_input_tokens"], inp["main_max_new_tokens"])
    processor = load_processor(str(root / "models/Qwen3-VL-2B-Instruct"), protocol["candidate"]["revision"], config)
    patch = int(processor.image_processor.patch_size)
    merge = int(processor.image_processor.merge_size)
    # Fixed Qwen3-VL RGB image configuration: each merged image token covers
    # (patch*merge)^2 pixels; static-image temporal grid has one merged slice.
    visual_allowance = 4 * math.ceil(config.max_pixels / (patch * merge) ** 2)
    output_dir = confined(root / "data/review_views")
    report_path = output_dir / "review_views_report.json"
    previous_report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report = {
        "scope": "Derived human-review text only; original train/test/predict/coverage manifests and all frozen files unchanged",
        "human_labels": "pending", "cuda_available": torch.cuda.is_available(), "cuda_initialized": torch.cuda.is_initialized(),
        "torch_threads": torch.get_num_threads(), "thread_environment": {key: os.environ[key] for key in THREAD_VARS},
        "protocol_sha256": sha256_file(protocol_path),
        "generator_sha256": sha256_file(Path(__file__)),
        "processor_auxiliary_sha256": processor_auxiliary,
        "frozen_source_sha256": {name: sha256_file(root / name) for name in scientific_sources},
        "pixel_budget": config.max_pixels, "patch_size": patch, "merge_size": merge,
        "four_image_conservative_visual_token_allowance": visual_allowance,
        "cap_proof": "Exact frozen text search; every recipe-feasible candidate plus all possible image tokens is below full input limit for answer and process, so the multimodal total-length branch cannot change the chosen cap",
        "views": {},
    }
    sources, views = {}, {}
    for source_name, view_name, expected in (("audit", "audit100_model_visible", 100), ("process", "process40_model_visible", 40)):
        source_path = confined(root / f"data/manifests/{source_name}.jsonl")
        before = sha256_file(source_path)
        if before != protocol["file_sha256"][str(source_path.relative_to(root))]:
            raise RuntimeError(f"Frozen original manifest changed: {source_path}")
        rows = read_rows(source_path)
        if len(rows) != expected or len({row["qid"] for row in rows}) != expected or len({row["recipe_id"] for row in rows}) != expected:
            raise RuntimeError(f"Review cohort count/uniqueness mismatch: {source_name}")
        derived = [derive_record(row, before, processor, config, visual_allowance, _render_recipe, build_messages) for row in rows]
        if [protected_record(row) for row in rows] != [protected_record(row) for row in derived]:
            raise AssertionError("Cohort order or protected fields changed")
        path = confined(output_dir / f"{view_name}.jsonl")
        atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in derived))
        if read_rows(path) != derived or sha256_file(source_path) != before:
            raise AssertionError("Derived serialization roundtrip failed or source was modified")
        changed_body = sum(any(a.get("body") != b.get("body") for a, b in zip(row["steps"], view["steps"])) for row, view in zip(rows, derived))
        report["views"][view_name] = {
            "source_manifest": str(source_path), "source_manifest_sha256": before,
            "path": str(path), "sha256": sha256_file(path), "questions": len(rows), "recipes": len(rows),
            "cohort_identity_sha256": hash_json([protected_record(row) for row in rows]),
            "cohort_order_and_all_non_body_fields_equal": True, "all_rendered_text_roundtrips_equal": True,
            "source_file_unchanged": True,
            "questions_with_clipped_body": sum(bool(row["review_text_view"]["body_clipped_step_ids"]) for row in derived),
            "questions_with_any_body_string_change": changed_body,
            "clipped_step_count": sum(len(row["review_text_view"]["body_clipped_step_ids"]) for row in derived),
            "max_input_upper_bound": {kind: max(row["review_text_view"]["largest_feasible_candidate_input_upper_bound"][kind] for row in derived) for kind in ("answer", "process")},
            "per_question": [{"qid": row["qid"], "recipe_id": row["recipe_id"], **row["review_text_view"]} for row in derived],
        }
        sources[source_name], views[source_name] = rows, derived
    audit_map = {row["qid"]: row for row in views["audit"]}
    if not {row["qid"] for row in views["process"]}.issubset(audit_map):
        raise AssertionError("Process40 no longer belongs to audit100")
    if any(row["steps"] != audit_map[row["qid"]]["steps"] or row["review_text_view"]["uniform_body_cap"] != audit_map[row["qid"]]["review_text_view"]["uniform_body_cap"] for row in views["process"]):
        raise AssertionError("Overlapping audit/process model-visible recipe text differs")
    report["process40_nested_in_audit100"] = True
    report["overlapping_model_visible_steps_and_caps_equal"] = True
    if args.include_calibration5:
        monitor_path = root / "data/manifests/monitor.jsonl"
        monitor_sha = sha256_file(monitor_path)
        expected_monitor_sha = json.loads((root / "data/manifests/manifest_report.json").read_text())["manifests"]["monitor"]["sha256"]
        if monitor_sha != expected_monitor_sha:
            raise RuntimeError("Fixed monitor source changed")
        # Keep this exact serialization/flag aligned with the new evaluation
        # wrapper's monitor_subset_payload(..., 5, 'process_calibration5').
        calibration = [dict(row, manifest_set="process_calibration5", control_excluded_from_main=True)
                       for row in read_rows(monitor_path)[:5]]
        if len(calibration) != 5 or len({row["recipe_id"] for row in calibration}) != 5:
            raise RuntimeError("Calibration needs exactly five independent stored monitor recipes")
        expected_payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in calibration).encode()
        expected_sha = hashlib.sha256(expected_payload).hexdigest()
        calibration_views = [derive_record(row, expected_sha, processor, config, visual_allowance, _render_recipe, build_messages) for row in calibration]
        calibration_path = output_dir / "calibration5_model_visible.jsonl"
        atomic_text(calibration_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in calibration_views))
        if read_rows(calibration_path) != calibration_views or [protected_record(row) for row in calibration] != [protected_record(row) for row in calibration_views]:
            raise AssertionError("Calibration serialization or protected fields changed")
        # Check against previously executed real W input audits, with no new
        # images/model/GPU call. Source-full-text failure report remains intact.
        w_step = json.loads((root / "metadata/w_selection.json").read_text())["selected"]["step"]
        actual_path = root / f"runs/warmup_dev_step_{w_step}/predictions.jsonl"
        actual = {row["qid"]: row for row in read_rows(actual_path)}
        proof = []
        for row in calibration_views:
            observed = actual[row["qid"]]
            audit = observed["input_audit"]
            text = _render_recipe(row["steps"], processor.tokenizer, None)
            same = frozen_hash_json(build_messages(text)) == audit["prompt_text_hash"]
            cap_equal = row["review_text_view"]["uniform_body_cap"] == audit["uniform_body_cap"]
            if not same or not cap_equal or row["display_perm"] != observed["display_perm"]:
                raise AssertionError(f"Calibration view differs from actual selected-W input audit: {row['qid']}")
            proof.append({"qid": row["qid"], "recipe_id": row["recipe_id"], **row["review_text_view"],
                          "actual_W_answer_prompt_sha256": audit["prompt_text_hash"],
                          "derived_answer_prompt_equals_actual_W": same, "cap_equals_actual_W": cap_equal})
        report["views"]["calibration5_model_visible"] = {
            "source_kind": "expected_canonical_monitor_first5_control_payload_not_materialized",
            "source_manifest": str(root / "data/controls/process_calibration5.jsonl"),
            "source_manifest_sha256": expected_sha, "parent_expected_payload_sha256": expected_sha,
            "source_monitor_manifest": str(monitor_path), "source_monitor_sha256": monitor_sha,
            "canonical_control_additions": {"manifest_set": "process_calibration5", "control_excluded_from_main": True},
            "path": str(calibration_path), "sha256": sha256_file(calibration_path), "questions": 5, "recipes": 5,
            "cohort_identity_sha256": hash_json([protected_record(row) for row in calibration]),
            "cohort_order_and_all_non_body_fields_equal": True, "all_rendered_text_roundtrips_equal": True,
            "all_rendered_text_equals_actual_W_input": True, "original_monitor_file_unchanged": sha256_file(monitor_path) == monitor_sha,
            "questions_with_clipped_body": sum(bool(row["review_text_view"]["body_clipped_step_ids"]) for row in calibration_views),
            "clipped_step_count": sum(len(row["review_text_view"]["body_clipped_step_ids"]) for row in calibration_views),
            "actual_W_audit_source": str(actual_path), "actual_W_audit_source_sha256": sha256_file(actual_path),
            "per_question": proof, "human_labels": "pending; calibration excluded from main E4 denominator",
        }
    if not args.build_input_packet and previous_report.get("input_audit_packet"):
        old_packet = previous_report["input_audit_packet"]
        old_archive = Path(old_packet["public_zip"])
        old_metadata = json.loads((Path(old_packet["directory"]) / "packet_manifest.json").read_text())
        if old_metadata["source_manifest_sha256"] == report["views"]["audit100_model_visible"]["sha256"] and sha256_file(old_archive) == old_packet["public_zip_sha256"]:
            report["input_audit_packet"] = old_packet
    write_json(report_path, report)
    if args.build_input_packet:
        report["input_audit_packet"] = build_input_packet(root, output_dir / "audit100_model_visible.jsonl",
            confined(root / "review/input_audit_v3_model_visible"), report_path)
        write_json(report_path, report)
    if any(sha256_file(root / name) != expected for name, expected in frozen_hashes.items()):
        raise AssertionError("A frozen source/data/config changed during the CPU review-view generation")
    report["all_frozen_file_hashes_verified_before_and_after"] = True
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "human_labels": "pending", "views": {
        name: {key: value for key, value in view.items() if key != "per_question"} for name, view in report["views"].items()},
        "input_audit_packet": report.get("input_audit_packet")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
