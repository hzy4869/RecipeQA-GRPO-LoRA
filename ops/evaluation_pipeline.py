#!/usr/bin/env python3
"""Run fixed E0-E4 evaluation after development-only checkpoint selection.

No training, model selection, tuning, automatic retries, or human labels occur
here. Start once; after inspecting a failure, --resume reuses verified stages.
--check-only fails fast while P6 is incomplete and never waits for training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PRIMARY = ("sft_seed42", "exact_seed42", "pair_seed42")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
PROCESS_REVIEW_SHA256 = "89f8397c463513bc6cf2ff37096ecae01ecd3f9d30eb473f334a2b0e5ff18691"
CALIBRATION_REVIEW_SHA256 = "decf887c7b36026032c881dc11f3cb334dd68953b0a29adb823921704629974b"
EVALUATION_SOURCES = ("ops/evaluation_pipeline.py", "ops/compare_greedy.py", "scripts/build_review_views.py", "scripts/plot_training_dynamics.py")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def monitor_subset_payload(rows, count, name):
    selected = [dict(row, manifest_set=name, control_excluded_from_main=True) for row in rows[:count]]
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in selected).encode("utf-8")


def calibration_payload(rows):
    """Fixed first-five monitor examples; never part of the main E4 denominator."""
    return monitor_subset_payload(rows, 5, "process_calibration5")


def checked_threads(root):
    policy = read_json(root / "metadata/runtime_threads.json")
    current = {key: os.environ.get(key) for key in THREAD_VARS}
    if any(policy.get(key) != 4 or current[key] != "4" for key in THREAD_VARS):
        raise RuntimeError("Evaluation requires the common four-thread runtime. After source env.sh run: export " +
                           " ".join(f"{key}=4" for key in THREAD_VARS) + f"; observed={current}")
    return current


def protected_review_record(row):
    return {key: ([{field: value for field, value in step.items() if field != "body"} for step in item]
                  if key == "steps" else item) for key, item in row.items() if key != "review_text_view"}


def checked_review_view(root, source, view_name, protocol_sha256, max_input_tokens, expected_sha256=None):
    from recipeqa.modeling import sha256_file
    path = root / f"data/review_views/{view_name}.jsonl"
    report_path = root / "data/review_views/review_views_report.json"
    report = read_json(report_path)
    view = report["views"][view_name]
    expected_sha256 = expected_sha256 or view["sha256"]
    if sha256_file(path) != expected_sha256 or view["sha256"] != expected_sha256 or view["source_manifest_sha256"] != source["sha256"] or report["protocol_sha256"] != protocol_sha256:
        raise RuntimeError("Human-visible E4 view does not match the reviewed source/protocol/content hashes")
    unchanged = "original_monitor_file_unchanged" if view_name == "calibration5_model_visible" else "source_file_unchanged"
    if not all(view.get(key) is True for key in ("cohort_order_and_all_non_body_fields_equal", "all_rendered_text_roundtrips_equal", unchanged)) or report.get("all_frozen_file_hashes_verified_before_and_after") is not True:
        raise RuntimeError("E4 review-view derivation proof is incomplete")
    derived = read_rows(path)
    if len(derived) != len(source["records"]) or len({row["recipe_id"] for row in derived}) != len(derived):
        raise RuntimeError("E4 review view must contain the same independent recipe inputs")
    for original, shown in zip(source["records"], derived):
        audit = shown.get("review_text_view", {})
        if protected_review_record(original) != protected_review_record(shown) or audit.get("parent_manifest_sha256") != source["sha256"] or audit.get("all_non_body_fields_equal") is not True or audit.get("roundtrip_text_equal") is not True:
            raise RuntimeError(f"Human review view changed a protected field or lacks text proof: {original['qid']}")
        if any(audit.get("input_upper_bound", {}).get(kind, max_input_tokens + 1) > max_input_tokens for kind in ("answer", "process")):
            raise RuntimeError(f"Human review text cap lacks a valid input bound: {original['qid']}")
    return {"path": str(path), "sha256": expected_sha256, "report": str(report_path),
            "report_sha256": sha256_file(report_path), "source_manifest": source["path"],
            "source_manifest_sha256": source["sha256"], "scope": "Human summary text only; prediction/coverage retain original inputs; human labels pending"}


def check_view_against_actual_prompts(shown_rows, predictions, process):
    from recipeqa.modeling import _render_recipe, build_messages, hash_json
    indexed = {(row["qid"], tuple(row["display_perm"])): row for row in predictions}
    for shown in shown_rows:
        observed = indexed[(shown["qid"], tuple(shown["display_perm"]))]
        audit = observed["input_audit"]
        text = _render_recipe(shown["steps"], None, None)  # No tokenizer used for an already-derived full visible text.
        if observed.get("technical_error") or audit["prompt_text_hash"] != hash_json(build_messages(text, process=process)) or audit["uniform_body_cap"] != shown["review_text_view"]["uniform_body_cap"]:
            raise RuntimeError(f"Human review text does not equal actual model-visible prompt: {shown['qid']}")


def ready_inputs(root):
    from recipeqa.modeling import model_weight_identity, sha256_file

    formal = read_json(root / "metadata/formal_training_pipeline.json")
    if formal.get("status") != "completed" or formal.get("phase") != "three_branches_selected":
        raise RuntimeError(f"P6 not ready: status={formal.get('status')}, phase={formal.get('phase')}; no polling or GPU work started")
    runtime_threads = checked_threads(root)
    protocol = read_json(root / "configs/protocol_lock.json")
    if protocol.get("status") != "frozen" or formal.get("protocol_sha256") != sha256_file(root / "configs/protocol_lock.json"):
        raise RuntimeError("Formal training and evaluation must share one frozen protocol")
    if protocol.get("formal_test_model_predictions_used_for_selection") is not False:
        raise RuntimeError("The development-only selection declaration is missing or contradictory")
    for name, expected in protocol["file_sha256"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"Frozen source/manifest changed: {name}")
    if model_weight_identity(protocol["base"]["directory"]) != protocol["base"]:
        raise RuntimeError("Base model contents changed")
    base = str(Path(protocol["base"]["directory"]).resolve())
    w = str(Path(protocol["W"]["directory"]).resolve())
    if model_weight_identity(w, adapter=w != base) != protocol["W"]:
        raise RuntimeError("Common W contents changed")
    models = {"base": {"checkpoint": base, "adapter": None}}
    if w != base:
        models["w"] = {"checkpoint": w, "adapter": w}
    selected = formal.get("selected", {})
    if not set(PRIMARY).issubset(selected) or set(selected) - {*PRIMARY, "sft_4pass_seed42"}:
        raise RuntimeError("Three primary checkpoints must be selected, with at most the predeclared SFT budget control")
    for name in (*PRIMARY, *(("sft_4pass_seed42",) if "sft_4pass_seed42" in selected else ())):
        directory = root / "runs" / name
        best = read_json(directory / "best_checkpoint.json")
        chosen = selected[name]
        checkpoint = str(Path(chosen["checkpoint"]).resolve())
        if not Path(checkpoint).is_relative_to(root) or best["selected"] != chosen:
            raise RuntimeError(f"Selected checkpoint disagreement: {name}")
        if not read_json(directory / "artifact_validation.json").get("passed"):
            raise RuntimeError(f"Training artifact gate has not passed: {name}")
        if model_weight_identity(checkpoint, adapter=True) != best["selected_identity"]:
            raise RuntimeError(f"Selected checkpoint bytes changed: {name}")
        models[name] = {"checkpoint": checkpoint, "adapter": checkpoint, "best_checkpoint": best}
    manifests = {}
    for name, rows, questions, recipes in (("test", 905, 905, 597), ("orbit", 1440, 60, 60), ("process", 40, 40, 40), ("bank", 100, 100, 100)):
        path = root / f"data/manifests/{name}.jsonl"
        records = read_rows(path)
        keys = {(str(row["recipe_id"]), str(row["qid"]), tuple(row["display_perm"])) for row in records}
        if len(records) != rows or len(keys) != rows or len({key[:2] for key in keys}) != questions or len({key[0] for key in keys}) != recipes:
            raise RuntimeError(f"Unexpected frozen manifest size/duplicates: {name}")
        manifests[name] = {"path": str(path), "records": records, "sha256": sha256_file(path)}
    test_inputs = {(row["recipe_id"], row["qid"], tuple(row["display_perm"])) for row in manifests["test"]["records"]}
    if not {(row["recipe_id"], row["qid"], tuple(row["display_perm"])) for row in manifests["process"]["records"]}.issubset(test_inputs):
        raise RuntimeError("E4 answer-only comparison must use identical E0 questions and layouts")
    process_review = checked_review_view(root, manifests["process"], "process40_model_visible", formal["protocol_sha256"], protocol["candidate"]["input"]["max_input_tokens"], PROCESS_REVIEW_SHA256)
    monitor = root / "data/manifests/monitor.jsonl"
    monitor_rows = read_rows(monitor)
    report = read_json(root / "data/manifests/manifest_report.json")
    if sha256_file(monitor) != report["manifests"]["monitor"]["sha256"] or len(monitor_rows) != 50:
        raise RuntimeError("Calibration source differs from the frozen 50-question monitor manifest")
    control_rows = monitor_rows[:10]
    if len({row["recipe_id"] for row in control_rows}) != 10 or any(row["source_split"] != "val" for row in control_rows) or {row["recipe_id"] for row in control_rows} & {row["recipe_id"] for row in manifests["test"]["records"]}:
        raise RuntimeError("Monitor controls must be independent validation recipes outside the formal test")
    for name, count in (("process_calibration5", 5), ("greedy_determinism10", 10)):
        manifests[name] = {"path": str(root / f"data/controls/{name}.jsonl"),
            "sha256": hashlib.sha256(monitor_subset_payload(monitor_rows, count, name)).hexdigest(), "source": str(monitor),
            "source_sha256": sha256_file(monitor), "count": count,
            "selection": f"first {count} stored monitor rows; excluded from main test/orbit/process statistics"}
    calibration_source = {**manifests["process_calibration5"], "records": [json.loads(line) for line in calibration_payload(monitor_rows).splitlines()]}
    calibration_review = checked_review_view(root, calibration_source, "calibration5_model_visible", formal["protocol_sha256"], protocol["candidate"]["input"]["max_input_tokens"], CALIBRATION_REVIEW_SHA256)
    calibration_proof = read_json(Path(calibration_review["report"]))["views"]["calibration5_model_visible"]
    actual_w = Path(calibration_proof["actual_W_audit_source"])
    if not actual_w.resolve().is_relative_to(root) or sha256_file(actual_w) != calibration_proof["actual_W_audit_source_sha256"] or calibration_proof.get("all_rendered_text_equals_actual_W_input") is not True or calibration_proof["source_monitor_sha256"] != sha256_file(monitor):
        raise RuntimeError("Calibration text is not supported by the selected-W real input audit")
    observed_w = read_rows(actual_w)
    if any(row["checkpoint"] != w for row in observed_w):
        raise RuntimeError("Calibration actual audit is from a different W checkpoint")
    check_view_against_actual_prompts(read_rows(Path(calibration_review["path"])), observed_w, process=False)
    bank = root / "runs/w_candidate_bank/predictions.jsonl"
    if not bank.is_file():
        raise RuntimeError("Completed E1 W bank is missing; this wrapper never regenerates it")
    bank_config = read_json(bank.with_suffix(bank.suffix + ".config.json"))
    if bank_config.get("weight_identity") != {"base": protocol["base"], "adapter": protocol["W"] if w != base else None}:
        raise RuntimeError("Existing E1 bank does not match frozen Base/W contents")
    snapshot = {"protocol_sha256": formal["protocol_sha256"], "selected": selected,
                "models": models, "manifests": {key: {k: v for k, v in value.items() if k != "records"} for key, value in manifests.items()},
                "W_alias_of_base": w == base, "test_used_for_selection": False,
                "runtime_threads": runtime_threads, "runtime_threads_sha256": sha256_file(root / "metadata/runtime_threads.json"),
                "process_review": process_review, "calibration_review": calibration_review,
                "evaluation_source_sha256": {name: sha256_file(root / name) for name in EVALUATION_SOURCES}}
    return protocol, snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/rl_recipeqa"))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Explicitly continue after inspecting a stopped wrapper; never launch a second live job")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_relative_to("/root/autodl-tmp") or root.stat().st_dev == Path("/").stat().st_dev:
        raise ValueError("A separately mounted /root/autodl-tmp project is required")
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from recipeqa.modeling import hash_json, sha256_file, write_json
    try:
        protocol, snapshot = ready_inputs(root)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "not_ready", "reason": str(error)}), flush=True)
        return 2
    if args.check_only:
        print(json.dumps({"status": "ready", "selected_models": list(snapshot["models"]), "protocol_sha256": snapshot["protocol_sha256"]}), flush=True)
        return 0
    lock = (root / "metadata/evaluation_pipeline.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "metadata/evaluation_pipeline.json"
    previous = read_json(state_path) if state_path.exists() else None
    if previous and not args.resume:
        raise RuntimeError("An evaluation record exists; inspect it and use explicit --resume")
    if args.resume and not previous:
        raise RuntimeError("No previous evaluation exists to resume")
    signature = hash_json(snapshot)
    if previous and previous.get("selection_snapshot_sha256") != signature:
        raise RuntimeError("Frozen evaluation identities changed; do not mix model selections after test")
    if previous:
        child = previous.get("active_child_pid")
        stat = Path(f"/proc/{child}/stat") if child else None
        if stat and stat.exists() and stat.read_text().rsplit(")", 1)[-1].split()[0] not in ("Z", "X"):
            raise RuntimeError(f"Prior child {child} is still live; observe it instead of restarting")
    state = previous or {"commands": [], "selection_snapshot_sha256": signature,
                         "started_at": datetime.now(timezone.utc).isoformat(), "selected": snapshot["selected"]}
    if state.get("error"):
        state.setdefault("previous_failures", []).append({"error": state.pop("error"), "updated_at": state.get("updated_at")})
    state.update(status="running", pid=os.getpid(), human_review_status="pending", active_child_pid=None)
    state.setdefault("invocations", []).append({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat(),
        "wrapper_sha256": sha256_file(__file__), "runtime_threads": snapshot["runtime_threads"], "explicit_resume": args.resume})
    write_json(root / "results/evaluation_selection_snapshot.json", snapshot)
    (root / "metadata/evaluation_pipeline.pid").write_text(str(os.getpid()) + "\n")

    def update(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)

    def run(stage, command, artifacts=()):
        old = next((entry for entry in reversed(state["commands"]) if entry["stage"] == stage and entry.get("returncode") == 0), None)
        if old:
            receipts = old.get("artifact_sha256")
            if old["arguments"] != command or not isinstance(receipts, dict) or set(receipts) != {str(path) for path in artifacts} or any(not Path(path).is_file() or sha256_file(path) != digest for path, digest in receipts.items()):
                raise RuntimeError(f"Previously completed stage changed: {stage}; review before continuing")
            return
        log = root / f"metadata/eval_{stage}.log"
        entry = {"stage": stage, "arguments": command, "log": str(log)}
        state["commands"].append(entry)
        update(phase=stage)
        started = time.monotonic()
        with log.open("a") as stream:
            process = subprocess.Popen([sys.executable, *command], stdout=stream, stderr=subprocess.STDOUT)
            (root / f"metadata/eval_{stage}.pid").write_text(str(process.pid) + "\n")
            entry["pid"] = process.pid
            update(active_child_pid=process.pid)
            code = process.wait()
        entry.update(returncode=code, elapsed_seconds=time.monotonic() - started)
        update(active_child_pid=None)
        if code:
            raise RuntimeError(f"{stage} failed ({code}); inspect {log}; no automatic retry")
        entry["artifact_sha256"] = {str(path): sha256_file(path) for path in artifacts}
        update()

    candidate = protocol["candidate"]
    common = ["--model-path", snapshot["models"]["base"]["checkpoint"], "--revision", candidate["revision"],
              "--image-root", str(root / "data/images"), "--seed", "42"]
    for field in ("min_pixels", "max_pixels", "recipe_token_budget", "max_input_tokens"):
        common += ["--" + field.replace("_", "-"), str(candidate["input"][field])]
    paths = {experiment: {} for experiment in ("E0", "E3_determinism_1", "E3_determinism_2", "E3", "E4_calibration", "E4")}

    def materialize_control(name):
        control = snapshot["manifests"][name]
        path = Path(control["path"])
        payload = monitor_subset_payload(read_rows(Path(control["source"])), control["count"], name)
        if hashlib.sha256(payload).hexdigest() != control["sha256"]:
            raise RuntimeError(f"Fixed monitor control source changed after the evaluation snapshot: {name}")
        if path.exists() and sha256_file(path) != control["sha256"]:
            raise RuntimeError(f"Existing control differs from fixed monitor-first-N selection: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        return path

    def predict(name, experiment, manifest, questions, recipes, process=False, orbit=False, run_id=None):
        model = snapshot["models"][name]
        output = root / f"runs/evaluation/{name}/{experiment.lower()}.jsonl"
        run_id = run_id or f"eval_{name}"
        command = ["scripts/predict.py", *common, "--manifest", snapshot["manifests"][manifest]["path"],
                   "--output", str(output), "--run-id", run_id, "--max-new-tokens", "256" if process else "32"]
        if model["adapter"]:
            command += ["--adapter-path", model["adapter"]]
        if process:
            command += ["--process"]
        run(f"{experiment}_{name}", command, (output, output.with_suffix(output.suffix + ".config.json")))
        coverage = output.with_suffix(".coverage.json")
        command = ["scripts/validate_run.py", "predictions", "--manifest", snapshot["manifests"][manifest]["path"],
                   "--predictions", str(output), "--run-id", run_id, "--checkpoint", model["checkpoint"], "--seed", "42",
                   "--expected-questions", str(questions), "--expected-recipes", str(recipes), "--output", str(coverage)]
        if orbit:
            command += ["--require-orbit"]
        if process:
            command += ["--prompt-kind", "process"]
        run(f"{experiment}_{name}_coverage", command, (coverage,))
        if not read_json(coverage).get("passed"):
            raise RuntimeError(f"Coverage did not pass: {experiment}/{name}")
        paths[experiment][name] = output

    def summarize(stage, files, directory, orbit=False):
        command = ["scripts/summarize_predictions.py", *map(str, files), "--paired", "--bootstrap-samples", "2000",
                   "--bootstrap-seed", "20261107", "--output-dir", str(directory)]
        if orbit:
            command += ["--orbit"]
        run(stage, command, tuple(directory / name for name in ("summary.json", "per_record.csv", "per_recipe.csv")))
        summary = read_json(directory / "summary.json")
        if len(summary["groups"]) != len(files) or len(summary["paired_comparisons"]) != len(files) * (len(files) - 1) // 2 or any(item["status"] != "computed" for item in summary["paired_comparisons"]):
            raise RuntimeError(f"Incomplete paired comparison: {stage}")
        return directory / "summary.json"

    try:
        update(phase="E0")
        main_models = [name for name in snapshot["models"] if name != "sft_4pass_seed42"]
        for name in main_models:
            predict(name, "E0", "test", 905, 597)
        e0 = summarize("E0_summary", paths["E0"].values(), root / "results/e0_main")
        if "sft_4pass_seed42" in snapshot["models"]:
            predict("sft_4pass_seed42", "E0", "test", 905, 597)
            summarize("E0_sft4pass_sensitivity", [paths["E0"][name] for name in (*PRIMARY, "sft_4pass_seed42")], root / "results/e0_sft4pass_sensitivity")
        update(phase="E2_reuse", E2={"source": str(e0), "additional_GPU_predictions": 0,
                                     "metrics": "six pairs, middle pair, first/last ranks, reversal and invalid/pair-bin distribution"})
        materialize_control("greedy_determinism10")
        determinism_reports = {}
        for name in PRIMARY:
            for repeat in (1, 2):
                predict(name, f"E3_determinism_{repeat}", "greedy_determinism10", 10, 10, run_id=f"eval_{name}_determinism")
            report = root / f"results/e3_determinism/{name}.json"
            run(f"E3_determinism_{name}_comparison", ["ops/compare_greedy.py", "--left", str(paths["E3_determinism_1"][name]),
                "--right", str(paths["E3_determinism_2"][name]), "--expected-inputs", "10", "--output", str(report)], (report,))
            if read_json(report).get("status") != "passed":
                raise RuntimeError(f"Greedy repeat differs for {name}; E3 paused; do not silently change backend")
            determinism_reports[name] = str(report)
        update(E3_determinism={"status": "passed_fixed_10_development_inputs_per_checkpoint", "reports": determinism_reports,
            "additional_GPU_predictions": 60, "backend_modified": False, "scope": "Not proof of all-input or cross-training determinism"})
        for name in PRIMARY:
            predict(name, "E3", "orbit", 60, 60, orbit=True)
        e3 = summarize("E3_summary", paths["E3"].values(), root / "results/e3_main", orbit=True)
        calibration_manifest = materialize_control("process_calibration5")
        calibration_model = "base" if snapshot["W_alias_of_base"] else "w"
        predict(calibration_model, "E4_calibration", "process_calibration5", 5, 5, process=True)
        calibration_dir = root / "results/e4_calibration"
        check_view_against_actual_prompts(read_rows(Path(snapshot["calibration_review"]["path"])), read_rows(paths["E4_calibration"][calibration_model]), process=True)
        run("E4_calibration_blind_packet", ["scripts/summarize_process.py", "--manifest", snapshot["calibration_review"]["path"],
            "--predictions", str(paths["E4_calibration"][calibration_model]), "--output-dir", str(calibration_dir),
            "--image-root", str(root / "data/images"), "--embed-images", "--bootstrap-samples", "2000",
            "--bootstrap-seed", "20261107", "--blind-seed", "20261108"],
            (calibration_dir / "summary.json", calibration_dir / "blind_mapping.private.json", calibration_dir / "blinded/review_items.jsonl"))
        calibration_summary = read_json(calibration_dir / "summary.json")
        if calibration_summary["expected_responses_per_method"] != 5 or len(calibration_summary["groups"]) != 1 or calibration_summary["groups"][0]["automatic"]["status"] != "complete":
            raise RuntimeError("E4 development calibration packet does not cover the five fixed W responses")
        update(E4_calibration={"summary": str(calibration_dir / "summary.json"), "responses": 5,
            "source_split": "val", "excluded_from_main_statistics": True, "human_review_status": "pending; no labels supplied"})
        for name in PRIMARY:
            predict(name, "E4", "process", 40, 40, process=True)
            check_view_against_actual_prompts(read_rows(Path(snapshot["process_review"]["path"])), read_rows(paths["E4"][name]), process=True)
        e4 = root / "results/e4_main"
        run("E4_automatic_and_blind_packet", ["scripts/summarize_process.py", "--manifest", snapshot["process_review"]["path"],
            "--predictions", *map(str, paths["E4"].values()), "--answer-only", *[str(paths["E0"][name]) for name in PRIMARY],
            "--output-dir", str(e4), "--image-root", str(root / "data/images"), "--embed-images",
            "--bootstrap-samples", "2000", "--bootstrap-seed", "20261107", "--blind-seed", "20261108"],
            (e4 / "summary.json", e4 / "blind_mapping.private.json", e4 / "blinded/review_items.jsonl"))
        process_summary = read_json(e4 / "summary.json")
        if len(process_summary["groups"]) != 3 or any(group["automatic"]["status"] != "complete" or group["automatic"]["answer_only_comparison"]["n_available"] != 40 for group in process_summary["groups"]) or len(process_summary["paired_comparisons"]) != 3 or any(item["automatic"]["status"] != "complete" for item in process_summary["paired_comparisons"]):
            raise RuntimeError("E4 automatic coverage or answer-only identity match is incomplete")
        bank = root / "runs/w_candidate_bank/predictions.jsonl"
        bank_coverage = root / "results/e1_w_bank/evaluation_coverage.json"
        run("E1_reuse_coverage", ["scripts/validate_run.py", "predictions", "--manifest", snapshot["manifests"]["bank"]["path"],
            "--predictions", str(bank), "--run-id", "w_candidate_bank", "--checkpoint", protocol["W"]["directory"], "--seed", "42",
            "--samples-per-input", "4", "--expected-questions", "100", "--expected-recipes", "100", "--output", str(bank_coverage)], (bank_coverage,))
        e1 = root / "results/e1_w_bank/summary.json"
        run("E1_reuse_summary", ["scripts/summarize_predictions.py", str(bank), "--candidate-bank", "--candidate-group-size", "4",
            "--output-dir", str(e1.parent)], (e1,))
        training_summary = root / "results/e1_formal_training_evidence.json"
        run("E1_training_logs_reuse", ["scripts/summarize_pilot.py", "--run-dir", *[str(root / "runs" / name) for name in PRIMARY],
            "--output", str(training_summary)], (training_summary,))
        if any(item["status"] != "completed_evidence_summary" or item["issues"] for item in read_json(training_summary)["runs"]):
            raise RuntimeError("Formal training signal/cost evidence is incomplete")
        figures = root / "results/evaluation_figures"
        figure_command = ["scripts/make_figures.py", "--summary", str(e0), str(e3), str(e1), "--output-dir", str(figures)]
        friendly = {"eval_base": "Base", "eval_w": "W", "eval_sft_seed42": "SFT-cont",
                    "eval_exact_seed42": "Exact-GRPO", "eval_pair_seed42": "Pair-GRPO", "w_candidate_bank": "W"}
        for summary_path in (e0, e3, e1):
            for group in read_json(summary_path)["groups"]:
                run_id = group["identity"].get("run_id")
                if run_id not in friendly:
                    raise RuntimeError(f"Unexpected model identity in figure input: {run_id}")
                figure_command += ["--label", f"{summary_path.parent.name}:{group['group_id']}={friendly[run_id]}"]
        run("figures", figure_command, (figures / "figures_manifest.json",))
        if read_json(figures / "figures_manifest.json")["unavailable"]:
            raise RuntimeError("Some required E1/E2/E3 figures are unavailable; inspect figure manifest")
        dynamics = root / "results/training_dynamics"
        run("training_development_dynamics", ["scripts/plot_training_dynamics.py", "--run-dir", *[str(root / "runs" / name) for name in PRIMARY],
            "--pipeline-state", str(root / "metadata/formal_training_pipeline.json"), "--output-dir", str(dynamics)],
            (dynamics / "figures_manifest.json", dynamics / "training_dynamics.json"))
        dynamics_manifest = read_json(dynamics / "figures_manifest.json")
        if dynamics_manifest.get("required_dev_curves_complete") is not True or dynamics_manifest.get("required_signal_curves_complete") is not True or any(item.get("severity", "required") == "required" for item in dynamics_manifest["unavailable"]):
            raise RuntimeError("Required training/development curves are incomplete; inspect training_dynamics/figures_manifest.json")
        update(status="automated_complete_human_pending", phase="E0_E4_automatic_complete", active_child_pid=None,
               human_review_status="pending; no human labels supplied or fabricated",
               optional_training_dynamics_gaps=dynamics_manifest["unavailable"],
               primary_comparisons=["Pair minus Exact", "Pair minus SFT at primary two-pass budgets"],
               next_action="Team completes fixed input audit and two independent E4 blinded review sheets; import labels separately before final conclusions/report")
    except BaseException as error:
        update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
