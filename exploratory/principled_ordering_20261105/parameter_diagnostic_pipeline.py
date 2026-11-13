#!/usr/bin/env python3
"""Bounded read-only parameter-gradient handoff after development diagnostics.

No training, optimizer, confirmation/test access, numerical retry or tolerance
change is authorized here. Only fixed first training records are inspected.
This is a NEW coordinator; it never edits the currently running coordinator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ops"))
from recipeqa.modeling import ensure_data_disk, hash_json, model_weight_identity, read_jsonl, sha256_file, write_json
from wait_and_evaluate import process_info, require_gpu_idle

LOCK_NAMES = ("formal_training_pipeline", "evaluation_pipeline", "evaluation_handoff",
              "seed43_pipeline", "reward_search_20261105")
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
TRAIN_MANIFEST = "data/manifests/train_visits.jsonl"
TRAIN_SHA = "e502e427cdfb0306e9ee8e581ad86ae159f0a7d96de1049716d81bb961622e20"
FROZEN_SHA = "9089f48c5f3b571dfa5fddd31b843a0ae2c22296d9ca4a4d03ee05ab7a75f18c"
PREDECESSOR_POLICY_SHA = "9ee33b61c7021474ee69e7d882b17474da583b918d8a29c6fe5acaedf4cbee9f"
BASE_PATH = "models/Qwen3-VL-2B-Instruct"
BASE_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
BASE_WEIGHT_SHA = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
BASE_DOWNLOAD_RECEIPT_SHA = "6d8b6685c5db0b25120045e93ad409a1fbbc9f7504ad8b15358f965dfd786ef1"
EXPECTED_SOURCES = {
    "exploratory/principled_ordering_20261105/gradient_alignment.py": "1adff082850e0b83a9a70c83565638c02b8efeebdf0fb0ade49af937df9fe41b",
    "exploratory/principled_ordering_20261105/test_gradient_alignment.py": "4e3b9cce80403d406ed86c2f53b8a77a2bc399d6700e37e9a4da34c4f0cd009d",
    "exploratory/principled_ordering_20261105/test_gradient_tensor_reference.py": "e0ff3e958bd0cc28ea69952134c4c765961f38d0cd874348fb5ea980b0291bc0",
    "exploratory/principled_ordering_20261105/score_permutations.py": "affeda80416145d2b6a134172e812d15d48a86d77c67128bd2e5f84e0c341819",
    "exploratory/principled_ordering_20261105/diagnostic_pipeline.py": "58559666220a4e8da5d9df700d4be30bb39f73d3f6f5f320ffa5109ffb86da08",
    "src/recipeqa/modeling.py": "5fedb7a60877e965861aa9174179880754cb0555c4167265fdd8c3781af39771",
    "src/recipeqa/metrics.py": "5287d44a3bb4ede3a44ce0a6625aee399260afecd42c984441a8e07492a2423a",
    "ops/wait_and_evaluate.py": "26c7b4574cb9f970e12dbd5aaa94cc9a587a001abebdc4df8b95bd92b11b0bdb",
}
MODELS = {
    "sft256": {"adapter": "runs/sft_seed42/checkpoint-256",
               "weight_sha256": "9b7b10bef5bd40b0ff0f49c9bb368ae926b8831f41dde2eeb0a598bdb7e05ab0",
               "adapter_config_sha256": "4dc9c20dd856cbff0c1e3dbd8324c1dd6219d4e72b44e2a4200f18257196144f"},
    "W16": {"adapter": "runs/warmup_v1/checkpoint-16",
            "weight_sha256": "5b8768ecc20cfc031ccf21ed8509e66c91170d384c9f747142dec53dcb1ab31a",
            "adapter_config_sha256": "c060590fab68427c8f72f5bace7f3d97b53a60ff5ac0394a2a26c310e64ce24e"},
    "pair1024": {"adapter": "runs/pair_seed42/checkpoint-1024",
                 "weight_sha256": "cd160fb2be8877b2b0a638a22126c45e0b43389067622a1e8fea6129b63f2202",
                 "adapter_config_sha256": "7d8ecf99a1803daaa405fdc6ad0558f5723ed5c3d1fc7d6ea3cdaf697a534b7d"},
}
STAGES = (("sft256", 1), ("sft256", 4), ("W16", 4), ("pair1024", 4))


class BudgetStop(RuntimeError):
    """An explicitly non-score budget gate, not a failed scientific result."""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def root_path(value):
    path = (ROOT / value).resolve()
    require(path.is_relative_to(ROOT), "Path escaped project data root: " + str(path))
    return path


def parent_decision(state, process, expected_pid=None, observed_start=None):
    """Pure handoff rule: committed success AND parent exit are both required."""
    pid = state.get("pid")
    require(isinstance(pid, int) and pid > 1, "Predecessor has no valid parent PID")
    if expected_pid is not None:
        require(pid == expected_pid, "Predecessor parent PID changed")
    if process is not None and observed_start is not None:
        require(process["start_ticks"] == observed_start, "Predecessor PID was reused")
    status = state.get("status")
    require(status != "failed_stopped_no_retry", "Development diagnostics failed; no parameter GPU work")
    require(status in ("waiting_exact_completion", "running", "completed_development_diagnostics"),
            "Unknown predecessor status; no GPU handoff")
    if status != "completed_development_diagnostics":
        require(process is not None and process["state"] not in ("Z", "X"),
                "Predecessor exited before committing diagnostic completion")
        return "wait_completion"
    require(state.get("active_child_pid") is None, "Completed predecessor still declares an active child")
    commands = state.get("commands", [])
    require(len(commands) == 6 and all(entry.get("returncode") == 0 for entry in commands),
            "Predecessor completion lacks all six successful command receipts")
    if process is not None and process["state"] not in ("Z", "X"):
        return "wait_parent_exit"
    return "ready"


def release_locks(handles):
    for handle in reversed(handles):
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


def acquire_lockset(paths, wait_seconds, poll_seconds, on_retry=None):
    """Bounded all-or-release retry; never retain a partial lock set or unlink."""
    deadline = time.monotonic() + wait_seconds
    attempts = 0
    while True:
        attempts += 1
        handles = []
        blocked = None
        try:
            for path in paths:
                handle = path.open("a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    blocked = str(path)
                    break
                except BaseException:
                    handle.close()
                    raise
                handles.append(handle)
            if blocked is None:
                return handles, attempts
        except BaseException:
            release_locks(handles)
            raise
        release_locks(handles)
        if on_retry:
            on_retry(attempts, blocked)
        require(time.monotonic() < deadline, "Timed out acquiring unchanged five lock files: " + blocked)
        time.sleep(min(poll_seconds, max(.01, deadline - time.monotonic())))


def timing_gate(smoke_elapsed_seconds, remaining_stages, remaining_seconds):
    """One fixed conservative extrapolation; never reads reward/accuracy/cosine."""
    require(math.isfinite(smoke_elapsed_seconds) and smoke_elapsed_seconds > 0, "Invalid smoke timing")
    # Each next process handles four questions; including model startup four
    # times deliberately overestimates its fixed startup cost. Add a 50% margin.
    one_stage_estimate = 1.5 * 4 * smoke_elapsed_seconds
    required = remaining_stages * one_stage_estimate
    return {"smoke_full_process_seconds": smoke_elapsed_seconds,
            "fixed_safety_factor": 1.5, "questions_per_remaining_stage": 4,
            "estimated_one_stage_seconds": one_stage_estimate,
            "remaining_fixed_stages": remaining_stages, "estimated_remaining_seconds": required,
            "actual_remaining_seconds": remaining_seconds, "passed": required <= remaining_seconds,
            "selection_inputs": ["elapsed time", "fixed stage count"], "score_based_selection": False}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-self-sha256", required=True, help="Root must bind the reviewed executable at deployment")
    parser.add_argument("--output-dir", type=Path, default=HERE / "parameter_diagnostics")
    parser.add_argument("--wait-seconds", type=float, default=9000)
    parser.add_argument("--parent-exit-wait-seconds", type=float, default=120)
    parser.add_argument("--lock-wait-seconds", type=float, default=120)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    return parser.parse_args()


def verify_gradient_outputs(output, limit, expected_identity, source_hashes):
    config_path = Path(str(output) + ".config.json")
    completion_path = Path(str(output) + ".completion.json")
    config = json.loads(config_path.read_text())
    completion = json.loads(completion_path.read_text())
    rows = read_jsonl(output)
    require(completion.get("status") == "complete_read_only_parameter_geometry"
            and completion.get("rows") == limit and len(rows) == limit, "Incomplete parameter diagnostic rows")
    require(completion.get("output_sha256") == sha256_file(output), "Gradient output completion hash mismatch")
    require(completion.get("all_model_parameters_unchanged") is True
            and completion.get("optimizer_constructed") is False and completion.get("updates") == 0,
            "Diagnostic does not establish zero updates and unchanged model parameters")
    require(completion["in_memory_parameters_before"] == completion["in_memory_parameters_after"],
            "Actual in-memory parameter fingerprints changed")
    require(config["weight_identity"] == expected_identity, "Child used another base/adapter identity")
    require(config["manifest_sha256"] == TRAIN_SHA and config["cli"]["limit"] == limit,
            "Child changed training manifest or fixed prefix")
    require(config["cli"]["logp_atol"] == .01 and config["cli"]["identity_atol"] == 1e-6
            and config["cli"]["identity_rtol"] == 1e-4, "Child numerical tolerance changed")
    expected_child_sources = {
        "exploratory/principled_ordering_20261105/gradient_alignment.py",
        "exploratory/principled_ordering_20261105/score_permutations.py",
        "src/recipeqa/modeling.py", "src/recipeqa/metrics.py",
    }
    require(set(config["source_sha256"]) == expected_child_sources, "Child source receipt is incomplete")
    require(config["source_sha256"] == {name: source_hashes[name] for name in expected_child_sources},
            "Child source hash mismatch")
    signature = hash_json(config)
    require(completion["configuration_sha256"] == signature
            and all(row["configuration_sha256"] == signature for row in rows), "Child config signature mismatch")
    expected_prefix = read_jsonl(root_path(TRAIN_MANIFEST))[:limit]
    require([(row["qid"], row["presentation_id"]) for row in rows]
            == [(row["qid"], row["presentation_id"]) for row in expected_prefix], "Child used a different training prefix")
    require(all(row.get("source_split") == "train" and row.get("manifest_set") == "train_visits"
                and row.get("adapter_unchanged") is True and row.get("parameter_grad_buffers_all_none") is True
                and row.get("gradient_aggregation_dtype") == "torch.float64"
                and row.get("technical_error") is None for row in rows), "Child row audit failed")
    for row in rows:
        require(len(row["candidates"]) == 24 and len(row["candidate_grad_forward_checks"]) == 24,
                "Child did not inspect all 24 candidates")
        require(all(check["passed"] is True and check["relative_l2_gate_passed"] is True
                    for check in row["parameter_gradient_geometry"]["identities"].values()), "Child gradient identity failed")
    return {"rows": len(rows), "configuration_sha256": signature,
            "all_model_parameters_unchanged": True, "optimizer_constructed": False, "updates": 0,
            "artifacts": {str(path.relative_to(ROOT)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                          for path in (output, config_path, completion_path)}}


def main():
    args = arguments()
    os.chdir(ROOT)
    require(ROOT.is_relative_to("/root/autodl-tmp") and ROOT.stat().st_dev != Path("/").stat().st_dev,
            "Pipeline must remain on the separately mounted project data disk")
    ensure_data_disk(ROOT, args.output_dir)
    require(0 < args.wall_seconds <= 1800 and 0 < args.wait_seconds <= 10800
            and 0 < args.parent_exit_wait_seconds <= 300 and 0 < args.lock_wait_seconds <= 300
            and 1 <= args.poll_seconds <= 10, "Invalid bounded wait/wall settings")
    require(sha256_file(__file__) == args.expected_self_sha256, "Pipeline source differs from deployment-bound SHA")
    source_hashes = {**EXPECTED_SOURCES, str(Path(__file__).relative_to(ROOT)): args.expected_self_sha256}
    for name, digest in source_hashes.items():
        require(sha256_file(root_path(name)) == digest, "Reviewed source changed: " + name)
    require(sha256_file(HERE / "DIAGNOSTIC_PROTOCOL.json") == PREDECESSOR_POLICY_SHA, "Predecessor policy changed")
    require(sha256_file(ROOT / "configs/protocol_lock.json") == FROZEN_SHA, "Original protocol lock changed")
    frozen = json.loads((ROOT / "configs/protocol_lock.json").read_text())
    for name, digest in frozen["file_sha256"].items():
        if name.startswith(("src/", "scripts/", "ops/", "configs/")) or name == "RESEARCH_PLAN.md":
            require(sha256_file(root_path(name)) == digest, "Original frozen source changed: " + name)
            source_hashes[name] = digest
    require(sha256_file(root_path(TRAIN_MANIFEST)) == TRAIN_SHA, "Original training manifest changed")
    environment = {key: os.environ.get(key) for key in THREADS}
    os.environ.update({key: "4" for key in THREADS})
    required_env = ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TMPDIR", "TORCH_HOME")
    require(all(os.environ.get(name) for name in required_env), "Data-disk caches/temp environment variables are required")
    ensure_data_disk(*(os.environ[name] for name in required_env))
    destination = args.output_dir.resolve()
    require(destination.is_relative_to(HERE), "Output must remain below this exploratory directory")
    require(not destination.exists(), "Parameter output already exists; no automatic restart")
    own_lock = (ROOT / "metadata/parameter_diagnostic_20261105.lock").open("a+")
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    destination.mkdir()
    (destination / "logs").mkdir()
    state_path = destination / "state.json"
    predecessor_path = HERE / "diagnostics/state.json"
    state = {"status": "waiting_development_diagnostics", "pid": os.getpid(), "started_utc": now(),
             "arguments": sys.argv, "commands": [], "source_sha256": source_hashes,
             "original_protocol_sha256": FROZEN_SHA, "predecessor_policy_sha256": PREDECESSOR_POLICY_SHA,
             "train_manifest_sha256": TRAIN_SHA, "fixed_stages": [list(stage) for stage in STAGES],
             "wall_limit_seconds": args.wall_seconds, "training": False, "optimizer": False,
             "updates": 0, "test_access": False, "confirmation_access": False, "automatic_retry": False,
             "tolerances_changed": False, "runtime_threads": {key: "4" for key in THREADS},
             "original_thread_env": environment}
    locks = []
    active = None
    deadline = None
    expected_predecessor_pid = None
    observed_start = None
    predecessor_sha = None

    def update(**values):
        state.update(values, updated_utc=now())
        write_json(state_path, state)

    def immutable():
        for name, digest in source_hashes.items():
            require(sha256_file(root_path(name)) == digest, "Bound source changed after launch: " + name)
        require(sha256_file(HERE / "DIAGNOSTIC_PROTOCOL.json") == PREDECESSOR_POLICY_SHA, "Predecessor policy changed after launch")
        require(sha256_file(ROOT / "configs/protocol_lock.json") == FROZEN_SHA, "Original protocol lock changed after launch")
        require(sha256_file(root_path(TRAIN_MANIFEST)) == TRAIN_SHA, "Training manifest changed after launch")
        if predecessor_sha is not None:
            require(sha256_file(predecessor_path) == predecessor_sha, "Committed predecessor state changed")

    def on_signal(signum, frame):
        raise RuntimeError("Pipeline interrupted by signal " + str(signum))

    previous_signals = {number: signal.signal(number, on_signal) for number in (signal.SIGTERM, signal.SIGINT)}

    def kill_active():
        nonlocal active
        if active is not None and active.poll() is None:
            # Only our own newly-created process group is signalled.
            try:
                os.killpg(active.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            active.wait(timeout=10)

    def run(stage, command, timeout, *, cpu=False):
        nonlocal active
        immutable()
        require_gpu_idle()
        log = destination / "logs" / (stage + ".log")
        entry = {"stage": stage, "arguments": command, "started_utc": now(), "timeout_seconds": timeout,
                 "cpu_only": cpu, "log": str(log.relative_to(ROOT))}
        state["commands"].append(entry)
        started = time.monotonic()
        try:
            env = dict(os.environ)
            if cpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            elif deadline is not None:
                timeout = min(timeout, deadline - time.monotonic())
                if timeout <= 0:
                    raise BudgetStop("GPU deadline expired during immutable/idle preflight")
                entry["timeout_seconds"] = timeout
            with log.open("x") as stream:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                entry["pid"] = active.pid
                update(status="running_cpu_tests" if cpu else "running_parameter_diagnostic", phase=stage,
                       active_child_pid=active.pid)
                actual_timeout = timeout if cpu else min(timeout, deadline - time.monotonic())
                entry["returncode"] = active.wait(timeout=max(.001, actual_timeout))
            require(entry["returncode"] == 0, stage + " failed; no retry or tolerance adjustment")
        except subprocess.TimeoutExpired as error:
            kill_active()
            entry.update(returncode=active.returncode if active is not None else None,
                         error=f"{type(error).__name__}: {error}")
            if not cpu:
                raise BudgetStop("GPU deadline reached; own child killed, no retry") from error
            raise
        except BaseException as error:
            kill_active()
            entry.update(returncode=active.returncode if active is not None else None,
                         error=f"{type(error).__name__}: {error}")
            raise
        finally:
            entry.update(finished_utc=now(), elapsed_seconds=time.monotonic() - started)
            if log.exists():
                entry["log_sha256"] = sha256_file(log)
            update(active_child_pid=None)
            active = None
        return entry

    try:
        update()
        wait_started = time.monotonic()
        exit_wait_started = None
        while True:
            immutable()
            require(time.monotonic() - wait_started <= args.wait_seconds, "Timed out waiting for development diagnostics")
            if not predecessor_path.exists():
                time.sleep(args.poll_seconds)
                continue
            predecessor = json.loads(predecessor_path.read_text())
            pid = predecessor.get("pid")
            require(isinstance(pid, int) and pid > 1, "Invalid predecessor PID")
            process = process_info(pid)
            if expected_predecessor_pid is None:
                expected_predecessor_pid = pid
            if process is not None and process["state"] not in ("Z", "X"):
                require(process["cwd"] == str(ROOT)
                        and any(Path(arg).name == "diagnostic_pipeline.py" for arg in process["argv"]),
                        "Predecessor PID is not this project's development coordinator")
                if observed_start is None:
                    observed_start = process["start_ticks"]
            action = parent_decision(predecessor, process, expected_predecessor_pid, observed_start)
            update(status="waiting_parent_exit" if action == "wait_parent_exit" else "waiting_development_diagnostics",
                   predecessor_pid=pid, predecessor_start_ticks=observed_start,
                   predecessor_status=predecessor.get("status"), predecessor_process=process,
                   wait_elapsed_seconds=time.monotonic() - wait_started)
            if action in ("wait_parent_exit", "ready") and predecessor_sha is None:
                predecessor_sha = sha256_file(predecessor_path)
                state["predecessor_completed_state_sha256"] = predecessor_sha
            if action == "ready":
                break
            if action == "wait_parent_exit":
                if exit_wait_started is None:
                    exit_wait_started = time.monotonic()
                require(time.monotonic() - exit_wait_started <= args.parent_exit_wait_seconds,
                        "Completed predecessor did not exit; do not signal or bypass it")
            time.sleep(args.poll_seconds)
        immutable()
        update(status="waiting_five_locks", phase="predecessor_completed_and_exited")
        locks, attempts = acquire_lockset(
            [ROOT / "metadata" / (name + ".lock") for name in LOCK_NAMES],
            args.lock_wait_seconds, min(2, args.poll_seconds),
            lambda attempt, blocked: update(lock_attempts=attempt, blocked_lock=blocked),
        )
        require_gpu_idle()
        update(status="checking_model_identities", lock_names=list(LOCK_NAMES), lock_attempts=attempts)
        # Primary weight bytes are pinned independently of the model's own metadata.
        base_identity = model_weight_identity(root_path(BASE_PATH))
        require(len(base_identity["weight_files"]) == 1
                and base_identity["weight_files"][0]["sha256"] == BASE_WEIGHT_SHA, "Base model weight changed")
        receipt_path = root_path(BASE_PATH) / "DOWNLOAD_COMPLETE.json"
        require(sha256_file(receipt_path) == BASE_DOWNLOAD_RECEIPT_SHA, "Original base download receipt changed")
        download = json.loads(receipt_path.read_text())
        require(download["status"] == "complete", "Original base download was incomplete")
        # Pin tokenizer/chat/processor/model configuration as well as weights.
        for name, record in download["files"].items():
            path = root_path(BASE_PATH) / name
            require(path.resolve().is_relative_to(root_path(BASE_PATH)), "Download receipt path escaped base directory")
            require(path.stat().st_size == record["bytes"] and sha256_file(path) == record["verified_sha256"],
                    "Base snapshot file changed: " + name)
        state["base_download_receipt_sha256"] = BASE_DOWNLOAD_RECEIPT_SHA
        identities = {}
        for name, model in MODELS.items():
            adapter = root_path(model["adapter"])
            identity = model_weight_identity(adapter, adapter=True)
            require(len(identity["weight_files"]) == 1
                    and identity["weight_files"][0]["sha256"] == model["weight_sha256"], name + " adapter weight changed")
            require(sha256_file(adapter / "adapter_config.json") == model["adapter_config_sha256"], name + " adapter config changed")
            identities[name] = {"base": base_identity, "adapter": identity}
        # Record actual original pixels for identical fixed-prefix comparisons.
        pixel_sha = {}
        for record in read_jsonl(root_path(TRAIN_MANIFEST))[:4]:
            for relative in record["image_paths_display"]:
                path = root_path("data/images/" + relative)
                pixel_sha[str(path.relative_to(ROOT))] = sha256_file(path)
        state.update(weight_identities=identities, fixed_prefix_image_sha256=pixel_sha)
        update()
        cpu_report = destination / "cpu_tests.json"
        cpu_code = (
            "import json,pathlib,sys,unittest; "
            "suite=unittest.defaultTestLoader.discover(sys.argv[1],pattern='test_gradient*.py'); "
            "r=unittest.TextTestRunner(verbosity=2).run(suite); "
            "p={'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'successful':r.wasSuccessful()}; "
            "pathlib.Path(sys.argv[2]).write_text(json.dumps(p,indent=2)+'\\n'); "
            "sys.exit(0 if r.wasSuccessful() and r.testsRun==8 and not r.skipped else 1)"
        )
        cpu_entry = run("cpu_gradient_reference_8", [sys.executable, "-c", cpu_code, str(HERE), str(cpu_report)], 120, cpu=True)
        cpu = json.loads(cpu_report.read_text())
        require(cpu == {"tests_run": 8, "failures": 0, "errors": 0, "skipped": 0, "successful": True},
                "Eight actual CPU tensor/reference tests must pass without skips")
        cpu_entry["artifacts"] = {str(cpu_report.relative_to(ROOT)): {"sha256": sha256_file(cpu_report), "bytes": cpu_report.stat().st_size}}
        deadline = time.monotonic() + args.wall_seconds
        update(gpu_budget_started_utc=now(), cpu_reference_status="8_passed_zero_skipped")
        smoke_elapsed = None
        for stage_index, (name, limit) in enumerate(STAGES):
            immutable()
            require(model_weight_identity(root_path(BASE_PATH)) == base_identity, "Base model/processor bundle changed")
            require(model_weight_identity(root_path(MODELS[name]["adapter"]), adapter=True) == identities[name]["adapter"],
                    name + " adapter bundle changed after launch")
            for relative, digest in pixel_sha.items():
                require(sha256_file(root_path(relative)) == digest, "Training-prefix image bytes changed")
            remaining = deadline - time.monotonic()
            if stage_index:
                gate = timing_gate(smoke_elapsed, len(STAGES) - stage_index, remaining)
                state.setdefault("timing_gates", []).append(gate)
                update()
                if not gate["passed"]:
                    raise BudgetStop("Fixed smoke-based timing estimate exceeds remaining 1800-second budget; no score-based subset")
            if remaining <= 1:
                raise BudgetStop("Read-only GPU wall budget exhausted")
            tag = f"{name}_train{limit}"
            output = destination / tag / "gradients.jsonl"
            output.parent.mkdir()
            command = [sys.executable, str(HERE / "gradient_alignment.py"),
                       "--model-path", str(root_path(BASE_PATH)), "--revision", BASE_REVISION,
                       "--adapter-path", str(root_path(MODELS[name]["adapter"])),
                       "--manifest", str(root_path(TRAIN_MANIFEST)), "--image-root", str(ROOT / "data/images"),
                       "--output", str(output), "--run-id", "principled_parameter_" + tag, "--limit", str(limit),
                       "--logp-atol", "0.01", "--identity-atol", "0.000001", "--identity-rtol", "0.0001"]
            entry = run(tag, command, remaining)
            entry["verified_output"] = verify_gradient_outputs(output, limit, identities[name], source_hashes)
            if stage_index == 0:
                smoke_elapsed = entry["elapsed_seconds"]
                state["smoke_elapsed_seconds"] = smoke_elapsed
            require_gpu_idle()
            update(gpu_wall_elapsed_seconds=args.wall_seconds - max(0, deadline - time.monotonic()))
        immutable()
        update(status="completed_parameter_diagnostics", phase="all_fixed_training_prefix_models_complete",
               finished_utc=now(), active_child_pid=None, gpu_wall_elapsed_seconds=args.wall_seconds - (deadline - time.monotonic()),
               fixed_gradient_rows=13, unique_training_questions=4, optimizer_constructed=False, updates=0)
    except BudgetStop as error:
        kill_active()
        update(status="stopped_budget", error=str(error), finished_utc=now(), active_child_pid=None,
               budget_gate_not_scientific_failure=True)
    except BaseException as error:
        kill_active()
        update(status="failed_stopped_no_retry", error=f"{type(error).__name__}: {error}", finished_utc=now(), active_child_pid=None)
        raise
    finally:
        release_locks(locks)
        fcntl.flock(own_lock, fcntl.LOCK_UN)
        own_lock.close()
        for number, handler in previous_signals.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    main()
