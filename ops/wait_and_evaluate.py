#!/usr/bin/env python3
"""Observe one live P6 parent; evaluate only after its committed completion and exit.

This never starts/restarts training or automatically resumes evaluation. A failed
or unaccountably vanished P6 stops the handoff and leaves an explicit record.
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

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
PRIMARY = {"sft_seed42", "exact_seed42", "pair_seed42"}
EVALUATION_SOURCES = ("ops/evaluation_pipeline.py", "ops/compare_greedy.py", "scripts/build_review_views.py", "scripts/plot_training_dynamics.py")


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_info(pid):
    directory = Path(f"/proc/{pid}")
    try:
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return {"state": fields[0], "start_ticks": fields[19],
                "argv": (directory / "cmdline").read_bytes().decode().split("\0")[:-1],
                "cwd": str((directory / "cwd").resolve(strict=True)) if fields[0] not in ("Z", "X") else None}
    except FileNotFoundError:
        return None


def decision(formal, process, expected_pid, observed_start):
    if formal.get("pid") != expected_pid:
        return "stop", "P6 metadata PID changed; this handoff follows only its original parent"
    status = formal.get("status")
    if status == "failed":
        return "stop", "P6 explicitly failed; inspect its error and child state; no training restart"
    if status not in ("running", "completed"):
        return "stop", f"Unknown P6 status: {status}"
    if process and observed_start is not None and process["start_ticks"] != observed_start:
        return "stop", "Parent PID was reused; original process identity is no longer established"
    if status == "completed" and (formal.get("phase") != "three_branches_selected" or not PRIMARY.issubset(formal.get("selected", {})) or not formal.get("commands") or any(command.get("returncode") != 0 for command in formal["commands"])):
        return "stop", "Completed P6 lacks its final phase, selections, or successful command receipts"
    if process and process["state"] not in ("Z", "X"):
        return "wait", "P6 parent still live; GPU handoff is forbidden until it exits"
    if observed_start is None:
        return "stop", "Original live parent was never observed; inspect completion and launch evaluation manually"
    if status == "completed":
        return "ready", "P6 committed all selections and its observed parent exited"
    return "stop", "Parent disappeared before committed P6 completion; child state is unknown"


def require_gpu_idle():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=15)
    if result.returncode or result.stdout.strip():
        raise RuntimeError(f"GPU idle state not established; no evaluation launched: rc={result.returncode}, processes={result.stdout.strip()}, stderr={result.stderr.strip()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/rl_recipeqa"))
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20)
    parser.add_argument("--max-wait-hours", type=float, default=24)
    parser.add_argument("--check-only", action="store_true", help="Read-only arming check; never wait or launch evaluation")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_relative_to("/root/autodl-tmp") or root.stat().st_dev == Path("/").stat().st_dev:
        raise ValueError("A separately mounted data-disk project is required")
    if not 1 <= args.poll_seconds <= 60 or args.max_wait_hours <= 0 or args.parent_pid <= 1:
        raise ValueError("Require parent PID >1, poll interval 1..60 seconds and positive wait budget")
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    from recipeqa.modeling import write_json
    original_env = {key: os.environ.get(key) for key in THREAD_VARS}
    os.environ.update({key: "4" for key in THREAD_VARS})
    policy = read_json(root / "metadata/runtime_threads.json")
    if any(policy.get(key) != 4 for key in THREAD_VARS):
        raise RuntimeError("Runtime policy is not uniformly four threads; inspect it without editing frozen sources")
    formal_path = root / "metadata/formal_training_pipeline.json"
    protocol_path = root / "configs/protocol_lock.json"
    wrapper = root / "ops/evaluation_pipeline.py"
    formal = read_json(formal_path)
    protocol_sha = sha256(protocol_path)
    if formal.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("P6 does not match the frozen protocol")
    parent = process_info(args.parent_pid)
    observed_start = parent["start_ticks"] if parent and parent["state"] not in ("Z", "X") else None
    action, reason = decision(formal, parent, args.parent_pid, observed_start)
    if observed_start is not None and (parent["cwd"] != str(root) or not any(Path(arg).name == "formal_training_pipeline.py" for arg in parent["argv"])):
        action, reason = "stop", "Parent PID is not this project's formal training pipeline"
    if args.check_only:
        print(json.dumps({"status": "not_armed" if action == "stop" else "ready_to_wait", "parent_pid": args.parent_pid, "phase": formal.get("phase"), "reason": reason}), flush=True)
        return 2 if action == "stop" else 0
    lock = (root / "metadata/evaluation_handoff.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "metadata/evaluation_handoff.json"
    if state_path.exists() or (root / "metadata/evaluation_pipeline.json").exists():
        raise RuntimeError("A handoff/evaluation record already exists; inspect it instead of automatically restarting")
    state = {"status": "waiting", "pid": os.getpid(), "parent_pid": args.parent_pid,
             "parent_start_ticks": observed_start, "protocol_sha256": protocol_sha,
             "wrapper_sha256": sha256(wrapper), "handoff_sha256": sha256(Path(__file__)),
             "evaluation_source_sha256": {name: sha256(root / name) for name in EVALUATION_SOURCES},
             "started_at": datetime.now(timezone.utc).isoformat(), "poll_seconds": args.poll_seconds,
             "original_env": original_env, "runtime_threads": {key: "4" for key in THREAD_VARS},
             "automatic_training_restart": False, "automatic_evaluation_resume": False}

    def update(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(state_path, state)

    (root / "metadata/evaluation_handoff.pid").write_text(str(os.getpid()) + "\n")
    started = time.monotonic()
    try:
        if action == "stop":
            raise RuntimeError(reason)
        while True:
            formal, parent = read_json(formal_path), process_info(args.parent_pid)
            if formal.get("protocol_sha256") != protocol_sha or sha256(protocol_path) != protocol_sha or any(sha256(root / name) != digest for name, digest in state["evaluation_source_sha256"].items()):
                raise RuntimeError("Protocol or reviewed evaluation wrapper changed while waiting")
            action, reason = decision(formal, parent, args.parent_pid, observed_start)
            update(status="waiting" if action == "wait" else "checking" if action == "ready" else "stopped",
                   last_p6_status=formal.get("status"), last_p6_phase=formal.get("phase"),
                   parent_state=parent["state"] if parent else "absent", reason=reason,
                   elapsed_wait_seconds=time.monotonic() - started)
            if action == "stop":
                raise RuntimeError(reason)
            if action == "ready":
                break
            if time.monotonic() - started >= args.max_wait_hours * 3600:
                raise RuntimeError("Handoff wait budget reached; training is not stopped or restarted")
            time.sleep(args.poll_seconds)
        command = [sys.executable, str(wrapper), "--root", str(root)]
        checked = subprocess.run([*command, "--check-only"], capture_output=True, text=True, timeout=180)
        update(readiness_returncode=checked.returncode, readiness_stdout=checked.stdout, readiness_stderr=checked.stderr)
        if checked.returncode:
            raise RuntimeError("Evaluation preflight failed; no GPU evaluation launched")
        require_gpu_idle()
        if (root / "metadata/evaluation_pipeline.json").exists():
            raise RuntimeError("Another evaluation record appeared during handoff; no second job launched")
        log = root / "metadata/evaluation_pipeline.stdout.log"
        with log.open("a") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            update(status="evaluation_running", evaluation_pid=child.pid, evaluation_command=command, evaluation_log=str(log))
            code = child.wait()
        update(evaluation_returncode=code)
        if code:
            raise RuntimeError(f"Evaluation exited {code}; inspect its log; no automatic retry")
        evaluation = read_json(root / "metadata/evaluation_pipeline.json")
        if evaluation.get("status") != "automated_complete_human_pending":
            raise RuntimeError(f"Evaluation stopped: returncode={code}, status={evaluation.get('status')}; no automatic retry")
        update(status="automated_complete_human_pending", evaluation_returncode=code, human_review_status="pending")
        print(json.dumps({"status": state["status"], "state": str(state_path)}), flush=True)
        return 0
    except BaseException as error:
        update(status="stopped", error=f"{type(error).__name__}: {error}")
        print(json.dumps({"status": "stopped", "reason": state["error"], "state": str(state_path)}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
