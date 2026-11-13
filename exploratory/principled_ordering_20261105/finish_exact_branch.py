#!/usr/bin/env python3
"""Finish the already-running Exact arm only; never train, resume or access test.

The old eight-arm parent was stopped because the user rejected heuristic methods.
Preserve that state and its policy; after the Exact child finishes naturally,
retire only the identified stopped parent and evaluate its four saved checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
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
OLD = ROOT / "exploratory/reward_search_20261105"
DEST = HERE / "exact_completion"
POLICY_SHA = "099bd40a0cb3fae390e2e85d88c4244e0e6624380b66a92e34e475b996fa5d2d"
FROZEN_SHA = "9089f48c5f3b571dfa5fddd31b843a0ae2c22296d9ca4a4d03ee05ab7a75f18c"
REASON = ("New user instruction: no heuristic methods; preserve current Exact branch, "
          "prevent pending exponential/gated reward training. Not stopped for poor scores.")
LOCKS = ("formal_training_pipeline", "evaluation_pipeline", "evaluation_handoff",
         "seed43_pipeline", "reward_search_20261105")
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def process_info(pid):
    try:
        base = Path("/proc") / str(pid)
        fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
        state = fields[0]
        return {"pid": pid, "state": state, "start_ticks": fields[19],
                "argv": (base / "cmdline").read_bytes().decode().split("\0")[:-1],
                "cwd": str((base / "cwd").resolve(strict=True)) if state not in ("Z", "X") else None}
    except FileNotFoundError:
        return None


def compute_pids(allowed=()):
    command = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    require(result.returncode == 0, "Cannot establish GPU process state: " + result.stderr.strip())
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(all(line.isdigit() for line in lines), "Unrecognized nvidia-smi process output")
    pids = set(map(int, lines))
    require(pids <= set(allowed), f"Unexpected GPU compute processes: {sorted(pids - set(allowed))}")
    return {"arguments": command, "returncode": result.returncode, "pids": sorted(pids), "utc": now()}


def root_path(value):
    path = (ROOT / value).resolve()
    require(path.is_relative_to(ROOT), f"Path escaped project data root: {path}")
    return path


def artifacts(paths):
    return {str(path.relative_to(ROOT)): {"sha256": sha(path), "bytes": path.stat().st_size}
            for path in map(Path, paths)}


def check_sources():
    require(sha(OLD / "SCREEN_PROTOCOL.json") == POLICY_SHA, "Old locked screening policy changed")
    policy = load(OLD / "SCREEN_PROTOCOL.json")
    require(policy["status"] == "locked" and policy["root"] == str(ROOT) and policy["seed"] == 44,
            "Wrong old policy identity")
    frozen = ROOT / "configs/protocol_lock.json"
    require(sha(frozen) == FROZEN_SHA, "Original frozen lock metadata changed")
    lock = load(frozen)
    require(lock["status"] == "frozen", "Original protocol is not frozen")
    sources = {str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__),
               str(frozen.relative_to(ROOT)): FROZEN_SHA,
               str((OLD / "SCREEN_PROTOCOL.json").relative_to(ROOT)): POLICY_SHA}
    for name, digest in policy["source_sha256"].items():
        require(sha(root_path(name)) == digest, "Old exploratory source changed: " + name)
        sources[name] = digest
    for name, digest in lock["file_sha256"].items():
        if name.startswith(("src/", "scripts/", "ops/", "configs/")) or name == "RESEARCH_PLAN.md":
            require(sha(root_path(name)) == digest, "Original frozen source changed: " + name)
            sources[name] = digest
    for field in ("train_manifest", "dev_manifest", "original_train_manifest"):
        require("test" not in Path(policy[field]).parts and "test" not in Path(policy[field]).name.lower(),
                "Test data access is forbidden")
        require(sha(root_path(policy[field])) == policy[field + "_sha256"], "Manifest changed: " + field)
    require(sha(root_path(policy["initial_adapter"]) / "adapter_model.safetensors") ==
            policy["initial_adapter_weight_sha256"], "Common initial SFT weight changed")
    require(sha(root_path(policy["baseline"]["predictions"])) == policy["baseline"]["predictions_sha256"],
            "Common initial development predictions changed")
    arms = [arm for arm in policy["arms"] if arm["name"] == "exact_g8"]
    require(len(arms) == 1 and arms[0]["reward"] == "exact" and arms[0]["max_steps"] == 128
            and arms[0]["save_steps"] == 32, "Exact arm identity changed")
    return policy, sources


def inspect_old(parent_pid, child_pid, expected_pause_sha=None):
    pause_path = OLD / "user_no_heuristics_pause.json"
    pause = load(pause_path)
    require(pause["reason"] == REASON and pause["parent_pid"] == parent_pid
            and pause["current_child_pid"] == child_pid and pause["phase_at_pause"] == "exact_g8_train"
            and pause["original_state_preserved"] is True and pause["no_training_restart"] is True,
            "Missing or inconsistent user-constraint pause receipt")
    require(expected_pause_sha is None or sha(pause_path) == expected_pause_sha, "Pause receipt changed")
    state_path = OLD / "screen/state.json"
    state = load(state_path)
    require(state["pid"] == parent_pid and state["active_child_pid"] == child_pid
            and state["phase"] == "exact_g8_train" and state["status"] == "running"
            and state["policy_sha256"] == POLICY_SHA, "Old paused state advanced or changed")
    active = [item for item in state["commands"] if item["stage"] == "exact_g8_train"]
    require(len(active) == 1 and active[0]["pid"] == child_pid
            and str(OLD / "train_reward_search.py") in active[0]["arguments"], "Old child command identity missing")
    require("exp_inv_g8" not in state.get("selected", {})
            and not any(item["stage"].startswith(("exp_inv_", "success_gate_")) for item in state["commands"]),
            "A heuristic arm already launched; manual review required")
    parent = process_info(parent_pid)
    require(parent and parent["state"] in ("T", "t") and parent["cwd"] == str(ROOT)
            and str(OLD / "screen_pipeline.py") in parent["argv"], "Old parent is not the identified stopped coordinator")
    child = process_info(child_pid)
    if child and child["state"] not in ("Z", "X"):
        require(child["cwd"] == str(ROOT) and str(OLD / "train_reward_search.py") in child["argv"],
                "Child PID was reused or is not Exact training")
    monitor_pid = pause.get("obsolete_completion_monitor_retired_pid")
    require(monitor_pid == 48343, "Expected obsolete monitor retirement is not recorded")
    monitor = process_info(monitor_pid)
    require(not monitor or monitor["state"] in ("Z", "X"), "Obsolete completion monitor is unexpectedly live")
    run = OLD / "screen/arms/exact_g8"
    outcome_path = run / "train_outcome.json"
    outcome = load(outcome_path) if outcome_path.exists() else None
    if outcome:
        require(outcome.get("status") == "completed" and outcome.get("global_step") == 128,
                "Exact outcome is terminal but not completed at 128; no automatic restart")
    terminal = not child or child["state"] in ("Z", "X")
    require(not terminal or outcome is not None, "Exact child ended without completed outcome")
    gpu = compute_pids(() if terminal else (child_pid,))
    return {"status": "ready" if terminal else "waiting", "parent": parent, "child": child,
            "old_state_sha256": sha(state_path), "pause_sha256": sha(pause_path),
            "outcome": outcome, "gpu": gpu, "old_phase": state["phase"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, default=46164)
    parser.add_argument("--child-pid", type=int, default=48008)
    parser.add_argument("--wait-seconds", type=int, default=1800)
    parser.add_argument("--evaluation-seconds", type=int, default=1800)
    parser.add_argument("--expected-self-sha256")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    require(args.parent_pid == 46164 and args.child_pid == 48008, "This one-off follows only the identified old processes")
    require(1 <= args.wait_seconds <= 1800 and 1 <= args.evaluation_seconds <= 1800, "Budget exceeds 30 minutes per phase")
    require(ROOT.is_relative_to("/root/autodl-tmp") and ROOT.stat().st_dev != Path("/").stat().st_dev,
            "Project and all artifacts must remain on the separately mounted data disk")
    require(args.expected_self_sha256 is None or sha(__file__) == args.expected_self_sha256,
            "Coordinator differs from reviewed source")
    os.chdir(ROOT)
    policy, sources = check_sources()
    observed = inspect_old(args.parent_pid, args.child_pid)
    if args.check_only:
        print(json.dumps({**observed, "check_only": True, "output_exists": DEST.exists(),
                          "test_access": False, "training_restart": False, "source_sha256": sources}, indent=2))
        return
    require(not DEST.exists(), "Completion output exists; inspect instead of resume/restart")
    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TMPDIR", "TORCH_HOME"):
        require(key in os.environ and Path(os.environ[key]).resolve().is_relative_to(ROOT),
                "Source the existing data-disk env.sh first: " + key)
    original_env = {key: os.environ.get(key) for key in THREADS}
    os.environ.update({key: "4" for key in THREADS})
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    own_lock = (HERE / "finish_exact_branch.lock").open("a+")
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    DEST.mkdir()
    (DEST / "logs").mkdir()
    state = {"status": "waiting", "pid": os.getpid(), "started_utc": now(), "commands": [],
             "training_restart": False, "test_access": False, "heuristic_training": False,
             "automatic_promotion": False, "original_policy_sha256": POLICY_SHA,
             "source_sha256": sources, "initial_observation": observed,
             "runtime_threads": {key: "4" for key in THREADS}, "original_env": original_env}
    write(DEST / "launch_receipt.json", {**state, "arguments": sys.argv,
          "wait_limit_seconds": args.wait_seconds, "evaluation_limit_seconds": args.evaluation_seconds,
          "external_stop_reason": REASON, "old_state_preserved": True})
    locks = []

    def update(**values):
        state.update(values, updated_utc=now())
        write(DEST / "state.json", state)

    def immutable():
        for name, digest in sources.items():
            require(sha(root_path(name)) == digest, "Bound source changed after launch: " + name)
        require(sha(OLD / "screen/state.json") == observed["old_state_sha256"], "Old state was modified")
        require(sha(OLD / "user_no_heuristics_pause.json") == observed["pause_sha256"], "Pause receipt changed")
        for field in ("train_manifest", "dev_manifest", "original_train_manifest"):
            require(sha(root_path(policy[field])) == policy[field + "_sha256"], "Bound manifest changed")

    def run(stage, arguments, required):
        immutable()
        remaining = deadline - time.monotonic()
        require(remaining > 1, "Exact-only evaluation budget exhausted")
        idle = compute_pids()
        log = DEST / "logs" / (stage + ".log")
        entry = {"stage": stage, "arguments": [sys.executable, *map(str, arguments)],
                 "started_utc": now(), "timeout_seconds": remaining, "log": str(log), "gpu_before": idle}
        state["commands"].append(entry)
        child = None
        started = time.monotonic()
        try:
            with log.open("x") as stream:
                child = subprocess.Popen(entry["arguments"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                entry.update(pid=child.pid, process_at_launch=process_info(child.pid))
                update(status="evaluating", phase=stage, active_child_pid=child.pid)
                entry["returncode"] = child.wait(timeout=max(.1, deadline - time.monotonic()))
            require(entry["returncode"] == 0, stage + " failed; inspect log, no automatic retry")
            entry["required_artifacts"] = artifacts(required)
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
            if child is not None:
                entry["returncode"] = child.returncode
            entry["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            entry.update(finished_utc=now(), elapsed_seconds=time.monotonic() - started,
                         process_after=process_info(child.pid) if child is not None else None)
            if log.exists():
                entry["log_sha256"] = sha(log)
            update(active_child_pid=None)

    try:
        update()
        wait_started = time.monotonic()
        while True:
            immutable()
            current = inspect_old(args.parent_pid, args.child_pid, observed["pause_sha256"])
            require(current["parent"]["start_ticks"] == observed["parent"]["start_ticks"], "Parent PID reused")
            if observed["child"] and current["child"]:
                require(current["child"]["start_ticks"] == observed["child"]["start_ticks"], "Child PID reused")
            update(last_observation=current, elapsed_wait_seconds=time.monotonic() - wait_started)
            if current["status"] == "ready":
                break
            require(time.monotonic() - wait_started < args.wait_seconds, "Exact-child wait limit; parent remains safely paused")
            time.sleep(min(10, max(.1, args.wait_seconds - (time.monotonic() - wait_started))))
        deadline = time.monotonic() + args.evaluation_seconds
        retirement = {"utc": now(), "reason": REASON, "old_state_sha256": observed["old_state_sha256"],
                      "before": current, "signals": [], "old_state_preserved": True,
                      "old_lock_files_unlinked": False, "old_training_child_signalled": False}
        write(DEST / "old_parent_retirement.json", retirement)
        # SIGTERM is queued while stopped; SIGCONT permits delivery of the pending
        # default termination, rather than allowing the obsolete queue to continue.
        for sig in (signal.SIGTERM, signal.SIGCONT):
            parent = process_info(args.parent_pid)
            if not parent or parent["state"] in ("Z", "X"):
                break
            require(parent["start_ticks"] == observed["parent"]["start_ticks"], "Parent PID reused before retirement")
            os.kill(args.parent_pid, sig)
            retirement["signals"].append({"signal": sig.name, "utc": now()})
            write(DEST / "old_parent_retirement.json", retirement)
        retired_deadline = time.monotonic() + 30
        while True:
            parent = process_info(args.parent_pid)
            if not parent or parent["state"] in ("Z", "X"):
                break
            require(parent["start_ticks"] == observed["parent"]["start_ticks"], "Parent PID reused during retirement")
            require(time.monotonic() < retired_deadline, "Old coordinator did not retire; no escalation or evaluation")
            time.sleep(.5)
        retirement.update(after=parent, finished_utc=now())
        write(DEST / "old_parent_retirement.json", retirement)
        immutable()
        compute_pids()
        # Parent exit releases flock descriptors. Never unlink/recreate lock files.
        for name in LOCKS:
            handle = (ROOT / "metadata" / (name + ".lock")).open("a+")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        update(status="checking", phase="old_parent_retired_locks_acquired", lock_names=list(LOCKS))
        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(ROOT / "scripts"))
        from recipeqa.metrics import score_order
        from recipeqa.modeling import model_weight_identity, read_jsonl
        from validate_run import validate_prediction_coverage
        manifest = root_path(policy["dev_manifest"])
        dev_rows = read_jsonl(manifest)
        require(len(dev_rows) == len({row["recipe_id"] for row in dev_rows}) == 200, "Wrong development denominator")
        baseline = root_path(policy["baseline"]["predictions"])
        require(sha(baseline) == policy["baseline"]["predictions_sha256"], "Baseline bytes changed")
        initial = root_path(policy["initial_adapter"])

        def score_candidate(path, checkpoint, step, expected_run_id=None, expected_seed=None):
            rows = read_jsonl(path)
            coverage = validate_prediction_coverage(dev_rows, rows, expected_questions=200, expected_recipes=200,
                        expected_checkpoint=str(checkpoint), expected_run_id=expected_run_id, expected_seed=expected_seed)
            require(coverage["passed"], "Independent development coverage failed")
            config = load(path.with_suffix(path.suffix + ".config.json"))
            identity = model_weight_identity(checkpoint, adapter=True)
            require(config["manifest_sha256"] == policy["dev_manifest_sha256"]
                    and config["weight_identity"]["adapter"] == identity, "Development config/model identity mismatch")
            require(all(not row["decoding"]["do_sample"] and row["decoding"]["max_new_tokens"] == 32 for row in rows),
                    "Development inference is not genuine greedy32")
            scores = [score_order(row["raw_response"], row["display_perm"], row["target_order"]) for row in rows]
            require(all(row["score"] == result for row, result in zip(rows, scores)), "Raw development rescore disagrees")
            metrics = {key: sum(result[key] for result in scores) / 200 for key in ("exact", "pair", "valid")}
            require(all(math.isfinite(value) for value in metrics.values()), "Nonfinite development metric")
            return {"step": step, "checkpoint": str(checkpoint), "predictions": str(path),
                    "predictions_sha256": sha(path), "config_sha256": sha(path.with_suffix(path.suffix + ".config.json")),
                    "weight_identity": identity, "questions": 200, "recipes": 200, **metrics}

        candidates = [score_candidate(baseline, initial, 0)]
        write(DEST / "baseline_reuse_receipt.json", {"reused_without_new_inference": True, **candidates[0]})
        output = OLD / "screen/arms/exact_g8"
        verification = DEST / "training_verification.json"
        run("exact_g8_verify", [OLD / "verify_search_run.py", "--run-dir", output,
             "--policy", OLD / "SCREEN_PROTOCOL.json", "--arm", "exact_g8", "--output", verification], [verification])
        require(load(verification)["passed"], "Independent old Exact training verification failed")
        common = ["--model-path", root_path(policy["model_path"]), "--revision", policy["revision"],
                  "--image-root", root_path(policy["image_root"])]
        for step in (32, 64, 96, 128):
            checkpoint = output / ("checkpoint-" + str(step))
            target = DEST / "dev" / ("step" + str(step))
            target.mkdir(parents=True)
            predictions = target / "predictions.jsonl"
            run_id = f"principled_exact_completion_seed44_dev{step}"
            run(f"exact_g8_dev_{step}", [ROOT / "scripts/predict.py", *common,
                "--adapter-path", checkpoint, "--manifest", manifest, "--output", predictions,
                "--run-id", run_id, "--seed", "44"], [predictions, predictions.with_suffix(".jsonl.config.json")])
            coverage = target / "coverage.json"
            run(f"exact_g8_coverage_{step}", [ROOT / "scripts/validate_run.py", "predictions", "--manifest", manifest,
                "--predictions", predictions, "--run-id", run_id, "--checkpoint", checkpoint, "--seed", "44",
                "--expected-questions", "200", "--expected-recipes", "200", "--output", coverage], [coverage])
            require(load(coverage)["passed"], "Development coverage command failed")
            candidate = score_candidate(predictions, checkpoint, step, run_id, 44)
            require(candidate["weight_identity"] == load(verification)["checkpoint_identities"][str(step)],
                    "Evaluation checkpoint differs from independently verified training weights")
            candidates.append(candidate)
            write(target / "raw_rescore.json", candidate)
            update(development_candidates=candidates)
        chosen = max(candidates, key=lambda item: (item["exact"], item["pair"], -item["step"]))
        selection = {"status": "completed_exact_only_development_selection", "selection_rule":
                     "fixed dev200 recipe-macro Exact, then Pair, then earlier; initial SFT step0 eligible",
                     "candidates": candidates, "selected": chosen, "selected_identity": chosen["weight_identity"],
                     "original_policy_sha256": POLICY_SHA, "test_access": False, "training_restart": False,
                     "external_stop_reason": REASON, "all_original_arms_completed": False,
                     "old_policy_and_state_preserved": True, "finished_utc": now()}
        write(DEST / "selection.json", selection)
        immutable()
        update(status="completed_exact_only", phase="four_checkpoints_and_baseline_selected_no_test",
               selection=selection, evaluation_seconds=args.evaluation_seconds - (deadline - time.monotonic()),
               final_artifacts=artifacts([DEST / "launch_receipt.json", DEST / "old_parent_retirement.json",
                   verification, DEST / "baseline_reuse_receipt.json", DEST / "selection.json"] +
                   [DEST / "dev" / f"step{step}" / name for step in (32, 64, 96, 128)
                    for name in ("predictions.jsonl", "predictions.jsonl.config.json", "coverage.json", "raw_rescore.json")]),
               finished_utc=now())
    except BaseException as error:
        update(status="failed_stopped_no_retry", error=f"{type(error).__name__}: {error}", finished_utc=now())
        raise
    finally:
        for handle in reversed(locks):
            handle.close()
        own_lock.close()


if __name__ == "__main__":
    main()
