#!/usr/bin/env python3
"""Locked development-only screen. Never evaluates or selects on test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ops"))
sys.path.insert(0, str(ROOT / "scripts"))
from recipeqa.metrics import score_order
from recipeqa.modeling import ensure_data_disk, model_weight_identity, read_jsonl, sha256_file, write_json
from validate_run import validate_prediction_coverage
from wait_and_evaluate import require_gpu_idle

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def path_in_root(value):
    path = (ROOT / value).resolve()
    ensure_data_disk(path)
    require(path.is_relative_to(ROOT), f"Path escaped project root: {path}")
    return path


def check_policy(policy_path, policy):
    require(ROOT == Path(policy["root"]).resolve(), "Policy refers to another project root")
    require(ROOT.is_relative_to("/root/autodl-tmp") and ROOT.stat().st_dev != Path("/").stat().st_dev,
            "Project must be on the separately mounted data disk")
    ensure_data_disk(policy_path)
    require(policy["status"] == "locked" and policy["seed"] == 44, "Wrong policy status or screening seed")
    require(0 < policy["wall_limit_seconds"] <= 18000, "Screen wall budget must not exceed five hours")
    required_sources = {str((HERE / name).relative_to(ROOT)) for name in
                        ("screen_pipeline.py", "verify_search_run.py", "train_reward_search.py", "rewards.py")}
    require(required_sources <= set(policy["source_sha256"]), "Policy must bind all new scientific sources")
    for name, digest in policy["source_sha256"].items():
        require(sha256_file(path_in_root(name)) == digest, f"Exploratory source changed: {name}")
    frozen = load(ROOT / "configs/protocol_lock.json")
    require(frozen["status"] == "frozen", "Original protocol is not frozen")
    # Source verification intentionally does not open old test manifests/results.
    checked_sources = {}
    for name, digest in frozen["file_sha256"].items():
        if name.startswith(("src/", "scripts/", "ops/", "configs/")) or name == "RESEARCH_PLAN.md":
            require(sha256_file(ROOT / name) == digest, f"Original frozen source changed: {name}")
            checked_sources[name] = digest
    train = path_in_root(policy["train_manifest"])
    dev = path_in_root(policy["dev_manifest"])
    require("test" not in train.name.lower() and "test" not in dev.name.lower(), "Test manifests are forbidden")
    require(sha256_file(train) == policy["train_manifest_sha256"], "Training manifest changed")
    require(sha256_file(dev) == policy["dev_manifest_sha256"], "Development manifest changed")
    visits = read_jsonl(train)
    dev_rows = read_jsonl(dev)
    require(len(visits) == policy.get("train_rows", 256) == 256, "Screen must use 256 visits")
    require(len(dev_rows) == policy.get("dev_questions", 200) == 200
            and len({row["recipe_id"] for row in dev_rows}) == policy.get("dev_recipes", 200) == 200,
            "Development set must contain exactly 200 independent recipes")
    require(not ({row["recipe_id"] for row in visits} & {row["recipe_id"] for row in dev_rows}),
            "Training/development recipe leakage")
    if policy.get("original_train_manifest"):
        original = path_in_root(policy["original_train_manifest"])
        require(sha256_file(original) == policy["original_train_manifest_sha256"], "Original stream changed")
        require(visits == read_jsonl(original)[:256], "Screen stream differs from original first 256 visits")
    initial = path_in_root(policy["initial_adapter"])
    require(sha256_file(initial / "adapter_model.safetensors") == policy["initial_adapter_weight_sha256"],
            "Strong SFT start changed")
    baseline = path_in_root(policy["baseline"]["predictions"])
    require("test" not in baseline.parts and "test" not in baseline.name.lower(), "Test baseline forbidden")
    require(sha256_file(baseline) == policy["baseline"]["predictions_sha256"], "Common baseline raw predictions changed")
    baseline_config = load(baseline.with_suffix(baseline.suffix + ".config.json"))
    require(baseline_config["manifest_sha256"] == policy["dev_manifest_sha256"], "Baseline used another development set")
    require(baseline_config["weight_identity"]["adapter"] == model_weight_identity(initial, adapter=True),
            "Baseline does not use common initial adapter")
    baseline_rows = read_jsonl(baseline)
    coverage = validate_prediction_coverage(
        dev_rows, baseline_rows, expected_questions=200, expected_recipes=200,
        expected_checkpoint=str(initial),
    )
    require(coverage["passed"], "Common initial baseline coverage failed")
    require(all(not row["decoding"]["do_sample"] and row["decoding"]["max_new_tokens"] == 32
                for row in baseline_rows), "Baseline must use genuine greedy short-answer inference")
    baseline_score = {key: sum(score_order(row["raw_response"], row["display_perm"], row["target_order"])[key]
                               for row in baseline_rows) / 200 for key in ("exact", "pair", "valid")}
    arm_names = [arm["name"] for arm in policy["arms"]]
    require(arm_names and len(set(arm_names)) == len(arm_names)
            and all(name and name.replace("_", "").isalnum() for name in arm_names), "Invalid or duplicate arm names")
    require(arm_names == policy.get("arm_order", arm_names), "Arm order disagrees with locked protocol")
    for arm in policy["arms"]:
        sft = arm["method"] == "sft"
        require(arm["method"] in ("sft", "grpo"), "Invalid method")
        require(arm["max_steps"] == (32 if sft else 128)
                and arm["save_steps"] == (8 if sft else 32), "Wrong screen/checkpoint budget")
        require(arm["learning_rate"] == (5e-5 if sft else 5e-6), "Unplanned learning rate")
        count = arm["max_steps"] * (arm["gradient_accumulation_steps"] if sft
                                    else arm["generation_batch_size"] // arm["G"])
        require(count == 256, "Arm does not have equal 256 prompt exposure")
        if not sft:
            require(arm["G"] in (4, 8) and arm["generation_batch_size"] == arm["gradient_accumulation_steps"],
                    "Invalid G or accumulation boundary")
            require(arm.get("scale_rewards", "group") in ("group", "none"), "Unsupported normalization")
    require_gpu_idle()
    return {"baseline": {"step": 0, "checkpoint": str(initial), "predictions": str(baseline),
                          "predictions_sha256": sha256_file(baseline), **baseline_score},
            "original_source_sha256": checked_sources,
            "protocol_sha256": sha256_file(ROOT / "configs/protocol_lock.json"),
            "initial_adapter_identity": model_weight_identity(initial, adapter=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=HERE / "SCREEN_PROTOCOL.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    os.chdir(ROOT)
    original_env = {name: os.environ.get(name) for name in THREAD_VARS}
    os.environ.update({name: "4" for name in THREAD_VARS})
    policy_path = args.policy.resolve()
    policy = load(policy_path)
    policy_sha = sha256_file(policy_path)
    checks = check_policy(policy_path, policy)
    destination = HERE / "screen"
    if args.check_only:
        print(json.dumps({"status": "ready_read_only", "policy_sha256": policy_sha,
                          "baseline": checks["baseline"], "output_exists": destination.exists(),
                          "test_access": False}, indent=2))
        return
    handles = []
    for name in ("formal_training_pipeline", "evaluation_pipeline", "evaluation_handoff",
                 "seed43_pipeline", "reward_search_20261105"):
        handle = (ROOT / "metadata" / f"{name}.lock").open("a+")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handles.append(handle)
    require(not destination.exists(), "Screen directory exists; inspect existing run instead of restart/resume")
    require_gpu_idle()
    destination.mkdir()
    state_path = destination / "state.json"
    state = {"status": "running", "phase": "initialized", "pid": os.getpid(),
             "started_utc": now(), "policy_sha256": policy_sha, "commands": [], "selected": {},
             "runtime_threads": {name: "4" for name in THREAD_VARS}, "original_env": original_env,
             "test_access": False, "automatic_restart": False, "automatic_promotion": False,
             "checks": checks}

    def update(**values):
        state.update(values, updated_utc=now(), elapsed_seconds=time.monotonic() - started)
        write_json(state_path, state)

    def immutable_check():
        require(sha256_file(policy_path) == policy_sha, "Screen policy changed after launch")
        for name, digest in policy["source_sha256"].items():
            require(sha256_file(path_in_root(name)) == digest, f"Source changed after launch: {name}")
        require(sha256_file(ROOT / "configs/protocol_lock.json") == checks["protocol_sha256"],
                "Original protocol changed after launch")
        for name, digest in checks["original_source_sha256"].items():
            require(sha256_file(ROOT / name) == digest, f"Frozen source changed after launch: {name}")

    def receipt(paths):
        return {str(Path(path).relative_to(ROOT)): {"sha256": sha256_file(path), "bytes": Path(path).stat().st_size}
                for path in paths}

    def run(stage, arguments, required):
        immutable_check()
        remaining = policy["wall_limit_seconds"] - (time.monotonic() - started)
        require(remaining > 1, "Five-hour screen wall budget exhausted")
        require_gpu_idle()
        log = destination / "logs" / f"{stage}.log"
        log.parent.mkdir(exist_ok=True)
        entry = {"stage": stage, "arguments": [sys.executable, *map(str, arguments)],
                 "log": str(log), "started_utc": now(), "timeout_seconds": remaining}
        state["commands"].append(entry)
        command_started = time.monotonic()
        child = None
        try:
            with log.open("x") as stream:
                child = subprocess.Popen(entry["arguments"], stdout=stream, stderr=subprocess.STDOUT)
                entry["pid"] = child.pid
                update(phase=stage, active_child_pid=child.pid)
                wait_budget = policy["wall_limit_seconds"] - (time.monotonic() - started)
                require(wait_budget > 0, "Screen wall budget expired during launch")
                entry["returncode"] = child.wait(timeout=wait_budget)
            require(entry["returncode"] == 0, f"{stage} failed: see {log}")
            entry["required_artifacts"] = receipt(required)
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if child is not None:
                entry["returncode"] = child.returncode
            entry["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            entry.update(elapsed_seconds=time.monotonic() - command_started, finished_utc=now())
            if log.exists():
                entry["log_sha256"] = sha256_file(log)
            update(active_child_pid=None)

    common = ["--model-path", str(path_in_root(policy.get("model_path", "models/Qwen3-VL-2B-Instruct"))),
              "--revision", policy.get("revision", "89644892e4d85e24eaac8bacfd4f463576704203"),
              "--image-root", str(path_in_root(policy.get("image_root", "data/images")))]
    try:
        update()
        for arm in policy["arms"]:
            name = arm["name"]
            output = destination / "arms" / name
            train_command = [str(HERE / "train_reward_search.py"), *common,
                "--method", arm["method"], "--manifest", str(path_in_root(policy["train_manifest"])),
                "--output-dir", str(output), "--initial-adapter", str(path_in_root(policy["initial_adapter"])),
                "--max-steps", str(arm["max_steps"]), "--save-steps", str(arm["save_steps"]),
                "--save-total-limit", "4", "--seed", str(policy["seed"]),
                "--learning-rate", str(arm["learning_rate"]),
                "--gradient-accumulation-steps", str(arm["gradient_accumulation_steps"])]
            if arm["method"] == "grpo":
                train_command += ["--reward", arm["reward"], "--num-generations", str(arm["G"]),
                                  "--generation-batch-size", str(arm["generation_batch_size"]),
                                  "--temperature", str(arm["temperature"]),
                                  "--scale-rewards", arm.get("scale_rewards", "group")]
            run(f"{name}_train", train_command,
                [output / item for item in ("resolved_config.json", "train_outcome.json", "run_signature.json")])
            verification = output / "exploratory_verification.json"
            run(f"{name}_verify", [str(HERE / "verify_search_run.py"), "--run-dir", str(output),
                 "--policy", str(policy_path), "--arm", name, "--output", str(verification)], [verification])
            verified = load(verification)
            require(verified["passed"], "Training verification failed")
            state.setdefault("training_verification", {})[name] = {
                "path": str(verification), "sha256": sha256_file(verification),
                "effective_adapter_update": verified["effective_adapter_update"],
                "scientific_outcome": verified["scientific_outcome"],
                "required_artifacts": verified["required_artifacts"]}
            candidates = [dict(checks["baseline"])]
            for step in range(arm["save_steps"], arm["max_steps"] + 1, arm["save_steps"]):
                checkpoint = output / f"checkpoint-{step}"
                dev_output = destination / "dev" / name / f"step{step}" / "predictions.jsonl"
                dev_output.parent.mkdir(parents=True)
                run_id = f"reward_search_{name}_seed{policy['seed']}_dev{step}"
                run(f"{name}_dev_{step}", ["scripts/predict.py", *common,
                     "--adapter-path", str(checkpoint), "--manifest", str(path_in_root(policy["dev_manifest"])),
                     "--output", str(dev_output), "--run-id", run_id, "--seed", str(policy["seed"])],
                    [dev_output, dev_output.with_suffix(".jsonl.config.json")])
                coverage = dev_output.parent / "coverage.json"
                run(f"{name}_coverage_{step}", ["scripts/validate_run.py", "predictions",
                     "--manifest", str(path_in_root(policy["dev_manifest"])), "--predictions", str(dev_output),
                     "--run-id", run_id, "--checkpoint", str(checkpoint), "--seed", str(policy["seed"]),
                     "--expected-questions", "200", "--expected-recipes", "200", "--output", str(coverage)], [coverage])
                require(load(coverage)["passed"], "Development coverage validation failed")
                rows = read_jsonl(dev_output)
                require(len(rows) == 200 and len({row["recipe_id"] for row in rows}) == 200,
                        "Development macro denominator changed")
                scores = [score_order(row["raw_response"], row["display_perm"], row["target_order"]) for row in rows]
                require(all(row["score"] == score for row, score in zip(rows, scores)), "Development stored scores disagree with raw rescore")
                candidates.append({"step": step, "checkpoint": str(checkpoint), "predictions": str(dev_output),
                    "predictions_sha256": sha256_file(dev_output),
                    **{metric: sum(score[metric] for score in scores) / 200 for metric in ("exact", "pair", "valid")}})
                state.setdefault("development_candidates", {})[name] = candidates
                update()
            chosen = max(candidates, key=lambda row: (row["exact"], row["pair"], -row["step"]))
            selection = {"selection_rule": "fixed dev200 recipe-macro Exact, then Pair, then earlier; common initial SFT step0 eligible",
                         "candidates": candidates, "selected": chosen,
                         "selected_identity": model_weight_identity(chosen["checkpoint"], adapter=True),
                         "test_access": False, "policy_sha256": policy_sha}
            write_json(output / "best_checkpoint.json", selection)
            state["selected"][name] = selection
            update()
        immutable_check()
        update(status="completed_development_screen", phase="all_arms_selected_no_test",
               finished_utc=now(), next_action="Human/root reviews full development results; no automatic promotion or test evaluation")
    except BaseException as error:
        update(status="failed", error=f"{type(error).__name__}: {error}", finished_utc=now())
        raise


if __name__ == "__main__":
    main()
