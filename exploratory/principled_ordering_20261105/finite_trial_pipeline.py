#!/usr/bin/env python3
"""Bounded development-only finite-policy experiment coordinator.

NEW orchestration only: reuses the frozen finite trainer/scorer/verifiers.
No test/confirmation inference, evaluation or selection, automatic retry,
changed algorithms or hardware. The secondary bridge audits original validation
annotations but computes model scores only on the fixed development200.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ops"))
from recipeqa.metrics import score_order
from recipeqa.modeling import hash_json, model_weight_identity, package_versions, read_jsonl, sha256_file, write_json
from wait_and_evaluate import process_info, require_gpu_idle
from parameter_diagnostic_pipeline import acquire_lockset, release_locks

LOCKS = ("formal_training_pipeline", "evaluation_pipeline", "evaluation_handoff", "seed43_pipeline", "reward_search_20261105")
METHODS = ("raw_sft", "exact", "exact_kl", "rloo_kl")
RL_ORDER = ("exact", "exact_kl", "rloo_kl")
POINTS = (16, 32, 48, 64)
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class BudgetStop(RuntimeError):
    pass


class BetaPolicyReview(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def receipt(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def root_path(value):
    path = (ROOT / value).resolve()
    require(path.is_relative_to(ROOT), "Path escaped project data disk: " + str(path))
    # Unit-test Python sources must be hashable/readable. Scientific inputs are
    # separately pinned below to train_all7558 and the original val200 only.
    return path


def predecessor_decision(state, process, expected_pid=None, observed_start=None):
    pid = state.get("pid")
    require(type(pid) is int and pid > 1, "Full-SFT predecessor has no valid PID")
    require(expected_pid is None or pid == expected_pid, "Full-SFT predecessor PID changed")
    require(process is None or observed_start is None or process["start_ticks"] == observed_start,
            "Full-SFT predecessor PID was reused")
    status = state.get("status")
    require(status != "failed_stopped_no_retry", "Full-SFT predecessor failed; no finite-policy launch")
    require(status in ("waiting_parameter_diagnostics", "waiting_gpu_locks", "running", "completed_full_data_sft"),
            "Unknown full-SFT predecessor state")
    live = process is not None and process["state"] not in ("Z", "X")
    if status != "completed_full_data_sft":
        require(live, "Full-SFT predecessor vanished before committing success")
        return "wait_completion"
    require(state.get("phase") == "one_pass_and_fixed_dev_selection_complete"
            and state.get("active_child_pid") is None, "Full-SFT terminal phase/child is incomplete")
    commands = state.get("commands", [])
    require(len(commands) == 11 and all(command.get("returncode") == 0 for command in commands),
            "Full-SFT completion lacks all11 successful command receipts")
    return "wait_parent_exit" if live else "ready"


def choose_checkpoint(candidates):
    require(len(candidates) == 5 and [item["step"] for item in candidates] == [0, *POINTS], "Need baseline+four fixed development points")
    return max(candidates, key=lambda item: (item["exact"], item["pair"], -item["step"]))


def choose_rl_family(arms):
    require(set(arms) == set(RL_ORDER), "Need all three RL families before family selection")
    return max(RL_ORDER, key=lambda method: (arms[method]["selected"]["exact"], arms[method]["selected"]["pair"], -RL_ORDER.index(method)))


def checkpoint_interval(limit):
    return 1 if limit == 4 else 16


def validate_greedy_row(row, source, identity_sha):
    """Conform to frozen predict.py, whose output has NO manifest_set field."""
    require(source.get("source_split") == source.get("manifest_set") == "val", "Expected manifest is not fixed development")
    require(not row.get("technical_error") and row.get("source_split") == "val"
            and row.get("model_weights_sha256") == identity_sha and row.get("prompt_kind") == "answer",
            "Invalid greedy development identity/status")
    require(row.get("decoding", {}).get("do_sample") is False
            and row["decoding"]["max_new_tokens"] == 32, "Greedy row decoding differs")
    require(all(row[field] == source[field] for field in
                ("qid", "presentation_id", "recipe_id", "display_perm", "target_order", "target_labels")),
            "Development verifier source mapping differs")


def shuffle_flow(records, seed):
    require(len(records) == 256, "Flow repetition requires exactly the same256 records")
    result = list(records)
    random.Random(seed).shuffle(result)
    require(sorted(map(hash_json, result)) == sorted(map(hash_json, records)), "Shuffled flow changed input membership")
    return result


def estimate_whole_batch(smoke, reference_seconds, predecessor_commands, factor=1.25, cpu_allowance=300):
    """Resource-only gate, pessimistically includes second-flow worst RL family.

    No reward, accuracy, confidence, model prediction or chosen family is read.
    The caller passes already-completed four-record smoke timing receipts.
    """
    require(set(smoke) == set(METHODS) and reference_seconds > 0, "Incomplete timing inputs")
    for item in smoke.values():
        require(0 < item["step_seconds"] <= item["command_seconds"], "Invalid smoke timing receipt")
    dev = [entry["elapsed_seconds"] for entry in predecessor_commands if entry["stage"] in ("dev256", "dev512", "dev768", "dev945")]
    coverage = [entry["elapsed_seconds"] for entry in predecessor_commands if entry["stage"].startswith("coverage")]
    require(len(dev) == len(coverage) == 4 and min(dev + coverage) > 0, "Missing full-SFT measured development timing")
    train = 64 * (sum(item["step_seconds"] for item in smoke.values())
                  + smoke["raw_sft"]["step_seconds"] + max(smoke[method]["step_seconds"] for method in RL_ORDER))
    overhead = 24 * max(item["command_seconds"] - item["step_seconds"] for item in smoke.values())
    greedy = 24 * (max(dev) + max(coverage))
    secondary = 6 * reference_seconds * 200 / 256
    flow45_reference = reference_seconds
    measured_sum = train + overhead + greedy + secondary + flow45_reference
    return {"estimated_seconds": factor * measured_sum + cpu_allowance,
            "components_seconds": {"six_full_training_branches": train, "startup_checkpoint_bound": overhead,
                                   "24_greedy_dev_evaluations_and_gates": greedy, "six_selected200_q24": secondary,
                                   "second_flow_reference256": flow45_reference},
            "safety_factor": factor, "fixed_cpu_allowance_seconds": cpu_allowance,
            "selection_inputs": "four smoke wall/step times, initial reference wall time, predecessor dev/coverage wall times",
            "score_based_selection": False, "assumptions": "64updates per full arm; six arms total; worst measured RL cost for repetition; no decoder-score shortcut assumed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=HERE / "FINITE_TRIAL_PROTOCOL.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    require(ROOT.is_relative_to("/root/autodl-tmp") and ROOT.stat().st_dev != Path("/").stat().st_dev,
            "Coordinator must run on the separately mounted project data disk")
    require(Path(sys.prefix).resolve().is_relative_to(ROOT) and Path(sys.executable).absolute().is_relative_to(ROOT),
            "Python environment escaped data disk")
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TMPDIR", "TORCH_HOME"):
        require(os.environ.get(name) and Path(os.environ[name]).resolve().is_relative_to(ROOT), "Missing data-disk cache/temp: " + name)
    for name in ("XDG_CACHE_HOME", "PIP_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        require(not os.environ.get(name) or Path(os.environ[name]).resolve().is_relative_to(ROOT), "Active cache escaped data disk: " + name)
    require(all(os.environ.get(name) == "4" for name in THREADS), "Reviewed four-thread runtime is required")
    policy_path = args.policy.resolve()
    require(policy_path.is_relative_to(HERE), "Policy escaped exploratory directory")
    policy = load(policy_path)
    policy_sha = sha256_file(policy_path)
    require(policy["status"] == "locked_before_finite_trials", "Draft/unreviewed finite trial policy cannot run")
    require(policy["wall_limit_seconds"] == 21600 and policy["methods"] == list(METHODS)
            and policy["rl_family_order"] == list(RL_ORDER) and policy["checkpoint_steps"] == list(POINTS)
            and policy["flow44_seed"] == 44 and policy["flow45_seed"] == 45
            and policy["shuffle_seed"] == 20261116 and policy["limit"] == 256 and policy["max_steps"] == 64,
            "Method/order/budget differs from reviewed fixed design")
    require(policy["train_manifest"] == "exploratory/principled_ordering_20261105/full_train_data/train_all7558.jsonl"
            and policy["dev_manifest"] == "data/manifests/val.jsonl" and policy["official_val_source"] == "data/raw/val.json"
            and policy["train_manifest_sha256"] == "b9979ee0612679d2e9460dd3bb8a6f68e7dea6794bc6cdcc3e1defdc87d8d949"
            and policy["dev_manifest_sha256"] == "c998796fbc07a24c7a24062d6d0aba07500c75b34568fd7ba67f8c38daae03c9",
            "Scientific inputs must be the fixed training manifest and reused development200; no held-out override")
    require(policy["sources_sha256"]["exploratory/principled_ordering_20261105/finite_policy_train.py"]
            == "34e29cc48744626dec27f99afdd5bdf34660130cf45f17e6d3ff205786a446a5"
            and policy["sources_sha256"]["exploratory/principled_ordering_20261105/score_permutations.py"]
            == "affeda80416145d2b6a134172e812d15d48a86d77c67128bd2e5f84e0c341819",
            "Coordinator cannot silently replace the reviewed trainer or scorer")
    destination = root_path(policy["output_directory"])
    require(destination.is_relative_to(HERE) and not destination.exists(), "Fresh exploratory output directory required; no automatic restart")
    own_lock = (ROOT / "metadata/finite_trials_20261105.lock").open("a+")
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    destination.mkdir()
    (destination / "logs").mkdir()
    state_path = destination / "state.json"
    state = {"status": "waiting_full_sft", "pid": os.getpid(), "started_utc": now(), "commands": [],
             "policy_sha256": policy_sha, "post_hoc_exploration": True, "test_access": False,
             "confirmation_access": False, "automatic_retry": False, "rented_new_hardware": False,
             "active_child_pid": None, "wall_limit_seconds": 21600}
    active = None
    locks = []
    deadline = None
    predecessor_sha = None
    selection_sha = None
    baseline = None
    initial = None
    initial_identity = None
    base_identity = None
    flow_results = {}

    def update(**values):
        state.update(values, updated_utc=now())
        write_json(state_path, state)

    def immutable():
        require(sha256_file(policy_path) == policy_sha, "Finite trial policy changed after launch")
        for name, digest in policy["sources_sha256"].items():
            require(sha256_file(root_path(name)) == digest, "Bound source/input changed: " + name)
        if predecessor_sha is not None:
            require(sha256_file(root_path(policy["predecessor_state"])) == predecessor_sha, "Committed full-SFT state changed")
        if selection_sha is not None:
            require(sha256_file(root_path(policy["predecessor_selection"])) == selection_sha, "Committed full-SFT selection changed")

    def remaining():
        if deadline is None:
            return None
        value = deadline - time.monotonic()
        if value <= 0:
            raise BudgetStop("Six-hour phase2 wall cap reached; no automatic retry or partial-success claim")
        return value

    def kill_owned():
        nonlocal active
        if active is not None and active.poll() is None:
            try:
                os.killpg(active.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                active.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(active.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                active.wait(timeout=10)

    def signal_stop(number, frame):
        raise RuntimeError("Finite coordinator interrupted by signal " + str(number))

    previous_signals = {number: signal.signal(number, signal_stop) for number in (signal.SIGTERM, signal.SIGINT)}

    def run(stage, arguments, *, cpu=False, allowed=(0,)):
        nonlocal active
        immutable()
        require_gpu_idle()
        timeout = remaining()
        command = [sys.executable, *map(str, arguments)]
        entry = {"stage": stage, "arguments": command, "started_utc": now(), "cpu_only": cpu,
                 "timeout_seconds": timeout, "log": str((destination / "logs" / (stage + ".log")).relative_to(ROOT))}
        state["commands"].append(entry)
        started = time.monotonic()
        try:
            env = dict(os.environ)
            if cpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            with root_path(entry["log"]).open("x") as stream:
                active = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                entry["pid"] = active.pid
                update(status="running", phase=stage, active_child_pid=active.pid)
                entry["returncode"] = active.wait(timeout=max(.001, remaining()))
            require(entry["returncode"] in allowed, stage + " failed; preserve all evidence, no retry")
        except subprocess.TimeoutExpired as error:
            kill_owned()
            raise BudgetStop("Six-hour phase2 cap killed only its owned child group") from error
        except BaseException:
            kill_owned()
            raise
        finally:
            entry.update(elapsed_seconds=time.monotonic() - started, finished_utc=now(),
                         returncode=active.returncode if active is not None else None)
            log = root_path(entry["log"])
            if log.exists():
                entry["log_sha256"] = sha256_file(log)
            update(active_child_pid=None)
            active = None
        return entry

    def cpu_tests(name, pattern, expected):
        require(type(expected) is int and expected > 0, "CPU test count is not frozen")
        report = destination / (name + "_tests.json")
        script = ("import json,pathlib,sys,unittest,torch; "
                  "s=unittest.defaultTestLoader.discover(sys.argv[1],pattern=sys.argv[2]); "
                  "r=unittest.TextTestRunner(verbosity=2).run(s); "
                  "p={'tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),"
                  "'successful':r.wasSuccessful(),'cuda_initialized':torch.cuda.is_initialized()}; "
                  "pathlib.Path(sys.argv[3]).write_text(json.dumps(p,indent=2)+'\\n'); "
                  "sys.exit(0 if r.wasSuccessful() and r.testsRun==int(sys.argv[4]) and not r.skipped and not torch.cuda.is_initialized() else 1)")
        run("cpu_" + name, ["-c", script, HERE, pattern, report, str(expected)], cpu=True)
        result = load(report)
        require(result["tests_run"] == expected and result["successful"] and result["skipped"] == 0
                and result["cuda_initialized"] is False, "Required installed CPU tests did not fully pass")

    def trainer_arguments(method, manifest, manifest_sha, out, limit, seed, beta, cache=None, prepare=False):
        arguments = [HERE / "finite_policy_train.py", "--method", method, "--model-path", root_path(policy["model_path"]),
                     "--adapter-path", initial, "--revision", policy["revision"], "--manifest", manifest,
                     "--expected-manifest-sha256", manifest_sha, "--image-root", root_path(policy["image_root"]),
                     "--output-dir", out, "--limit", str(limit), "--max-steps", str(limit // 4),
                     "--seed", str(seed), "--learning-rate", "0.000005", "--save-steps", str(checkpoint_interval(limit)),
                     "--wall-seconds", str(min(10800, max(.001, remaining())))]
        if method in ("exact_kl", "rloo_kl"):
            arguments += ["--beta", str(beta)]
        if cache is not None:
            arguments += ["--reference-cache", cache]
        if prepare:
            arguments += ["--prepare-reference-only"]
        return arguments

    def prepare_reference(tag, manifest, manifest_sha, limit, seed):
        out = destination / tag
        entry = run(tag, trainer_arguments("raw_sft", manifest, manifest_sha, out, limit, seed, 0, prepare=True))
        report = load(out / "reference_only_outcome.json")
        cache = out / "reference_policy.jsonl"
        completion = load(str(cache) + ".completion.json")
        config = load(str(cache) + ".config.json")
        require(report["status"] == "completed_reference_only" and report["rows"] == limit
                and report["updates"] == 0 and report["optimizer_constructed"] is False
                and report["initial_weights_unchanged"] is True, "Reference-only command trained or did not complete")
        require(completion["status"] == "complete_fixed_initial_reference" and completion["rows"] == limit
                and completion["cache_sha256"] == sha256_file(cache) == report["reference_policy_sha256"], "Reference cache receipt differs")
        require(config["initial_identity"]["adapter"] == initial_identity
                and config["manifest_sha256"] == manifest_sha, "Reference used wrong common initial model/manifest")
        entry["artifacts"] = {str(path.relative_to(ROOT)): receipt(path) for path in
                              (out / "reference_only_outcome.json", cache, Path(str(cache) + ".config.json"), Path(str(cache) + ".completion.json"))}
        update()
        return cache, entry

    def verify_run(tag, out, method, manifest, manifest_sha, seed, limit, beta):
        report_path = out / "independent_verification.json"
        entry = run(tag, [HERE / "verify_finite_run.py", "--run-dir", out, "--manifest", manifest,
                         "--expected-manifest-sha256", manifest_sha, "--initial-adapter", initial,
                         "--method", method, "--seed", str(seed), "--limit", str(limit), "--steps", str(limit // 4),
                         "--beta", str(beta if method in ("exact_kl", "rloo_kl") else 0), "--output", report_path], cpu=True)
        report = load(report_path)
        require(report["status"] == "complete_verified_finite_policy_run" and report["passed"] is True,
                "Independent finite training audit failed")
        entry["artifacts"] = {str(report_path.relative_to(ROOT)): receipt(report_path)}
        update()
        return report_path

    def score_greedy(path, checkpoint, expected_seed=None):
        config = load(str(path) + ".config.json")
        identity = {"base": base_identity, "adapter": model_weight_identity(checkpoint, adapter=True)}
        require(config["manifest_sha256"] == policy["dev_manifest_sha256"] and config["weight_identity"] == identity
                and config["input"] == policy["input"] and config["generation"]["do_sample"] is False
                and config["generation"]["max_new_tokens"] == 32, "Development baseline weights/input/decoder differ")
        if expected_seed is not None:
            require(config["cli"]["seed"] == expected_seed, "Development evaluation seed differs")
        rows = read_jsonl(path)
        dev = read_jsonl(root_path(policy["dev_manifest"]))
        require(len(rows) == len(dev) == 200 and len({row["recipe_id"] for row in rows}) == 200, "Incomplete development denominator")
        expected = {(row["qid"], row["presentation_id"]): row for row in dev}
        require(set(expected) == {(row["qid"], row["presentation_id"]) for row in rows}, "Development question set differs")
        for row in rows:
            source = expected[row["qid"], row["presentation_id"]]
            validate_greedy_row(row, source, hash_json(identity))
        scores = [score_order(row["raw_response"], row["display_perm"], row["target_order"]) for row in rows]
        require(all(score == row["score"] for score, row in zip(scores, rows)), "Development raw rescore differs")
        return {"predictions": str(path), "predictions_sha256": sha256_file(path),
                **{key: sum(score[key] for score in scores) / 200 for key in ("exact", "pair", "valid")}}

    def evaluate_arm(flow, method, out, seed):
        candidates = [{"step": 0, "checkpoint": str(initial), **baseline}]
        for step in POINTS:
            checkpoint = out / f"checkpoint-{step}"
            dev_dir = out / f"dev{step}"
            dev_dir.mkdir()
            predictions = dev_dir / "predictions.jsonl"
            run_id = f"finite_{flow}_{method}_dev{step}"
            run(f"{flow}_{method}_dev{step}", [ROOT / "scripts/predict.py", "--model-path", root_path(policy["model_path"]),
                "--revision", policy["revision"], "--adapter-path", checkpoint, "--image-root", root_path(policy["image_root"]),
                "--manifest", root_path(policy["dev_manifest"]), "--output", predictions, "--run-id", run_id, "--seed", str(seed)])
            gate = dev_dir / "coverage.json"
            run(f"{flow}_{method}_coverage{step}", [ROOT / "scripts/validate_run.py", "predictions", "--manifest", root_path(policy["dev_manifest"]),
                "--predictions", predictions, "--run-id", run_id, "--checkpoint", checkpoint, "--seed", str(seed),
                "--expected-questions", "200", "--expected-recipes", "200", "--output", gate], cpu=True)
            require(load(gate)["passed"] is True, "Fixed development coverage failed")
            candidates.append({"step": step, "checkpoint": str(checkpoint), **score_greedy(predictions, checkpoint, seed)})
        selected = choose_checkpoint(candidates)
        result = {"flow": flow, "method": method, "seed": seed, "candidates": candidates, "selected": selected,
                  "selected_identity": model_weight_identity(selected["checkpoint"], adapter=True),
                  "rule": "greedy dev200Exact,thenPair,thenearlier; baseline0eligible", "test_access": False, "confirmation_access": False}
        write_json(out / "selection.json", result)
        return result

    try:
        update()
        immutable()
        full_policy_path = root_path(policy["predecessor_policy"])
        full_policy = load(full_policy_path)
        require(full_policy["status"] == "locked_before_full_data_training", "Full-SFT predecessor policy is not locked")
        require(sha256_file(full_policy_path) == policy["predecessor_policy_sha256"], "Full-SFT predecessor policy differs")
        expected_pid, observed_start = None, None
        wait_started = time.monotonic()
        exit_wait_started = None
        while True:
            immutable()
            prior_path = root_path(policy["predecessor_state"])
            require(time.monotonic() - wait_started <= policy["wait_seconds"], "Timed out waiting for full-SFT completion")
            if not prior_path.exists():
                time.sleep(5)
                continue
            prior = load(prior_path)
            pid = prior.get("pid")
            require(type(pid) is int and pid > 1, "Malformed predecessor PID")
            process = process_info(pid)
            if expected_pid is None:
                expected_pid = pid
            if process is not None and process["state"] not in ("Z", "X"):
                require(process["cwd"] == str(ROOT) and any(Path(arg).name == "full_sft_pipeline.py" for arg in process["argv"]),
                        "Predecessor PID is not this project's full-SFT coordinator")
                if observed_start is None:
                    observed_start = process["start_ticks"]
            action = predecessor_decision(prior, process, expected_pid, observed_start)
            update(status="waiting_parent_exit" if action == "wait_parent_exit" else "waiting_full_sft",
                   predecessor_pid=pid, predecessor_start_ticks=observed_start, predecessor_status=prior["status"])
            if action in ("ready", "wait_parent_exit"):
                if predecessor_sha is None:
                    predecessor_sha = sha256_file(prior_path)
                require(sha256_file(prior_path) == predecessor_sha, "Committed predecessor changed while waiting for exit")
            if action == "ready":
                break
            if action == "wait_parent_exit":
                if exit_wait_started is None:
                    exit_wait_started = time.monotonic()
                require(time.monotonic() - exit_wait_started <= 120, "Completed full-SFT parent did not exit")
            time.sleep(5)
        require(prior["policy_sha256"] == sha256_file(full_policy_path), "Full-SFT state policy binding differs")
        for name, digest in full_policy["sources_sha256"].items():
            require(sha256_file(root_path(name)) == digest, "Full-SFT predecessor source/input changed: " + name)
        selected_file = root_path(policy["predecessor_selection"])
        selection_sha = sha256_file(selected_file)
        full_selection = load(selected_file)
        require(prior["selection"] == full_selection and full_selection["policy_sha256"] == prior["policy_sha256"],
                "Full-SFT state/selection do not agree")
        initial = root_path(full_selection["selected"]["checkpoint"])
        initial_identity = model_weight_identity(initial, adapter=True)
        require(initial_identity == full_selection["selected_identity"], "Selected full-SFT actual weights differ")
        base_identity = model_weight_identity(root_path(policy["model_path"]))
        require(base_identity["bundle_sha256"] == full_policy["base_bundle_sha256"],
                "Base/processor actual bytes differ from full-SFT baseline")
        baseline = score_greedy(root_path(full_selection["selected"]["predictions"]), initial)
        require(baseline["predictions_sha256"] == full_selection["selected"]["predictions_sha256"],
                "Full-SFT selected prediction bytes differ from the committed selection receipt")
        require(all(abs(baseline[key] - full_selection["selected"][key]) <= 1e-12 for key in ("exact", "pair", "valid")),
                "Full-SFT selected development metrics do not independently rescore")
        update(status="waiting_five_locks", common_initial_checkpoint=str(initial), common_initial_identity=initial_identity,
               predecessor_completed_state_sha256=predecessor_sha, predecessor_selection_sha256=selection_sha)
        locks, attempts = acquire_lockset([ROOT / "metadata" / (name + ".lock") for name in LOCKS], 120, 2,
                                         lambda attempt, blocked: update(lock_attempts=attempt, blocked_lock=blocked))
        require_gpu_idle()
        require(shutil.disk_usage(ROOT).free > 6 * 1024**3, "Require6GiB free on data disk for all finite checkpoints")
        deadline = time.monotonic() + 21600
        update(status="phase2_preflight", phase2_started_utc=now(), lock_attempts=attempts)
        for suite in policy["cpu_suites"]:
            cpu_tests(suite["name"], suite["pattern"], suite["expected_tests"])
        manifest = root_path(policy["train_manifest"])
        manifest_sha = sha256_file(manifest)
        require(manifest_sha == policy["train_manifest_sha256"], "Full training manifest changed")
        records = read_jsonl(manifest)[:256]
        require(len(records) == 256 and all(row["source_split"] == "train" and row["technical_status"] == "ok" for row in records),
                "Fixed training prefix is incomplete or contains held-out input")
        flow45_path = destination / "train256_flow45.jsonl"
        shuffled = shuffle_flow(records, policy["shuffle_seed"])
        with flow45_path.open("x") as stream:
            for row in shuffled:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        flow45_sha = sha256_file(flow45_path)
        write_json(destination / "flow_membership.json", {"source_manifest_sha256": manifest_sha, "source_prefix_sha256": hash_json(records),
                   "flow45_manifest_sha256": flow45_sha, "shuffle_seed": 20261116, "rows": 256,
                   "identical_record_multiset": True, "source_fields_unchanged": True, "created_before_finite_development_selection": True})
        ref256, ref_entry = prepare_reference("reference256_flow44", manifest, manifest_sha, 256, 44)
        calibration_path = destination / "beta_calibration.json"
        calibration_entry = run("beta_calibration", [HERE / "calibrate_beta.py", "--reference-cache", ref256,
            "--manifest", manifest, "--prefix", "64", "--delta", "0.02", "--output", calibration_path], cpu=True, allowed=(0, 2))
        calibration = load(calibration_path)
        if calibration.get("status") == "non_binding_or_boundary_needs_policy_review":
            raise BetaPolicyReview("Declared KL radius is boundary/non-binding; stop for explicit policy review, no beta0 fallback")
        require(calibration_entry["returncode"] == 0 and calibration.get("status") == "calibrated_positive_beta"
                and math.isfinite(calibration.get("beta", float("nan"))) and calibration["beta"] > 0, "KL calibration did not establish positive beta")
        beta = calibration["beta"]
        update(beta=beta, beta_calibration_receipt=receipt(calibration_path), beta_recalibration_for_flow45=False)
        ref4, _ = prepare_reference("reference4_smoke", manifest, manifest_sha, 4, 44)
        smoke = {}
        for method in METHODS:
            out = destination / "smoke" / method
            entry = run("smoke_" + method, trainer_arguments(method, manifest, manifest_sha, out, 4, 44, beta, ref4))
            verify_run("smoke_verify_" + method, out, method, manifest, manifest_sha, 44, 4, beta)
            times = read_jsonl(out / "step_times.jsonl")
            require(len(times) == 1 and times[0]["global_step"] == 1, "Smoke did not execute exactly one planned update")
            smoke[method] = {"step_seconds": times[0]["elapsed_seconds"], "command_seconds": entry["elapsed_seconds"],
                             "scientific_outcome": load(out / "train_outcome.json")["scientific_outcome"]}
        estimate = estimate_whole_batch(smoke, ref_entry["elapsed_seconds"], prior["commands"],
                                        policy["timing_safety_factor"], policy["timing_cpu_allowance_seconds"])
        estimate.update(actual_remaining_seconds=remaining(), passed=estimate["estimated_seconds"] <= remaining())
        write_json(destination / "whole_batch_timing_gate.json", estimate)
        update(smoke=smoke, whole_batch_timing_gate=estimate)
        if not estimate["passed"]:
            raise BudgetStop("All fixed six formal arms plus endpoints do not fit measured remaining budget; no score-based arm/subset selection")
        for method in METHODS:
            out = destination / "flow44" / method
            run("flow44_train_" + method, trainer_arguments(method, manifest, manifest_sha, out, 256, 44, beta, ref256))
            verify_run("flow44_verify_" + method, out, method, manifest, manifest_sha, 44, 256, beta)
            flow_results.setdefault("flow44", {})[method] = evaluate_arm("flow44", method, out, 44)
            update(flow_results=flow_results)
        selected_family = choose_rl_family({method: flow_results["flow44"][method] for method in RL_ORDER})
        write_json(destination / "rl_family_selection.json", {"selected_family": selected_family,
                   "rule": "selectedgreedyExact,thenPair,thenfixedfamilyorderexact/exact_kl/rloo_kl", "family_order": list(RL_ORDER),
                   "families": {method: flow_results["flow44"][method]["selected"] for method in RL_ORDER},
                   "new_seed_selection": False, "decoder_metric_selection": False})
        update(selected_rl_family=selected_family)
        ref45, _ = prepare_reference("reference256_flow45", flow45_path, flow45_sha, 256, 45)
        for method in ("raw_sft", selected_family):
            out = destination / "flow45" / method
            run("flow45_train_" + method, trainer_arguments(method, flow45_path, flow45_sha, out, 256, 45, beta, ref45))
            verify_run("flow45_verify_" + method, out, method, flow45_path, flow45_sha, 45, 256, beta)
            flow_results.setdefault("flow45", {})[method] = evaluate_arm("flow45", method, out, 45)
            update(flow_results=flow_results)
        # Supplementary decoder analysis AFTER every primary greedy selection.
        scored_by_identity = {}
        selected_decoders = {}
        for flow, arms in flow_results.items():
            for method, arm in arms.items():
                checkpoint = root_path(arm["selected"]["checkpoint"])
                identity_key = hash_json(model_weight_identity(checkpoint, adapter=True))
                logical = flow + "_" + method
                if identity_key not in scored_by_identity:
                    score_path = destination / "selected_decoders" / logical / "scores.jsonl"
                    score_path.parent.mkdir(parents=True)
                    run("selected24_" + logical, [HERE / "score_permutations.py", "--model-path", root_path(policy["model_path"]),
                        "--revision", policy["revision"], "--adapter-path", checkpoint, "--manifest", root_path(policy["dev_manifest"]),
                        "--expected-manifest-sha256", policy["dev_manifest_sha256"], "--image-root", root_path(policy["image_root"]),
                        "--output", score_path, "--run-id", "finite_selected_" + logical, "--limit", "200", "--microbatch", "1",
                        "--greedy-predictions", root_path(arm["selected"]["predictions"]), "--seed", str(arm["seed"])])
                    completion = load(str(score_path) + ".completion.json")
                    require(completion["status"] == "complete" and completion["rows"] == 200
                            and completion["output_sha256"] == sha256_file(score_path), "Selected24 decoder analysis incomplete")
                    scored_by_identity[identity_key] = str(score_path)
                selected_decoders[logical] = {"scores": scored_by_identity[identity_key], "selected_checkpoint": str(checkpoint),
                    "selected_identity": arm["selected_identity"], "primary_selection_used_only_greedy": True}
        official = destination / "official_choice_analysis"
        run("official_four_choice_bridge", [HERE / "analyze_official_choices.py", "--source", root_path(policy["official_val_source"]),
            "--manifest", root_path(policy["dev_manifest"]), "--scores", *sorted(set(scored_by_identity.values())), "--output-dir", official], cpu=True)
        require(load(official / "summary.json")["status"] == "verified_development_four_choice_bridge", "Official-choice supplementary analysis failed")
        immutable()
        require(model_weight_identity(root_path(policy["model_path"])) == base_identity,
                "Base/processor bytes changed during finite trials")
        require(all(command.get("returncode") == 0 for command in state["commands"]), "Completion contains failed/incomplete commands")
        rows = []
        for flow, arms in flow_results.items():
            for method, arm in arms.items():
                for candidate in arm["candidates"]:
                    rows.append({"flow": flow, "method": method, "seed": arm["seed"], **candidate,
                                 "selected": candidate["step"] == arm["selected"]["step"]})
        import csv
        with (destination / "development_curves.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {"status": "completed_finite_trials_development_only", "flow_results": flow_results,
                   "selected_rl_family_for_flow45": selected_family, "selected_decoders": selected_decoders,
                   "beta": beta, "beta_calibration": receipt(calibration_path), "flow_membership": receipt(destination / "flow_membership.json"),
                   "official_choice_summary": receipt(official / "summary.json"), "curves": receipt(destination / "development_curves.csv"),
                   "source_sha256": policy["sources_sha256"], "policy_sha256": policy_sha,
                   "cost": {"command_wall_seconds_sum": sum(command["elapsed_seconds"] for command in state["commands"]),
                            "phase_wall_seconds": 21600 - remaining(), "commands": len(state["commands"]),
                            "disclosure": "Command/phase wall measurements are not the cloud rental invoice; wait time excluded from phase cap"},
                   "test_access": False, "confirmation_access": False, "new_hardware": False,
                   "access_disclosure": {"test_inference_evaluation_selection": False,
                       "confirmation_inference_evaluation_selection": False,
                       "original_val_annotation_audit": True,
                       "model_scoring_scope": "fixed200 development manifest only",
                       "confirmation_access_field_means": "inference, evaluation and model selection; original val annotation audit is disclosed separately"},
                   "interpretation": "Development exploration and conditional-on-common-SFT repetition; no official test/SOTA or new-algorithm claim"}
        write_json(destination / "summary.json", summary)
        update(status="completed_finite_trials_development_only", phase="six_arms_verified_primary_and_secondary_dev_complete",
               active_child_pid=None, finished_utc=now(), summary_receipt=receipt(destination / "summary.json"))
    except BetaPolicyReview as error:
        kill_owned()
        update(status="stopped_beta_policy_review", active_child_pid=None, error=str(error), finished_utc=now(), flow_results=flow_results)
        return 2
    except BudgetStop as error:
        kill_owned()
        update(status="stopped_budget_no_retry", active_child_pid=None, error=str(error), finished_utc=now(), flow_results=flow_results)
        return 2
    except BaseException as error:
        kill_owned()
        update(status="failed_stopped_no_retry", active_child_pid=None, error=f"{type(error).__name__}: {error}", finished_utc=now(), flow_results=flow_results)
        raise
    finally:
        release_locks(locks)
        own_lock.close()
        for number, handler in previous_signals.items():
            signal.signal(number, handler)


if __name__ == "__main__":
    raise SystemExit(main())
