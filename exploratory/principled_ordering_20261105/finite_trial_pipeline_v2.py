#!/usr/bin/env python3
"""Audited v2 recovery of the finite-policy preflight budget stop.

NEW orchestration only: reuses the frozen finite trainer/scorer/verifiers.
No test/confirmation inference, evaluation or selection, automatic retry,
changed algorithms or hardware. The secondary bridge audits original validation
annotations but computes model scores only on the fixed development200.
The frozen v1 run remains stopped. Only SHA-bound preflight/cache evidence is
reused; every formal optimizer starts fresh from the selected full-data SFT.
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
from finite_trial_pipeline import estimate_whole_batch
from finite_policy_train import CANDIDATES, log_policy, pixel_identities

LOCKS = ("formal_training_pipeline", "evaluation_pipeline", "evaluation_handoff", "seed43_pipeline", "reward_search_20261105")
METHODS = ("raw_sft", "exact", "exact_kl", "rloo_kl")
RL_ORDER = ("exact", "exact_kl", "rloo_kl")
POINTS = (16, 32, 48, 64)
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
ABSOLUTE_DEADLINE_UTC = "2026-11-06T01:31:05Z"
RECOVERY_DIRECTORY = "exploratory/principled_ordering_20261105/finite_trials"
OLD_CPU_SUITES = (("finite", "test_finite_policy*.py", 11), ("calibrator", "test_calibrate_beta.py", 8),
                  ("verifier", "test_verify_finite_run.py", 8), ("coordinator", "test_finite_trial_pipeline.py", 7))
OLD_STAGES = ("cpu_finite", "cpu_calibrator", "cpu_verifier", "cpu_coordinator", "reference256_flow44",
              "beta_calibration", "reference4_smoke", "smoke_raw_sft", "smoke_verify_raw_sft",
              "smoke_exact", "smoke_verify_exact", "smoke_exact_kl", "smoke_verify_exact_kl",
              "smoke_rloo_kl", "smoke_verify_rloo_kl")


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


def remaining_seconds(absolute_deadline, phase_deadline=None, *, utc_now=None, monotonic_now=None):
    """The phase cap never extends the independent, absolute 12-hour deadline."""
    utc_now = datetime.now(timezone.utc) if utc_now is None else utc_now
    monotonic_now = time.monotonic() if monotonic_now is None else monotonic_now
    require(absolute_deadline.tzinfo is not None and utc_now.tzinfo is not None, "Deadline must be timezone-aware")
    absolute = (absolute_deadline - utc_now).total_seconds()
    return absolute if phase_deadline is None else min(absolute, phase_deadline - monotonic_now)


def recovery_decision(state, process):
    require(state.get("status") == "stopped_budget_no_retry" and state.get("active_child_pid") is None,
            "Only an inactive, committed v1 budget stop can be recovered")
    require(type(state.get("pid")) is int and state["pid"] > 1, "Missing v1 coordinator PID")
    require(state.get("flow_results") == {}, "A formal arm/selection already ran in v1")
    commands = state.get("commands", [])
    require(tuple(command.get("stage") for command in commands) == OLD_STAGES
            and all(command.get("returncode") == 0 for command in commands),
            "Recovery requires exactly the successful CPU/reference/calibration/four-smoke preflight and no formal commands")
    require(state.get("whole_batch_timing_gate", {}).get("passed") is False,
            "The v1 stop was not the whole-batch timing gate")
    return "wait_v1_parent_exit" if process is not None and process["state"] not in ("Z", "X") else "ready"


def required_recovery_artifacts():
    names = {"state.json", "whole_batch_timing_gate.json", "train256_flow45.jsonl", "flow_membership.json", "beta_calibration.json"}
    names.update(name + "_tests.json" for name, _, _ in OLD_CPU_SUITES)
    for tag in ("reference256_flow44", "reference4_smoke"):
        names.update(tag + "/" + name for name in ("reference_policy.jsonl", "reference_policy.jsonl.config.json",
                     "reference_policy.jsonl.completion.json", "reference_only_outcome.json", "run_signature.json"))
    for method in METHODS:
        names.update("smoke/" + method + "/" + name for name in
                     ("run_signature.json", "actual_prompt_visits.jsonl", "step_times.jsonl", "train_outcome.json", "independent_verification.json"))
    return {RECOVERY_DIRECTORY + "/" + name for name in names}


def validate_cpu_receipt(report, expected):
    require(report.get("tests_run") == expected and report.get("successful") is True
            and report.get("failures") == report.get("errors") == report.get("skipped") == 0
            and report.get("cuda_initialized") is False, "A recovered CPU suite did not pass completely without CUDA")


def estimate_recovery_batch(smoke, reference_seconds, predecessor_commands, factor=1.25, cpu_allowance=300):
    """Same six-arm gate, conservatively bound RLOO by the slowest RL smoke.

    Retain the original startup/checkpoint, decoder, reference, safety and CPU
    allowances. The proxy avoids assuming that the initial sparse RLOO gradient
    cost remains sparse after KL utilities change. It is not a runtime guarantee.
    """
    result = estimate_whole_batch(smoke, reference_seconds, predecessor_commands, factor, cpu_allowance)
    proxy = max(smoke[method]["step_seconds"] for method in RL_ORDER)
    increment = 64 * (proxy - smoke["rloo_kl"]["step_seconds"])
    result["components_seconds"]["six_full_training_branches"] += increment
    result["estimated_seconds"] += factor * increment
    result["rloo_step_seconds_smoke"] = smoke["rloo_kl"]["step_seconds"]
    result["rloo_step_seconds_bound"] = proxy
    result["resource_only_revision"] = "RLOO uses maximum RL smoke step time; no deduction for cold start or duplicated startup allowances"
    result["assumptions"] += "; sparse initial RLOO cost replaced by slowest RL smoke proxy; measured estimate only, hard deadline still enforced"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=HERE / "FINITE_TRIAL_PROTOCOL_V2.json")
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
    require(policy["status"] == "locked_before_finite_trials_v2", "Draft/unreviewed v2 policy cannot run")
    require(policy["phase_wall_limit_seconds"] == 25200 and policy["absolute_deadline_utc"] == ABSOLUTE_DEADLINE_UTC,
            "Recovery must use the declared seven-hour phase cap and original absolute12h deadline")
    absolute_deadline = datetime.fromisoformat(ABSOLUTE_DEADLINE_UTC.replace("Z", "+00:00"))
    require(policy["methods"] == list(METHODS)
            and policy["rl_family_order"] == list(RL_ORDER) and policy["checkpoint_steps"] == list(POINTS)
            and policy["flow44_seed"] == 44 and policy["flow45_seed"] == 45
            and policy["shuffle_seed"] == 20261116 and policy["limit"] == 256 and policy["max_steps"] == 64,
            "Method/order/budget differs from reviewed fixed design")
    require(policy["recovery_state"] == RECOVERY_DIRECTORY + "/state.json"
            and policy["recovery_policy"] == "exploratory/principled_ordering_20261105/FINITE_TRIAL_PROTOCOL.json"
            and policy["output_directory"] == "exploratory/principled_ordering_20261105/finite_trials_v2",
            "Recovery must read the original v1 preflight and write a separate v2 namespace")
    require(required_recovery_artifacts() <= set(policy["recovery_artifacts"])
            and policy["recovery_policy"] in policy["recovery_artifacts"], "Policy omits required immutable recovery artifacts")
    require([(suite["name"], suite["pattern"], suite["expected_tests"]) for suite in policy["cpu_suites"]] == list(OLD_CPU_SUITES),
            "Recovered CPU suite counts/patterns differ")
    require(policy["v2_cpu_suite"] == {"name": "coordinator_v2", "pattern": "test_finite_trial_pipeline_v2.py", "expected_tests": 8},
            "New v2 CPU suite is not fixed")
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
    state = {"status": "waiting_recovery_preflight", "pid": os.getpid(), "started_utc": now(), "commands": [],
             "policy_sha256": policy_sha, "post_hoc_exploration": True, "test_access": False,
             "confirmation_access": False, "automatic_retry": False, "rented_new_hardware": False,
             "active_child_pid": None, "wall_limit_seconds": 25200, "absolute_deadline_utc": ABSOLUTE_DEADLINE_UTC,
             "recovery_only_preflight": True, "recovery_artifacts_sha256": policy["recovery_artifacts"]}
    active = None
    locks = []
    deadline = None
    phase_started_monotonic = None
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
        for name, digest in policy["recovery_artifacts"].items():
            path = root_path(name)
            require(path == root_path(policy["recovery_policy"]) or path.is_relative_to(root_path(RECOVERY_DIRECTORY)),
                    "Recovery artifact escaped v1 metadata/cache namespace")
            require(not (path.name.endswith(".safetensors") or path.suffix in (".pt", ".pth")),
                    "Recovery reuses metadata/cache only, never smoke trained weights/optimizer/RNG")
            require(sha256_file(path) == digest, "Bound recovery artifact changed: " + name)
        if predecessor_sha is not None:
            require(sha256_file(root_path(policy["predecessor_state"])) == predecessor_sha, "Committed full-SFT state changed")
        if selection_sha is not None:
            require(sha256_file(root_path(policy["predecessor_selection"])) == selection_sha, "Committed full-SFT selection changed")

    def remaining():
        value = remaining_seconds(absolute_deadline, deadline)
        if value <= 0:
            raise BudgetStop("V2 seven-hour phase or independent absolute12h deadline reached; no automatic retry or partial-success claim")
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
            raise BudgetStop("V2 phase/absolute deadline killed only its owned child group") from error
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

    def validate_reused_reference(tag, records, manifest_sha):
        out = root_path(RECOVERY_DIRECTORY) / tag
        cache = out / "reference_policy.jsonl"
        config = load(str(cache) + ".config.json")
        completion = load(str(cache) + ".completion.json")
        outcome = load(out / "reference_only_outcome.json")
        signature = load(out / "run_signature.json")
        config_sha = hash_json(config)
        require(config["initial_identity"] == {"base": base_identity, "adapter": initial_identity}
                and config["manifest_sha256"] == manifest_sha and config["selected_records_sha256"] == hash_json(records)
                and config["input"] == policy["input"] and config["revision"] == policy["revision"]
                and config["versions"] == package_versions() and config["candidate_texts"] == list(CANDIDATES)
                and config["pixel_sha256"] == pixel_identities(records, root_path(policy["image_root"])),
                "Recovered reference weights/input/record order/pixels/versions differ")
        for name, digest in config["scoring_source_sha256"].items():
            require(policy["sources_sha256"].get(name) == digest == sha256_file(root_path(name)),
                    "Recovered reference scoring source differs")
        require(signature["sha256"] == hash_json(signature["payload"])
                and signature["payload"]["initial_identity"] == config["initial_identity"]
                and signature["payload"]["reference_spec_sha256"] == config_sha
                and signature["payload"]["reference_policy_sha256"] == sha256_file(cache),
                "Recovered reference run signature differs")
        require(outcome["status"] == "completed_reference_only" and outcome["rows"] == len(records)
                and outcome["updates"] == 0 and outcome["optimizer_constructed"] is False
                and outcome["initial_weights_unchanged"] is True
                and outcome["reference_policy_sha256"] == sha256_file(cache)
                and outcome["reference_spec_sha256"] == config_sha
                and outcome["run_signature_sha256"] == signature["sha256"], "Recovered cache has no successful no-update receipt")
        require(completion["status"] == "complete_fixed_initial_reference" and completion["rows"] == len(records)
                and completion["cache_sha256"] == sha256_file(cache)
                and completion["reference_spec_sha256"] == config_sha, "Recovered cache completion differs")
        refs = read_jsonl(cache)
        require(len(refs) == len(records), "Recovered reference denominator differs")
        for index, (record, ref) in enumerate(zip(records, refs)):
            require(ref["index"] == index and ref["reference_spec_sha256"] == config_sha
                    and ref["source_split"] == record["source_split"] == "train"
                    and all(ref[name] == record[name] for name in
                            ("qid", "recipe_id", "presentation_id", "display_perm", "target_order", "target_labels")),
                    "Recovered reference source mapping differs")
            require(len(ref["candidates"]) == len(ref["token_logps"]) == len(ref["q"]) == len(ref["logq"]) == 24
                    and tuple(candidate["text"] for candidate in ref["candidates"]) == CANDIDATES,
                    "Recovered reference does not cover the full fixed24 actions")
            for candidate, values in zip(ref["candidates"], ref["token_logps"]):
                require(len(values) == candidate["token_count"] == len(candidate["token_ids"])
                        and candidate["token_ids"][-1] == candidate["eos_token_id"]
                        and all(math.isfinite(value) and value <= 1e-6 for value in values), "Recovered token/EOS scores differ")
            recomputed = log_policy([math.fsum(values) for values in ref["token_logps"]])
            require(all(abs(recomputed[key][i] - ref[key][i]) <= 1e-9 for key in ("q", "logq") for i in range(24)),
                    "Recovered q24 is inconsistent with token log probabilities")
            rewards = [score_order(candidate["text"], record["display_perm"], record["target_order"])["exact"]
                       for candidate in ref["candidates"]]
            require(rewards == ref["exact_rewards"] and sum(rewards) == 1
                    and rewards.index(1) == ref["gold_candidate_index"], "Recovered verifier targets differ")
        return cache

    def recover_preflight(old, records, manifest_sha):
        """Validate SHA-bound evidence without GPU scoring or optimizer use."""
        old_dir = root_path(RECOVERY_DIRECTORY)
        require(old["common_initial_checkpoint"] == str(initial) and old["common_initial_identity"] == initial_identity
                and old["predecessor_completed_state_sha256"] == predecessor_sha
                and old["predecessor_selection_sha256"] == selection_sha, "V1 preflight used a different selected common start")
        cpu_receipts = {}
        for name, _, expected in OLD_CPU_SUITES:
            path = old_dir / (name + "_tests.json")
            validate_cpu_receipt(load(path), expected)
            cpu_receipts[name] = receipt(path)
        write_json(destination / "reused_cpu_receipts.json", cpu_receipts)
        ref256 = validate_reused_reference("reference256_flow44", records, manifest_sha)
        ref4 = validate_reused_reference("reference4_smoke", records[:4], manifest_sha)
        calibration_path = old_dir / "beta_calibration.json"
        calibration = load(calibration_path)
        require(calibration["status"] == "calibrated_positive_beta" and calibration["exit_code"] == 0
                and math.isfinite(calibration["beta"]) and calibration["beta"] > 0
                and calibration["prefix"] == 64 and calibration["delta"] == .02
                and calibration["automatic_beta_zero_fallback"] is False
                and calibration["reference_weight_identity"] == {"base": base_identity, "adapter": initial_identity}
                and calibration["analyzer_source_sha256"] == sha256_file(HERE / "calibrate_beta.py")
                and calibration["input_receipts"]["reference_cache"]["sha256"] == sha256_file(ref256)
                and calibration["input_receipts"]["training_manifest"]["sha256"] == manifest_sha
                and old["beta"] == calibration["beta"]
                and old["beta_calibration_receipt"]["sha256"] == sha256_file(calibration_path),
                "Recovered positive beta is not bound to the same training prefix/reference")
        beta = calibration["beta"]
        commands = {entry["stage"]: entry for entry in old["commands"]}
        smoke = {}
        for method in METHODS:
            out = old_dir / "smoke" / method
            report = load(out / "independent_verification.json")
            signature = load(out / "run_signature.json")
            outcome = load(out / "train_outcome.json")
            steps = read_jsonl(out / "step_times.jsonl")
            visits = read_jsonl(out / "actual_prompt_visits.jsonl")
            require(report["status"] == "complete_verified_finite_policy_run" and report["passed"] is True
                    and report["method"] == method and report["seed"] == 44 and report["limit"] == 4
                    and report["optimizer_steps"] == 1 and report["manifest_sha256"] == manifest_sha
                    and report["beta"] == (beta if method in ("exact_kl", "rloo_kl") else 0)
                    and report["reference"]["sha256"] == sha256_file(ref4)
                    and report["provenance"]["verifier_sha256"] == sha256_file(HERE / "verify_finite_run.py")
                    and report["provenance"]["arguments"]["initial_adapter"] == str(initial),
                    "Recovered smoke independent acceptance differs")
            require(signature["sha256"] == hash_json(signature["payload"]) == report["run_signature_sha256"]
                    and signature["payload"]["initial_identity"] == {"base": base_identity, "adapter": initial_identity}
                    and signature["payload"]["reference_policy_sha256"] == sha256_file(ref4),
                    "Recovered smoke common initial/reference signature differs")
            for name, digest in report["source_sha256"].items():
                require(policy["sources_sha256"].get(name) == digest == sha256_file(root_path(name)), "Recovered smoke source differs")
            require(len(steps) == 1 and steps[0]["global_step"] == 1 and len(visits) == 4
                    and [(row["qid"], row["presentation_id"]) for row in visits]
                        == [(row["qid"], row["presentation_id"]) for row in records[:4]]
                    and outcome["status"] == "completed" and outcome["global_step"] == 1
                    and outcome["actual_prompt_visits"] == 4 and outcome["run_signature_sha256"] == signature["sha256"],
                    "Recovered smoke actual records/update receipt differs")
            smoke[method] = {"step_seconds": steps[0]["elapsed_seconds"],
                             "command_seconds": commands["smoke_" + method]["elapsed_seconds"],
                             "scientific_outcome": outcome["scientific_outcome"]}
            require(smoke[method] == old["smoke"][method], "Recovered smoke timing differs from committed v1 state")
        return ref256, calibration_path, beta, smoke, commands["reference256_flow44"]["elapsed_seconds"]

    try:
        update()
        remaining()
        immutable()
        old_policy_path = root_path(policy["recovery_policy"])
        old_policy = load(old_policy_path)
        require(old_policy["status"] == "locked_before_finite_trials", "Original v1 policy is not the locked policy")
        unchanged = ("methods", "rl_family_order", "checkpoint_steps", "flow44_seed", "flow45_seed", "shuffle_seed",
                     "limit", "max_steps", "accumulation", "learning_rate", "input", "model_path", "image_root", "revision",
                     "train_manifest", "train_manifest_sha256", "dev_manifest", "dev_manifest_sha256", "official_val_source",
                     "calibration", "cpu_suites", "timing_safety_factor", "timing_cpu_allowance_seconds")
        require(all(policy[key] == old_policy[key] for key in unchanged), "V2 changed a scientific setting or original timing allowance")
        for name, digest in old_policy["sources_sha256"].items():
            require(policy["sources_sha256"].get(name) == digest == sha256_file(root_path(name)),
                    "Recovery changed a v1 source/model/manifest input: " + name)
        old = load(root_path(policy["recovery_state"]))
        require(old["policy_sha256"] == sha256_file(old_policy_path), "Recovered v1 state policy binding differs")
        require(not any((root_path(RECOVERY_DIRECTORY) / name).exists() for name in
                        ("flow44", "flow45", "selected_decoders", "rl_family_selection.json", "summary.json")),
                "Original v1 directory already contains formal experiment/selection outputs")
        old_exit_wait = time.monotonic()
        while True:
            remaining()
            immutable()
            old_process = process_info(old["pid"])
            action = recovery_decision(old, old_process)
            if action == "ready":
                break
            require(old_process["cwd"] == str(ROOT)
                    and any(Path(arg).name == "finite_trial_pipeline.py" for arg in old_process["argv"]),
                    "V1 coordinator PID was reused by another live process")
            require(time.monotonic() - old_exit_wait <= 120, "Stopped v1 coordinator did not exit")
            update(status="waiting_v1_parent_exit", v1_parent_pid=old["pid"])
            time.sleep(5)
        update(v1_parent_pid=old["pid"], v1_stopped_state_sha256=sha256_file(root_path(policy["recovery_state"])))
        full_policy_path = root_path(policy["predecessor_policy"])
        full_policy = load(full_policy_path)
        require(full_policy["status"] == "locked_before_full_data_training", "Full-SFT predecessor policy is not locked")
        require(sha256_file(full_policy_path) == policy["predecessor_policy_sha256"], "Full-SFT predecessor policy differs")
        expected_pid, observed_start = None, None
        wait_started = time.monotonic()
        exit_wait_started = None
        while True:
            remaining()
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
        def lock_wait(attempt, blocked):
            remaining()
            update(lock_attempts=attempt, blocked_lock=blocked)
        locks, attempts = acquire_lockset([ROOT / "metadata" / (name + ".lock") for name in LOCKS], 120, 2, lock_wait)
        require_gpu_idle()
        require(shutil.disk_usage(ROOT).free > 6 * 1024**3, "Require6GiB free on data disk for all finite checkpoints")
        phase_started_monotonic = time.monotonic()
        deadline = phase_started_monotonic + min(25200, remaining())
        update(status="phase2_preflight", phase2_started_utc=now(), lock_attempts=attempts,
               initial_effective_budget_seconds=remaining(), budget_rule="min(new25200s phase, original absolute12h deadline)")
        suite = policy["v2_cpu_suite"]
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
        old_flow45 = root_path(RECOVERY_DIRECTORY) / "train256_flow45.jsonl"
        require(sha256_file(old_flow45) == flow45_sha and read_jsonl(old_flow45) == shuffled,
                "V2 flow45 membership/order must exactly match the predeclared v1 flow45")
        write_json(destination / "flow_membership.json", {"source_manifest_sha256": manifest_sha, "source_prefix_sha256": hash_json(records),
                   "flow45_manifest_sha256": flow45_sha, "shuffle_seed": 20261116, "rows": 256,
                   "identical_record_multiset": True, "source_fields_unchanged": True, "created_before_finite_development_selection": True})
        ref256, calibration_path, beta, smoke, reference_seconds = recover_preflight(old, records, manifest_sha)
        update(beta=beta, beta_calibration_receipt=receipt(calibration_path), beta_recalibration_for_flow45=False,
               recovered_smoke=smoke, no_smoke_adapter_or_optimizer_reused=True,
               recovered_reference256=receipt(ref256), recovered_original_cpu_suites=receipt(destination / "reused_cpu_receipts.json"))
        original_estimate = estimate_whole_batch(smoke, reference_seconds, prior["commands"],
                                               policy["timing_safety_factor"], policy["timing_cpu_allowance_seconds"])
        original_gate = load(root_path(RECOVERY_DIRECTORY) / "whole_batch_timing_gate.json")
        require(original_gate == old["whole_batch_timing_gate"] and original_gate["passed"] is False
                and abs(original_gate["estimated_seconds"] - original_estimate["estimated_seconds"]) <= 1e-6
                and all(abs(original_gate["components_seconds"][name] - value) <= 1e-6
                        for name, value in original_estimate["components_seconds"].items()), "V1 timing gate cannot be reconstructed from recovered timings")
        write_json(destination / "original_whole_batch_timing_gate.json", original_gate)
        estimate = estimate_recovery_batch(smoke, reference_seconds, prior["commands"],
                                           policy["timing_safety_factor"], policy["timing_cpu_allowance_seconds"])
        estimate.update(actual_remaining_seconds=remaining(), passed=estimate["estimated_seconds"] <= remaining())
        write_json(destination / "whole_batch_timing_gate.json", estimate)
        update(smoke=smoke, whole_batch_timing_gate=estimate, original_whole_batch_timing_gate_receipt=receipt(destination / "original_whole_batch_timing_gate.json"))
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
                            "phase_wall_seconds": time.monotonic() - phase_started_monotonic, "commands": len(state["commands"]),
                            "v1_reused_preflight_command_seconds": sum(command["elapsed_seconds"] for command in old["commands"]),
                            "disclosure": "New command/phase wall measurements are not the cloud rental invoice; original reused preflight and wait costs are separate; original v1 remains stopped"},
                   "recovery": {"original_state": receipt(root_path(policy["recovery_state"])),
                                "reused_artifacts_sha256": policy["recovery_artifacts"],
                                "smoke_trained_weights_or_optimizer_reused": False,
                                "absolute_deadline_utc": ABSOLUTE_DEADLINE_UTC,
                                "original_gate": receipt(destination / "original_whole_batch_timing_gate.json"),
                                "revised_resource_gate": receipt(destination / "whole_batch_timing_gate.json")},
                   "test_access": False, "confirmation_access": False, "new_hardware": False,
                   "access_disclosure": {"test_inference_evaluation_selection": False,
                       "confirmation_inference_evaluation_selection": False,
                       "original_val_annotation_audit": True,
                       "model_scoring_scope": "fixed200 development manifest only",
                       "confirmation_access_field_means": "inference, evaluation and model selection; original val annotation audit is disclosed separately"},
                   "interpretation": "Development exploration and conditional-on-common-SFT repetition; no official test/SOTA or new-algorithm claim"}
        remaining()
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
