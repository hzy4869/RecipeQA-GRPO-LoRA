#!/usr/bin/env python3
"""Optional S2: fixed-data seed43 Exact/Pair repetition after the seed42 core.

Only new run/result paths are written. No training or pipeline restart is
automatic; a failure preserves its command, PID and log for operator inspection.
--check-only is CPU preflight; the GPU-idle gate is applied before execution.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recipeqa.metrics import score_order
from recipeqa.modeling import hash_json, model_weight_identity, package_versions, read_jsonl, sha256_file, write_json
from wait_and_evaluate import process_info, require_gpu_idle

METHODS = ("exact", "pair")
STEPS = (256, 512, 768, 1024)
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def read(path):
    return json.loads(path.read_text())


def selection(candidates):
    if [row["step"] for row in candidates] != list(STEPS):
        raise RuntimeError("All four predeclared development checkpoints are required")
    return max(candidates, key=lambda row: (row["exact"], row["pair"], -row["step"]))


def verify_stream(records, visits):
    fields = ("visit_id", "qid", "recipe_id", "presentation_id", "display_perm")
    if len(records) != 2048 or len(visits) != 2048:
        raise RuntimeError("S2 must consume exactly the original 2048 stored visits")
    if any(any(record[key] != visit[key] for key in fields) for record, visit in zip(records, visits)):
        raise RuntimeError("Seed43 changed the fixed seed42 data order or presentation stream")
    if [row["global_step_before_update"] for row in visits] != [index // 2 for index in range(2048)]:
        raise RuntimeError("Actual GRPO grouping must remain two independent visits per update")


def train_command(root, reference, method, policy):
    cli = reference["cli"]
    command = ["scripts/train_grpo.py", "--model-path", cli["model_path"], "--revision", cli["revision"],
               "--image-root", cli["image_root"], "--manifest", str(root / policy["training_manifest"]),
               "--output-dir", str(root / f"runs/{method}_seed43"), "--reward", method,
               "--initial-adapter", cli["initial_adapter"], "--seed", "43"]
    for key in ("max_steps", "save_steps", "learning_rate", "gradient_accumulation_steps", "generation_batch_size", "num_generations"):
        command += ["--" + key.replace("_", "-"), str(policy["training"][key])]
    for key in ("min_pixels", "max_pixels", "recipe_token_budget", "max_input_tokens"):
        command += ["--" + key.replace("_", "-"), str(cli[key])]
    return command


def preflight(root, policy_path):
    policy = read(policy_path)
    if policy.get("status") != "fixed_before_seed43_execution" or policy.get("model_training_seed") != 43 or policy.get("fixed_data_order_seed") != 42 or policy.get("methods") != list(METHODS):
        raise RuntimeError("Only the fixed two-method, model-seed43/data-stream42 S2 policy is supported")
    disclosure = policy["decision_disclosure"]
    if disclosure.get("seed42_test_results_known_when_execution_chosen") is not True or disclosure.get("both_reward_methods_required_and_all_results_disclosed") is not True or disclosure.get("hyperparameter_tuning") is not False or policy["evaluation"].get("SFT_retrained") is not False or policy["evaluation"].get("E4_added") is not False:
        raise RuntimeError("Optional-after-test provenance and fixed reporting scope must remain explicit")
    protocol_path = root / "configs/protocol_lock.json"
    protocol = read(protocol_path)
    if protocol.get("status") != "frozen" or sha256_file(protocol_path) != policy["parent_protocol_sha256"]:
        raise RuntimeError("S2 must inherit the original frozen scientific protocol")
    if package_versions() != protocol["versions"]:
        raise RuntimeError("Package versions changed since the original frozen runs")
    sources = {**protocol["file_sha256"], **policy["unchanged_additional_source_sha256"]}
    sources[str(policy_path.relative_to(root))] = sha256_file(policy_path)
    sources["ops/seed43_pipeline.py"] = sha256_file(root / "ops/seed43_pipeline.py")
    for name, digest in sources.items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"Original source/data or new execution policy changed: {name}")
    formal = read(root / "metadata/formal_training_pipeline.json")
    core = read(root / "metadata/evaluation_pipeline.json")
    if formal.get("status") != "completed" or formal.get("phase") != "three_branches_selected" or core.get("status") != "automated_complete_human_pending" or core.get("phase") != "E0_E4_automatic_complete" or core.get("active_child_pid") or any(row.get("returncode") != 0 for row in core["commands"]):
        raise RuntimeError("The original formal selection and complete E0-E4 core must finish first")
    for state in (formal, core):
        info = process_info(state.get("pid")) if state.get("pid") else None
        if info and info["state"] not in ("Z", "X"):
            raise RuntimeError("An original pipeline parent is still alive; do not overlap it")
    threads = read(root / "metadata/runtime_threads.json")
    if any(threads.get(key) != 4 or os.environ.get(key) != "4" for key in THREADS):
        raise RuntimeError("After source env.sh, export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4")
    for key, adapter in (("base", False), ("W", True)):
        if model_weight_identity(protocol[key]["directory"], adapter=adapter) != protocol[key]:
            raise RuntimeError(f"Original {key} actual model contents changed")
    records = read_jsonl(root / policy["training_manifest"])
    if sha256_file(root / policy["training_manifest"]) != policy["training_manifest_sha256"] or len(records) != 2048 or len({row["qid"] for row in records}) != 1024 or len({row["visit_id"] for row in records}) != 2048 or any(row.get("training_seed") != 42 for row in records):
        raise RuntimeError("The original 1024-question seed42 two-pass input stream must remain unchanged")
    references = {method: read(root / f"runs/{method}_seed42/resolved_config.json") for method in METHODS}
    signatures = {method: read(root / f"runs/{method}_seed42/run_signature.json") for method in METHODS}
    for method in METHODS:
        cli = references[method]["cli"]
        if cli["seed"] != 42 or cli["initial_adapter"] != protocol["W"]["directory"] or Path(cli["model_path"]).resolve() != Path(protocol["base"]["directory"]).resolve() or hash_json(signatures[method]["payload"]) != signatures[method]["sha256"]:
            raise RuntimeError("Original seed42 training identity receipts do not match the common W/Base")
        if any(cli[key] != policy["training"][key] for key in ("max_steps", "save_steps", "learning_rate", "gradient_accumulation_steps", "generation_batch_size", "num_generations")):
            raise RuntimeError("S2 training hyperparameters differ from the original branch")
    monitor = read_jsonl(root / "data/manifests/monitor.jsonl")
    expected_control = [dict(row, manifest_set="greedy_determinism10", control_excluded_from_main=True) for row in monitor[:10]]
    if read_jsonl(root / policy["evaluation"]["determinism"]["manifest"]) != expected_control:
        raise RuntimeError("Reuse exactly the seed42 monitor-first-ten greedy control")
    for name, nrows, nrecipes in (("val", 200, 200), ("test", 905, 597), ("orbit", 1440, 60)):
        rows = read_jsonl(root / f"data/manifests/{name}.jsonl")
        if len(rows) != nrows or len({row["recipe_id"] for row in rows}) != nrecipes:
            raise RuntimeError(f"Frozen {name} cohort size changed")
    snapshot = {"policy_sha256": sha256_file(policy_path), "protocol_sha256": sha256_file(protocol_path),
                "source_sha256": sources, "seed42_selected_unchanged": formal["selected"],
                "Base": protocol["base"], "W": protocol["W"], "seed43_model_seed": 43,
                "fixed_training_data_seed": 42, "human_review_added": False}
    return policy, protocol, references, signatures, records, snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/rl_recipeqa"))
    parser.add_argument("--policy", type=Path, help="Defaults to root/configs/seed43_execution_policy.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve(); policy_path = (args.policy or root / "configs/seed43_execution_policy.json").resolve()
    if not root.is_relative_to("/root/autodl-tmp") or root.stat().st_dev == Path("/").stat().st_dev or not policy_path.is_relative_to(root):
        raise ValueError("Project, policy and every output must remain on the mounted data disk")
    os.chdir(root)
    try:
        policy, protocol, references, signatures, records, snapshot = preflight(root, policy_path)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(json.dumps({"status": "not_ready", "reason": str(error)}), flush=True); return 2
    if args.check_only:
        print(json.dumps({"status": "CPU_preflight_passed_GPU_idle_gate_pending", "seed": 43,
                          "fixed_data_order_seed": 42, "training_commands": [train_command(root, references[m], m, policy) for m in METHODS]}), flush=True); return 0
    locks = []
    for name in ("seed43_pipeline", "formal_training_pipeline", "evaluation_pipeline", "evaluation_handoff"):
        handle = (root / f"metadata/{name}.lock").open("a+")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(handle)
    state_path = root / "metadata/seed43_pipeline.json"
    if state_path.exists() or any(path.exists() and any(path.iterdir()) for m in METHODS for path in (root / f"runs/{m}_seed43", root / f"runs/evaluation/{m}_seed43")):
        raise RuntimeError("S2 record/output already exists. Inspect it; never restart this pipeline over old training")
    require_gpu_idle()
    state = {"status": "running", "pid": os.getpid(), "commands": [], "selected": {}, "active_child_pid": None,
             "started_at": datetime.now(timezone.utc).isoformat(), "execution_snapshot_sha256": hash_json(snapshot),
             "decision_disclosure": policy["decision_disclosure"], "human_review_added": False}
    write_json(root / "metadata/seed43_execution_snapshot.json", snapshot)
    (root / "metadata/seed43_pipeline.pid").write_text(str(os.getpid()) + "\n")

    def update(**values):
        state.update(values, updated_at=datetime.now(timezone.utc).isoformat()); write_json(state_path, state)

    def run(stage, command, artifacts=(), gpu=False):
        if read(root / "metadata/formal_training_pipeline.json")["selected"] != snapshot["seed42_selected_unchanged"]:
            raise RuntimeError("Seed42 checkpoint selections changed during optional S2")
        for name, digest in snapshot["source_sha256"].items():
            if sha256_file(root / name) != digest: raise RuntimeError(f"Locked execution source/data changed: {name}")
        if gpu: require_gpu_idle()
        log = root / f"metadata/s2_{stage}.log"; entry = {"stage": stage, "arguments": command, "log": str(log)}
        state["commands"].append(entry); update(phase=stage)
        started = time.monotonic()
        with log.open("a") as stream:
            child = subprocess.Popen([sys.executable, *command], stdout=stream, stderr=subprocess.STDOUT)
            entry["pid"] = child.pid; update(active_child_pid=child.pid)
            (root / f"metadata/s2_{stage}.pid").write_text(str(child.pid) + "\n"); code = child.wait()
        entry.update(returncode=code, elapsed_seconds=time.monotonic()-started); update(active_child_pid=None)
        if code: raise RuntimeError(f"{stage} failed ({code}); inspect {log}; no automatic retry or training restart")
        entry["artifact_sha256"] = {str(path): sha256_file(path) for path in artifacts}; update()

    common = ["--model-path", references["exact"]["cli"]["model_path"], "--revision", protocol["candidate"]["revision"],
              "--image-root", str(root / "data/images"), "--seed", "43"]
    for key in ("min_pixels", "max_pixels", "recipe_token_budget", "max_input_tokens"):
        common += ["--"+key.replace("_", "-"), str(protocol["candidate"]["input"][key])]
    def predict(method, tag, manifest, checkpoint, questions, recipes, orbit=False, run_id=None):
        name=f"{method}_seed43"; run_id=run_id or f"eval_{name}"; output=root/f"runs/evaluation/{name}/{tag}.jsonl"
        command=["scripts/predict.py",*common,"--adapter-path",checkpoint,"--manifest",str(manifest),"--output",str(output),"--run-id",run_id,"--max-new-tokens","32"]
        run(f"{name}_{tag}",command,(output,output.with_suffix(".jsonl.config.json")),True)
        config=read(output.with_suffix(".jsonl.config.json"))
        if config["weight_identity"] != {"base":protocol["base"],"adapter":read(root/f"runs/{name}/best_checkpoint.json")["selected_identity"]}:
            raise RuntimeError("Endpoint prediction loaded a different selected model")
        coverage=output.with_suffix(".coverage.json")
        command=["scripts/validate_run.py","predictions","--manifest",str(manifest),"--predictions",str(output),"--run-id",run_id,"--checkpoint",checkpoint,"--seed","43","--prompt-kind","answer","--expected-questions",str(questions),"--expected-recipes",str(recipes),"--output",str(coverage)]
        if orbit: command += ["--require-orbit"]
        run(f"{name}_{tag}_coverage",command,(coverage,))
        if not read(coverage)["passed"]: raise RuntimeError("S2 endpoint coverage gate failed")
        return output

    try:
        for method in METHODS:  # Both complete development selections precede any seed43 test prediction.
            name=f"{method}_seed43"; directory=root/f"runs/{name}"
            run(name,train_command(root,references[method],method,policy),tuple(directory/n for n in ("train_outcome.json","run_signature.json","resolved_config.json")),True)
            gate=directory/"artifact_validation.json"
            run(name+"_validation",["scripts/validate_run.py","training","--run-dir",str(directory),"--min-steps","1024","--output",str(gate)],(gate,))
            actual=read(directory/"run_signature.json"); expected=deepcopy(signatures[method]["payload"]); expected["scientific_cli"]["seed"]=43
            config=read(directory/"resolved_config.json"); groups=read_jsonl(directory/"reward_groups.jsonl")
            if actual["payload"]!=expected or hash_json(actual["payload"])!=actual["sha256"] or not read(gate)["passed"] or config["initial_adapter_fingerprint"]!=references[method]["initial_adapter_fingerprint"] or config["initial_visual_fingerprint"]!=references[method]["initial_visual_fingerprint"] or config["trainer"]["seed"]!=43 or config["trainer"]["data_seed"]!=43 or config["trainer"]["shuffle_dataset"] is not False or config["trainer"]["train_sampling_strategy"]!="sequential":
                raise RuntimeError("S2 actual training conditions differ beyond the intended optimization/rollout seed")
            verify_stream(records,groups)
            stream_audit={"passed":True,"model_training_seed":43,"stored_input_order_seed":42,"actual_groups":len(groups),"actual_order_matches_original_2048_stream":True,"signature_matches_seed42_except_model_seed":True}
            write_json(directory/"seed43_stream_audit.json",stream_audit); update(last_stream_audit=stream_audit)
            candidates=[]; candidate_identities={}
            for step in STEPS:
                checkpoint=str(directory/f"checkpoint-{step}"); dev_id=f"{name}_dev_{step}"; output=root/f"runs/dev/{dev_id}/predictions.jsonl"
                run(dev_id,["scripts/predict.py",*common,"--adapter-path",checkpoint,"--manifest",str(root/"data/manifests/val.jsonl"),"--output",str(output),"--run-id",dev_id,"--max-new-tokens","32"],(output,output.with_suffix(".jsonl.config.json")),True)
                coverage=output.parent/"coverage.json"
                run(dev_id+"_coverage",["scripts/validate_run.py","predictions","--manifest",str(root/"data/manifests/val.jsonl"),"--predictions",str(output),"--run-id",dev_id,"--checkpoint",checkpoint,"--seed","43","--expected-questions","200","--expected-recipes","200","--output",str(coverage)],(coverage,))
                if not read(coverage)["passed"]: raise RuntimeError("S2 development coverage failed")
                dev_config=read(output.with_suffix(".jsonl.config.json"))
                if dev_config["weight_identity"]["base"]!=protocol["base"] or dev_config["weight_identity"]["adapter"]["directory"]!=checkpoint:
                    raise RuntimeError("S2 development prediction loaded a different Base/checkpoint")
                candidate_identities[step]=dev_config["weight_identity"]["adapter"]
                rows=read_jsonl(output); scores=[score_order(r["raw_response"],r["display_perm"],r.get("target_order",[0,1,2,3])) for r in rows]
                candidates.append({"step":step,"checkpoint":checkpoint,"predictions":str(output),**{k:sum(s[k] for s in scores)/200 for k in ("exact","pair","valid")}})
            chosen=selection(candidates)
            chosen_identity=model_weight_identity(chosen["checkpoint"],adapter=True)
            if chosen_identity!=candidate_identities[chosen["step"]]: raise RuntimeError("Selected S2 weight bytes differ from their real development load")
            write_json(directory/"best_checkpoint.json",{"selection_rule":policy["development"]["selection_rule"],"candidates":candidates,"selected":chosen,"selected_identity":chosen_identity})
            state["selected"][name]=chosen; update()
        outputs={"e0":[],"e3":[]}
        for method in METHODS:
            checkpoint=state["selected"][f"{method}_seed43"]["checkpoint"]
            outputs["e0"].append(predict(method,"e0",root/"data/manifests/test.jsonl",checkpoint,905,597))
        for method in METHODS:
            checkpoint=state["selected"][f"{method}_seed43"]["checkpoint"]; repeat_id=f"eval_{method}_seed43_determinism"
            repeats=[predict(method,f"e3_determinism_{i}",root/policy["evaluation"]["determinism"]["manifest"],checkpoint,10,10,run_id=repeat_id) for i in (1,2)]
            report=root/f"results/seed43/determinism/{method}.json"
            run(f"{method}_determinism_comparison",["ops/compare_greedy.py","--left",str(repeats[0]),"--right",str(repeats[1]),"--expected-inputs","10","--output",str(report)],(report,))
            if read(report)["status"]!="passed": raise RuntimeError("Seed43 greedy repeat differs; E3 paused, backend not changed")
        for method in METHODS:
            outputs["e3"].append(predict(method,"e3",root/"data/manifests/orbit.jsonl",state["selected"][f"{method}_seed43"]["checkpoint"],60,60,orbit=True))
        for experiment,files in outputs.items():
            directory=root/f"results/seed43/{experiment}"
            command=["scripts/summarize_predictions.py",*map(str,files),"--paired","--bootstrap-samples","2000","--bootstrap-seed","20261107","--output-dir",str(directory)]
            if experiment=="e3": command += ["--orbit"]
            run(experiment+"_summary",command,tuple(directory/n for n in ("summary.json","per_record.csv","per_recipe.csv")))
            summary=read(directory/"summary.json")
            if len(summary["groups"])!=2 or len(summary["paired_comparisons"])!=1 or summary["paired_comparisons"][0]["status"]!="computed" or any(group["identity"].get("seed")!=43 for group in summary["groups"]):
                raise RuntimeError("S2 summaries must contain only the two fully paired seed43 groups")
        update(status="completed_optional_seed43_repetition",phase="S2_E0_E3_complete",active_child_pid=None,
               actual_command_wall_seconds=sum(c["elapsed_seconds"] for c in state["commands"]),
               reporting=policy["reporting"],next_action="Report both seeds separately and their differences/range; never pool recipes across training seeds or select the better seed. SFT remains one training seed; original E4 human work unchanged.")
    except BaseException as error:
        update(status="failed",error=f"{type(error).__name__}: {error}"); print(json.dumps({"status":"failed","error":state["error"],"state":str(state_path)}),flush=True); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
