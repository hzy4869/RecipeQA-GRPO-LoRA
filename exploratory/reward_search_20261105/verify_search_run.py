#!/usr/bin/env python3
"""Independently rescore one completed exploratory training run, without GPUs."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recipeqa.metrics import score_order
from recipeqa.modeling import hash_json, model_weight_identity, read_jsonl, sha256_file, write_json


def require(value, message):
    if not value:
        raise ValueError(message)


def near(actual, expected, label):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            f"Incorrect {label}: {actual!r} != {expected!r}")


def load(path):
    return json.loads(Path(path).read_text())


def verify_run(run_dir, manifest, arm, seed, initial_adapter):
    run = Path(run_dir).resolve()
    records = read_jsonl(manifest)
    config = load(run / "resolved_config.json")
    outcome = load(run / "train_outcome.json")
    signature = load(run / "run_signature.json")
    inventory = load(run / "trainable_parameters.json")
    steps = arm["max_steps"]
    method = arm["method"]
    require(signature["sha256"] == hash_json(signature["payload"]), "Run signature hash is inconsistent")
    require(signature["payload"].get("post_hoc_exploration") is True, "Missing exploratory identity")
    for name, digest in signature["payload"]["code_sha256"].items():
        source = {"train_reward_search.py": HERE / "train_reward_search.py",
                  "exploratory_rewards.py": HERE / "rewards.py",
                  "modeling.py": ROOT / "src/recipeqa/modeling.py",
                  "metrics.py": ROOT / "src/recipeqa/metrics.py"}.get(name)
        require(source is not None and sha256_file(source) == digest, f"Signed source changed: {name}")
    cli = config["cli"]
    require(config["versions"] == signature["payload"]["versions"], "Package versions disagree with signature")
    require(signature["payload"]["base_identity"] == model_weight_identity(cli["model_path"]),
            "Base model bytes changed")
    require(config["method"] == f"exploratory_{method}", "Wrong run method")
    expected_cli = {"seed": seed, "max_steps": steps, "save_steps": arm["save_steps"],
                    "learning_rate": arm["learning_rate"],
                    "gradient_accumulation_steps": arm["gradient_accumulation_steps"]}
    if method == "grpo":
        expected_cli.update(reward=arm["reward"], num_generations=arm["G"],
                            generation_batch_size=arm["generation_batch_size"],
                            temperature=arm["temperature"], scale_rewards=arm.get("scale_rewards", "group"))
    for key, value in expected_cli.items():
        require(cli.get(key) == value, f"Wrong CLI {key}: {cli.get(key)} != {value}")
    require(not cli.get("resume"), "Screening must start fresh, not resume")
    require(Path(cli["initial_adapter"]).resolve() == Path(initial_adapter).resolve(), "Wrong common start path")
    actual_initial = model_weight_identity(initial_adapter, adapter=True)
    require(signature["payload"]["initial_W_adapter_identity"] == actual_initial, "Initial adapter bytes changed")
    require(config["manifest_sha256"] == sha256_file(manifest)
            == signature["payload"]["manifest_sha256"], "Training manifest changed")
    require(signature["payload"]["scientific_cli"] == {
        key: value for key, value in cli.items() if key not in ("resume", "output_dir")
    }, "Resolved CLI disagrees with signed configuration")
    require(config["input"] == {"min_pixels": 65536, "max_pixels": 262144,
                                "recipe_token_budget": 1536, "max_input_tokens": 4096,
                                "max_completion_tokens": 32, "prompt_version": "recipeqa-order-v1"},
            "Input budget or task changed")
    require(outcome["status"] == "completed" and outcome["global_step"] == steps, "Incomplete training")
    require(outcome["run_signature_sha256"] == config["run_signature_sha256"] == signature["sha256"],
            "Artifact signature mismatch")
    require(outcome.get("actual_visit_sequence_verified") is True, "Missing consumption verification")
    require(outcome["initial_adapter_fingerprint"] == config["initial_adapter_fingerprint"], "Initial fingerprint disagrees")
    changed = outcome["final_adapter_fingerprint"] != outcome["initial_adapter_fingerprint"]
    require(outcome["effective_adapter_update"] == changed, "Wrong update flag")
    require(outcome.get("scientific_outcome") == ("effective_training" if changed else "no_effective_adapter_update"),
            "Scientific outcome mislabels ineffective run")
    require(outcome["visual_weights_unchanged"] is True
            and outcome["final_visual_fingerprint"] == outcome["initial_visual_fingerprint"]
            == config["initial_visual_fingerprint"], "Visual weights changed")
    require(len(inventory["parameters"]) > 0 and all(
        "lora_" in row["name"] and ".language_model.layers." in row["name"]
        for row in inventory["parameters"]), "Unapproved trainable parameter")
    require(sum(row["numel"] for row in inventory["parameters"]) == inventory["trainable_parameters"],
            "Trainable inventory count mismatch")
    trainer = config["trainer"]
    expected_trainer = {"shuffle_dataset": False, "train_sampling_strategy": "sequential",
                        "per_device_train_batch_size": 1, "save_only_model": False,
                        "weight_decay": 0.0, "max_grad_norm": 1.0}
    if method == "grpo":
        expected_trainer.update(beta=0.0, loss_type="grpo", num_iterations=1,
                                scale_rewards=arm.get("scale_rewards", "group"),
                                max_completion_length=32, use_vllm=False,
                                mask_truncated_completions=False)
    else:
        expected_trainer.update(completion_only_loss=True, packing=False, max_length=None)
        require(load(run / "label_mask_audit.json")["passed"] is True, "SFT label masking failed")
    for key, value in expected_trainer.items():
        require(trainer.get(key) == value, f"Trainer setting mismatch: {key}")
    attempts = list((run / "attempts").iterdir())
    require(len(attempts) == 1 and attempts[0].is_dir(), "Unexpected restart or multiple attempts")
    require(load(attempts[0] / "attempt.json")["status"] == "completed", "Attempt did not complete")
    audits = read_jsonl(run / "input_audit.jsonl")
    require(len(audits) == len(records), "Incomplete preprocessing audit")
    input_by_visit = {}
    for record, audit in zip(records, audits):
        require(all(audit[key] == record[key] for key in ("qid", "recipe_id", "presentation_id")), "Input audit identity mismatch")
        require(0 < audit["input_tokens"] <= 4096 and audit["recipe_tokens_used"] <= 1536,
                "Input budget exceeded")
        input_by_visit[record["visit_id"]] = audit
    expected_visits = steps * (arm["generation_batch_size"] // arm["G"] if method == "grpo"
                              else arm["gradient_accumulation_steps"])
    require(len(records) == expected_visits, "Screening must consume the exact full manifest")
    gradients = read_jsonl(run / "gradient_audit.jsonl")
    require([row["global_step_before_update"] for row in gradients] == list(range(steps)), "Gradient coverage mismatch")
    nonzero = 0
    for row in gradients:
        require(math.isfinite(row["l2_norm"]) and row["l2_norm"] >= 0, "Nonfinite gradient")
        require(0 <= row["nonzero_gradient_tensors"] <= row["trainable_gradient_tensors"]
                == len(inventory["parameters"]), "Gradient tensor inventory mismatch")
        nonzero += row["l2_norm"] > 0
    times = read_jsonl(run / "step_times.jsonl")
    require([row["global_step"] for row in times] == list(range(1, steps + 1)), "Step time coverage mismatch")
    require(all(math.isfinite(row["optimizer_step_seconds"]) and row["optimizer_step_seconds"] >= 0
                for row in times), "Invalid step timing")
    receipt_files = ["resolved_config.json", "train_outcome.json", "trainable_parameters.json",
                     "run_signature.json", "input_audit.jsonl", "gradient_audit.jsonl", "step_times.jsonl",
                     "trainer_metrics.jsonl", "trainer_state.json"]
    checkpoint_steps = list(range(arm["save_steps"], steps + 1, arm["save_steps"]))
    actual_checkpoints = sorted(int(path.name.split("-")[-1]) for path in run.glob("checkpoint-*") if path.is_dir())
    require(actual_checkpoints == checkpoint_steps, "Planned checkpoint coverage mismatch")
    checkpoint_identities = {}
    for step in checkpoint_steps:
        checkpoint = run / f"checkpoint-{step}"
        audit = load(checkpoint / "recipeqa_checkpoint_audit.json")
        require(audit["global_step"] == step and audit["run_signature_sha256"] == signature["sha256"],
                "Checkpoint signature or step mismatch")
        required = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"}
        require(set(audit["state_sha256"]) == required and audit["complete_single_process_resume_state"] is True,
                "Incomplete recovery state")
        for name, digest in audit["state_sha256"].items():
            require(sha256_file(checkpoint / name) == digest, "Checkpoint state bytes changed")
            receipt_files.append(f"checkpoint-{step}/{name}")
        identity = model_weight_identity(checkpoint, adapter=True)
        require(identity == audit["adapter_identity"], "Checkpoint adapter bytes changed")
        require(load(checkpoint / "trainer_state.json")["global_step"] == step, "Trainer checkpoint step mismatch")
        checkpoint_identities[str(step)] = identity
        receipt_files.extend(f"checkpoint-{step}/{item['name']}" for item in identity["weight_files"] + identity["auxiliary_files"])
        receipt_files.append(f"checkpoint-{step}/recipeqa_checkpoint_audit.json")
    final_identity = model_weight_identity(run / "adapter_final", adapter=True)
    require(final_identity["weights_sha256"] == checkpoint_identities[str(steps)]["weights_sha256"],
            "Final adapter differs from final optimizer checkpoint")
    receipt_files.extend(f"adapter_final/{item['name']}" for item in final_identity["weight_files"] + final_identity["auxiliary_files"])
    zero_groups, all_constant_steps = 0, 0
    if method == "sft":
        visits = read_jsonl(run / "actual_prompt_visits.jsonl")
        receipt_files += ["actual_prompt_visits.jsonl", "label_mask_audit.json"]
        require(len(visits) == expected_visits, "SFT visit count differs")
        for index, (actual, record) in enumerate(zip(visits, records)):
            require(actual["global_step_before_update"] == index // arm["gradient_accumulation_steps"], "SFT update mapping differs")
            require(all(actual[key] == record[key] for key in ("qid", "recipe_id", "presentation_id", "visit_id", "display_perm")),
                    "SFT consumed stream differs")
            require(actual["input_hash"] == input_by_visit[record["visit_id"]]["input_hash"], "SFT preprocessed input differs")
    else:
        groups = read_jsonl(run / "reward_groups.jsonl")
        rollouts = read_jsonl(run / "rollouts.jsonl")
        receipt_files += ["reward_groups.jsonl", "rollouts.jsonl"]
        require(len(groups) == expected_visits and len(rollouts) == expected_visits * arm["G"], "RL candidate budget differs")
        require(len({row["group_id"] for row in groups}) == expected_visits, "Duplicate reward group")
        by_step = collections.defaultdict(list)
        for index, (group, record) in enumerate(zip(groups, records)):
            step = index // (arm["generation_batch_size"] // arm["G"])
            require(group["global_step_before_update"] == step, "RL update mapping differs")
            require(all(group[key] == record[key] for key in ("qid", "recipe_id", "presentation_id", "visit_id", "display_perm", "target_order")),
                    "RL consumed stream differs")
            audit = input_by_visit[record["visit_id"]]
            require(group["input_hash"] == audit["input_hash"] and group["input_tokens"] == audit["input_tokens"],
                    "RL preprocessed input differs")
            candidates = rollouts[index * arm["G"]:(index + 1) * arm["G"]]
            scores = []
            for sample, candidate in enumerate(candidates):
                require(candidate["group_id"] == group["group_id"] and candidate["sample_index"] == sample,
                        "Candidate grouping or index differs")
                require(all(candidate[key] == record[key] for key in ("qid", "recipe_id", "presentation_id", "visit_id", "display_perm", "target_order")),
                        "Candidate belongs to another input")
                score = score_order(candidate["raw_response"], record["display_perm"], record["target_order"])
                require(candidate["score"] == score, "Stored score differs from strict raw rescore")
                require(candidate["selected_reward"] == arm["reward"]
                        and candidate["input_hash"] == audit["input_hash"]
                        and candidate["input_tokens"] == audit["input_tokens"], "Candidate reward/input metadata differs")
                scores.append(score)
            require(group["G"] == arm["G"] and group["selected_reward"] == arm["reward"]
                    and group["scale_rewards"] == arm.get("scale_rewards", "group"), "Group recipe differs from locked arm")
            require(group["exact_rewards"] == [score["exact"] for score in scores]
                    and group["pair_rewards"] == [score["pair"] for score in scores]
                    and group["valid_count"] == sum(score["valid"] for score in scores), "Group diagnostics disagree")
            vectors = {"exact": [float(score["exact"]) for score in scores],
                       "pair": [score["pair"] for score in scores],
                       "exp_inv": [math.exp(-score["inversions"]) if score["valid"] else 0.0 for score in scores]}
            has_success = any(score["exact"] for score in scores)
            vectors["success_gate_pair"] = vectors["exact"] if has_success else vectors["pair"]
            expected = vectors[arm["reward"]]
            mean, std = statistics.mean(expected), statistics.stdev(expected)
            for offset, (candidate, reward) in enumerate(zip(candidates, expected)):
                near(candidate["reward"], reward, "candidate reward")
                near(group["selected_rewards"][offset], reward, "selected group reward")
                for name in vectors:
                    near(candidate["component_rewards"][name], vectors[name][offset], "alternate reward")
                    near(group["component_rewards"][name][offset], vectors[name][offset], "group alternate reward")
                denominator = std + 1e-4 if arm.get("scale_rewards", "group") == "group" else 1.0
                near(group["reference_advantages"][offset], (reward - mean) / denominator, "reference advantage")
            near(group["selected_mean"], mean, "group mean")
            near(group["selected_sample_std"], std, "group sample std")
            require(group["selected_zero_variance"] == (len(set(expected)) == 1), "Wrong constant group flag")
            expected_rule = ("exact_when_success_present" if has_success else "pair_when_no_success") if arm["reward"] == "success_gate_pair" else arm["reward"]
            require(group["reward_rule"] == expected_rule and group["group_has_success"] == has_success,
                    "Wrong success gate")
            zero_groups += len(set(expected)) == 1
            by_step[step].append(len(set(expected)) == 1)
        all_constant_steps = sum(all(flags) for flags in by_step.values())
    artifacts = {name: {"sha256": sha256_file(run / name), "bytes": (run / name).stat().st_size}
                 for name in sorted(set(receipt_files))}
    return {"status": "verified_complete_exploratory_run", "passed": True,
            "run_dir": str(run), "method": method, "arm": arm["name"], "seed": seed,
            "optimizer_steps": steps, "prompt_visits": expected_visits,
            "effective_adapter_update": changed, "scientific_outcome": outcome["scientific_outcome"],
            "nonzero_gradient_updates": nonzero, "zero_variance_groups": zero_groups,
            "all_constant_reward_updates": all_constant_steps,
            "constant_group_gradient_note": "Descriptive counts; accumulated gradients, floating precision and optimizer momentum prevent simplistic zero-group=unchanged-weight claims",
            "run_signature_sha256": signature["sha256"], "checkpoint_identities": checkpoint_identities,
            "required_artifacts": artifacts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        policy = load(args.policy)
        arms = [arm for arm in policy["arms"] if arm["name"] == args.arm]
        require(len(arms) == 1, "Arm name must be unique in locked policy")
        manifest = Path(policy["root"]) / policy["train_manifest"]
        report = verify_run(args.run_dir, manifest, arms[0], policy["seed"], policy["initial_adapter"])
        report["policy_sha256"] = sha256_file(args.policy)
    except Exception as error:
        report = {"passed": False, "status": "failed_validation", "error": f"{type(error).__name__}: {error}"}
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "required_artifacts"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
