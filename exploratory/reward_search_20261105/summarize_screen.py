#!/usr/bin/env python3
"""CPU-only, descriptive summary of a locked development-only reward screen.

Never launches a model, accesses test predictions, changes training sources, or
promotes a method. Incomplete state requires the explicit --allow-partial flag.
"""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE.parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "src"))
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
from recipeqa.metrics import score_order
from recipeqa.modeling import ensure_data_disk, model_weight_identity, read_jsonl, sha256_file, write_json
from validate_run import validate_prediction_coverage


def require(value, message):
    if not value:
        raise ValueError(message)


def near(actual, expected, message):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), message)


def load(path):
    return json.loads(Path(path).read_text())


def distribution(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values)}


def make_summary(policy_path, state_path, root, allow_partial=False):
    policy_bytes, state_bytes = policy_path.read_bytes(), state_path.read_bytes()
    policy, state = json.loads(policy_bytes), json.loads(state_bytes)
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    state_sha = hashlib.sha256(state_bytes).hexdigest()
    require(policy["status"] == "locked" and state["policy_sha256"] == policy_sha, "Policy/state identity mismatch")
    terminal = state.get("status") == "completed_development_screen"
    require(terminal or allow_partial, "State is incomplete; use --allow-partial to request an explicitly partial summary")
    require(state.get("test_access") is False and not state.get("automatic_promotion", True),
            "State must explicitly forbid test access and automatic promotion")
    remote_root = Path(policy["root"])
    hashes = {}
    receipts = {}

    def project_path(value):
        value = Path(value)
        if value.is_absolute():
            if value.is_relative_to(remote_root):
                value = root / value.relative_to(remote_root)
            else:
                require(value.is_relative_to(root), f"Artifact escaped project: {value}")
        else:
            value = root / value
        value = value.resolve()
        require(value.is_relative_to(root), f"Artifact escaped project: {value}")
        # Avoid even accidentally opening old test files during this analysis.
        require(not any(part.lower() in ("test", "e0", "e3", "e4_main") for part in value.relative_to(root).parts)
                and "test" not in value.name.lower(), f"Test/evaluation artifact forbidden: {value}")
        return value

    def digest(path):
        path = Path(path)
        if path not in hashes:
            hashes[path] = sha256_file(path)
        return hashes[path]

    def check_receipt(path_value, receipt):
        path = project_path(path_value)
        expected = receipt if isinstance(receipt, str) else receipt["sha256"]
        require(digest(path) == expected, f"Artifact SHA mismatch: {path}")
        if isinstance(receipt, dict) and "bytes" in receipt:
            require(path.stat().st_size == receipt["bytes"], f"Artifact length mismatch: {path}")
        receipts[str(path.relative_to(root))] = {"sha256": digest(path), "bytes": path.stat().st_size}
        return path

    for name, expected in policy["source_sha256"].items():
        check_receipt(name, expected)
    for name, expected in state["checks"]["original_source_sha256"].items():
        check_receipt(name, expected)
    check_receipt("configs/protocol_lock.json", state["checks"]["protocol_sha256"])
    check_receipt(policy["train_manifest"], policy["train_manifest_sha256"])
    dev_path = check_receipt(policy["dev_manifest"], policy["dev_manifest_sha256"])
    manifest = read_jsonl(dev_path)
    require(len(manifest) == 200 and len({row["recipe_id"] for row in manifest}) == 200, "Development denominator changed")
    initial = project_path(policy["initial_adapter"])
    check_receipt(initial / "adapter_model.safetensors", policy["initial_adapter_weight_sha256"])
    baseline_path = check_receipt(policy["baseline"]["predictions"], policy["baseline"]["predictions_sha256"])

    def score_dev(path, checkpoint, run_id=None, seed=None):
        rows = read_jsonl(path)
        coverage = validate_prediction_coverage(
            manifest, rows, expected_questions=200, expected_recipes=200,
            expected_checkpoint=str(checkpoint), expected_run_id=run_id, expected_seed=seed,
        )
        # For local copies the raw checkpoint remains the remote path. Retry its
        # policy-equivalent string, not a looser coverage or score criterion.
        if not coverage["passed"] and root != remote_root:
            equivalent = str(remote_root / Path(checkpoint).relative_to(root))
            coverage = validate_prediction_coverage(
                manifest, rows, expected_questions=200, expected_recipes=200,
                expected_checkpoint=equivalent, expected_run_id=run_id, expected_seed=seed,
            )
        require(coverage["passed"], f"Strict development coverage failed: {path}")
        config = load(path.with_suffix(path.suffix + ".config.json"))
        require(config["manifest_sha256"] == policy["dev_manifest_sha256"], "Prediction manifest mismatch")
        expected_identity = model_weight_identity(checkpoint, adapter=True)
        actual_identity = config["weight_identity"]["adapter"]
        require(actual_identity["weight_files"] == expected_identity["weight_files"]
                and actual_identity["auxiliary_files"] == expected_identity["auxiliary_files"],
                f"Prediction checkpoint bytes mismatch: {path}")
        require(config["cli"]["temperature"] == 0 and config["cli"]["num_samples"] == 1
                and config["cli"]["max_new_tokens"] == 32 and not config["cli"].get("process"),
                "Development decoding is not the unchanged greedy short-answer task")
        scores = []
        for row in rows:
            require(not row.get("technical_error"), "Technical development error")
            require(not row["decoding"]["do_sample"] and row["decoding"]["max_new_tokens"] == 32,
                    "Recorded decoding differs from greedy task")
            score = score_order(row["raw_response"], row["display_perm"], row["target_order"])
            require(row["score"] == score, f"Stored development score disagrees with raw text: {path}")
            scores.append(score)
        return {"exact": statistics.mean(score["exact"] for score in scores),
                "pair": statistics.mean(score["pair"] for score in scores),
                "valid": statistics.mean(score["valid"] for score in scores), "n_predictions": len(rows)}

    baseline_metrics = score_dev(baseline_path, initial)
    baseline = {"step": 0, "checkpoint": str(initial), "predictions": str(baseline_path),
                "predictions_sha256": digest(baseline_path), **baseline_metrics}
    for metric in ("exact", "pair", "valid"):
        near(state["checks"]["baseline"][metric], baseline[metric], "State baseline score mismatch")
    expected_stages = []
    for arm in policy["arms"]:
        name = arm["name"]
        expected_stages += [f"{name}_train", f"{name}_verify"]
        for step in range(arm["save_steps"], arm["max_steps"] + 1, arm["save_steps"]):
            expected_stages += [f"{name}_dev_{step}", f"{name}_coverage_{step}"]
    commands = state["commands"]
    require([command["stage"] for command in commands] == expected_stages[:len(commands)], "Command sequence changed or duplicated")
    require(len(commands) <= len(expected_stages), "Unexpected commands after locked screen")
    by_stage = {command["stage"]: command for command in commands}
    completed_commands, failed_commands, running_commands = [], [], []
    wall = 0.0
    for command in commands:
        require(isinstance(command.get("pid"), int) and command["pid"] > 1, "Missing actual child PID")
        args = command["arguments"]
        require("--process" not in args and all("test.jsonl" not in str(arg).lower() for arg in args),
                "Test/process command found in screen")
        if "returncode" not in command:
            running_commands.append(command["stage"])
            continue
        require(math.isfinite(command["elapsed_seconds"]) and command["elapsed_seconds"] >= 0,
                "Invalid command wall time")
        wall += command["elapsed_seconds"]
        if command["returncode"] != 0:
            failed_commands.append(command["stage"])
            continue
        require(command.get("required_artifacts"), "Successful command lacks required artifact receipts")
        for name, receipt in command["required_artifacts"].items():
            check_receipt(name, receipt)
        check_receipt(command["log"], command["log_sha256"])
        completed_commands.append(command["stage"])

    def training_diagnostics(arm, verification):
        name = arm["name"]
        directory = project_path(f"exploratory/reward_search_20261105/screen/arms/{name}")
        require(verification["passed"] and verification["status"] == "verified_complete_exploratory_run",
                "Unverified training run cannot provide diagnostics")
        require(verification["arm"] == name and verification["seed"] == policy["seed"]
                and verification["policy_sha256"] == policy_sha, "Training verification refers to another arm/policy")
        for filename, receipt in verification["required_artifacts"].items():
            check_receipt(directory / filename, receipt)
        gradients = read_jsonl(directory / "gradient_audit.jsonl")
        norms = [row["l2_norm"] for row in gradients]
        require(all(math.isfinite(value) and value >= 0 for value in norms), "Nonfinite audited gradient")
        result = {"verified": True, "effective_adapter_update": verification["effective_adapter_update"],
                  "scientific_outcome": verification["scientific_outcome"],
                  "gradient_norm": distribution(norms),
                  "zero_gradient_updates": sum(value == 0 for value in norms),
                  "nonzero_gradient_updates": sum(value > 0 for value in norms),
                  "prompt_visits": verification["prompt_visits"], "optimizer_steps": verification["optimizer_steps"]}
        if arm["method"] == "sft":
            return result
        groups = read_jsonl(directory / "reward_groups.jsonl")
        rollouts = read_jsonl(directory / "rollouts.jsonl")
        by_group = collections.defaultdict(list)
        for row in rollouts:
            by_group[row["group_id"]].append(row)
        require(len(groups) == 256 and len(rollouts) == 256 * arm["G"], "Verified training budget changed")
        semantic_unique, raw_unique, distinct_rewards, stds, absolute_advantages = [], [], [], [], []
        categories = collections.Counter()
        for group in groups:
            candidates = by_group[group["group_id"]]
            require(len(candidates) == arm["G"], "Verified group size changed")
            semantic_unique.append(len({tuple(row["score"]["semantic_order"]) for row in candidates if row["score"]["valid"]}))
            raw_unique.append(len({row["raw_response"] for row in candidates}))
            distinct_rewards.append(len(set(group["selected_rewards"])))
            stds.append(group["selected_sample_std"])
            absolute_advantages.extend(abs(value) for value in group["reference_advantages"])
            categories[group["reward_rule"]] += 1
        constant = sum(group["selected_zero_variance"] for group in groups)
        result.update({
            "candidate_count": len(rollouts), "group_count": len(groups),
            "constant_reward_groups": constant, "constant_reward_group_fraction": constant / len(groups),
            "active_reward_groups": len(groups) - constant,
            "exact_active_groups": sum(not group["exact_zero_variance"] for group in groups),
            "pair_active_groups": sum(not group["pair_zero_variance"] for group in groups),
            "unique_valid_semantic_orders_per_group": distribution(semantic_unique),
            "unique_raw_strings_per_group": distribution(raw_unique),
            "distinct_selected_reward_levels_per_group": distribution(distinct_rewards),
            "groups_at_least_three_reward_levels": sum(value >= 3 for value in distinct_rewards),
            "selected_reward_sample_std": distribution(stds),
            "offline_reference_absolute_advantage": distribution(absolute_advantages),
            "candidate_exact_fraction": statistics.mean(row["score"]["exact"] for row in rollouts),
            "candidate_pair_mean": statistics.mean(row["score"]["pair"] for row in rollouts),
            "candidate_valid_fraction": statistics.mean(row["score"]["valid"] for row in rollouts),
            "reward_rule_group_counts": dict(categories),
            "note": "Offline advantage audit and on-policy candidate diversity are descriptive, not actual token-gradient attribution or proof of causality",
        })
        return result

    arms, points = [], [{"arm": "common_start", "step": 0, "status": "verified", **baseline_metrics}]
    for arm in policy["arms"]:
        name = arm["name"]
        result = {"name": name, "method": arm["method"], "settings": arm,
                  "status": "pending", "selection": None, "training_diagnostics": None,
                  "command_wall_seconds": sum(command.get("elapsed_seconds", 0) for command in commands
                                              if command["stage"].startswith(name + "_")),
                  "candidates": [dict(baseline)]}
        verify_stage = by_stage.get(f"{name}_verify", {})
        if verify_stage.get("returncode") == 0:
            verification_path = project_path(f"exploratory/reward_search_20261105/screen/arms/{name}/exploratory_verification.json")
            verification = load(verification_path)
            if name in state.get("training_verification", {}):
                saved = state["training_verification"][name]
                require(digest(verification_path) == saved["sha256"], "Training verification receipt changed")
            result["training_diagnostics"] = training_diagnostics(arm, verification)
            result["status"] = "training_verified_dev_pending"
        for step in range(arm["save_steps"], arm["max_steps"] + 1, arm["save_steps"]):
            point = {"arm": name, "step": step, "status": "pending", "exact": None,
                     "pair": None, "valid": None, "n_predictions": None}
            pred_stage = by_stage.get(f"{name}_dev_{step}", {})
            cov_stage = by_stage.get(f"{name}_coverage_{step}", {})
            if pred_stage.get("returncode") == cov_stage.get("returncode") == 0:
                require(result["training_diagnostics"] is not None, "Development output has no verified training run")
                directory = project_path(f"exploratory/reward_search_20261105/screen/dev/{name}/step{step}")
                path = directory / "predictions.jsonl"
                checkpoint = project_path(f"exploratory/reward_search_20261105/screen/arms/{name}/checkpoint-{step}")
                require(load(directory / "coverage.json")["passed"], "Recorded development coverage failed")
                run_id = f"reward_search_{name}_seed{policy['seed']}_dev{step}"
                metrics = score_dev(path, checkpoint, run_id, policy["seed"])
                point.update(status="verified", **metrics)
                candidate = {"step": step, "checkpoint": str(checkpoint), "predictions": str(path),
                             "predictions_sha256": digest(path), **metrics}
                result["candidates"].append(candidate)
                for recorded in state.get("development_candidates", {}).get(name, []):
                    if recorded["step"] == step:
                        require(recorded["predictions_sha256"] == digest(path), "State dev prediction SHA mismatch")
                        for metric in ("exact", "pair", "valid"):
                            near(recorded[metric], metrics[metric], "State dev score differs from raw rescore")
            points.append(point)
        if name in state.get("selected", {}):
            require(len(result["candidates"]) == 5, "Selected arm lacks all four complete development points")
            expected = max(result["candidates"], key=lambda value: (value["exact"], value["pair"], -value["step"]))
            saved = state["selected"][name]
            selection_file = project_path(f"exploratory/reward_search_20261105/screen/arms/{name}/best_checkpoint.json")
            require(load(selection_file) == saved, "Selection file and state differ")
            recorded = saved["selected"]
            require(recorded["step"] == expected["step"]
                    and project_path(recorded["checkpoint"]) == Path(expected["checkpoint"])
                    and recorded["predictions_sha256"] == expected["predictions_sha256"], "Development selection rule violated")
            for metric in ("exact", "pair", "valid"):
                near(recorded[metric], expected[metric], "Selected metric differs from raw rescore")
            identity = model_weight_identity(expected["checkpoint"], adapter=True)
            require(saved["selected_identity"]["weight_files"] == identity["weight_files"]
                    and saved["selected_identity"]["auxiliary_files"] == identity["auxiliary_files"], "Selected checkpoint bytes changed")
            check_receipt(selection_file, digest(selection_file))
            result["selection"] = {**expected, "exact_minus_common_start_pp": 100 * (expected["exact"] - baseline["exact"]),
                                   "pair_minus_common_start_pp": 100 * (expected["pair"] - baseline["pair"])}
            result["status"] = "completed_selected_development_only"
        arms.append(result)
    complete_arms = sum(arm["status"] == "completed_selected_development_only" for arm in arms)
    if terminal:
        require(state["phase"] == "all_arms_selected_no_test" and complete_arms == len(policy["arms"])
                and len(commands) == len(expected_stages) and not failed_commands and not running_commands
                and all(command.get("returncode") == 0 for command in commands),
                "Terminal-complete state lacks all arms or successful commands")
    return {
        "status": "verified_complete_development_screen" if terminal else "verified_partial_development_snapshot",
        "scope": "Post-hoc exploratory screen on the reused fixed development200; not heldout confirmation or an official benchmark SOTA",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "policy_sha256": policy_sha,
        "state_snapshot_sha256": state_sha, "state_status": state["status"], "state_phase": state.get("phase"),
        "state_updated_utc": state.get("updated_utc"), "state_changed_during_summary": hashlib.sha256(state_path.read_bytes()).hexdigest() != state_sha,
        "summary_source_sha256": sha256_file(__file__), "test_access": False, "automatic_promotion": False,
        "planned_arms": len(policy["arms"]), "completed_selected_arms": complete_arms,
        "baseline": baseline, "arms": arms, "all_development_points": points,
        "command_cost": {"completed_commands": len(completed_commands), "failed_commands": failed_commands,
                         "running_commands": running_commands, "attempted_commands": len(commands),
                         "planned_commands": len(expected_stages), "sum_finished_command_wall_seconds": wall,
                         "sum_finished_command_wall_hours": wall / 3600,
                         "pipeline_snapshot_elapsed_seconds": state.get("elapsed_seconds"),
                         "wall_limit_seconds": policy["wall_limit_seconds"],
                         "note": "Command wall is measured execution time, not the cloud rental bill or a claim of equal compute"},
        "verified_artifacts": receipts,
        "limitations": ["Same shared SFT checkpoint and one screening seed44; not a full seed replication",
                        "All arms share maximum256promptvisits; G8 uses twice G4 candidate generation, and SFT/RL compute differs",
                        "G4 versus G8 also changes sampling temperature in this locked design",
                        "group versus none changes gradient magnitude as part of the algorithm; fixed LR does not isolate objective shape",
                        "Checkpoint and arm selection reuse development200, creating selection optimism",
                        "Original test outcomes were known before exploration; no new test results are read here",
                        "Human image/ambiguity and explanation audits remain unresolved"],
    }


def write_outputs(summary, output):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    point_columns = ["arm", "step", "status", "exact", "pair", "valid", "n_predictions"]
    with (output / "development_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=point_columns)
        writer.writeheader()
        writer.writerows({key: point.get(key) for key in point_columns} for point in summary["all_development_points"])
    rows = []
    for arm in summary["arms"]:
        selected = arm["selection"] or {}
        diag = arm["training_diagnostics"] or {}
        rows.append({"arm": arm["name"], "method": arm["method"], "status": arm["status"],
                     "selected_step": selected.get("step"), "dev_exact": selected.get("exact"),
                     "dev_pair": selected.get("pair"), "dev_valid": selected.get("valid"),
                     "exact_minus_common_start_pp": selected.get("exact_minus_common_start_pp"),
                     "effective_adapter_update": diag.get("effective_adapter_update"),
                     "active_reward_groups": diag.get("active_reward_groups"),
                     "constant_reward_group_fraction": diag.get("constant_reward_group_fraction"),
                     "mean_unique_valid_orders": diag.get("unique_valid_semantic_orders_per_group", {}).get("mean"),
                     "groups_at_least_three_reward_levels": diag.get("groups_at_least_three_reward_levels"),
                     "mean_gradient_norm": diag.get("gradient_norm", {}).get("mean"),
                     "zero_gradient_updates": diag.get("zero_gradient_updates"),
                     "command_wall_seconds": arm["command_wall_seconds"]})
    with (output / "arm_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    baseline = summary["baseline"]
    lines = ["# Development-only reward screen", "", summary["scope"], "",
             f"Status: **{summary['status']}**. Completed selections: {summary['completed_selected_arms']}/{summary['planned_arms']}.",
             f"Common SFT start: Exact {100*baseline['exact']:.2f}%, Pair {100*baseline['pair']:.2f}%, valid {100*baseline['valid']:.2f}% on 200 development recipes.", "",
             "| Arm | Selected step | Dev Exact | Change vs common start | Active groups | Mean unique valid orders |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in summary["arms"]:
        selected, diag = arm["selection"], arm["training_diagnostics"] or {}
        if selected:
            step, exact, change = selected["step"], f"{100*selected['exact']:.2f}%", f"{selected['exact_minus_common_start_pp']:+.2f} pp"
        else:
            step, exact, change = "pending", "pending", "pending"
        active = diag.get("active_reward_groups", "—")
        unique = diag.get("unique_valid_semantic_orders_per_group", {}).get("mean")
        lines.append(f"| {arm['name']} | {step} | {exact} | {change} | {active} | {unique:.3f} |" if unique is not None
                     else f"| {arm['name']} | {step} | {exact} | {change} | {active} | — |")
    lines += ["", "All four planned checkpoint points are preserved in development_points.csv, including pending points in a partial snapshot.",
              "A selected step0 means continued training did not beat the shared start under the locked Exact→Pair→earlier rule.", "",
              f"Finished command wall: {summary['command_cost']['sum_finished_command_wall_hours']:.3f} hours. This is not the rental bill.", "",
              "## What can be concluded", "",
              "These are descriptive, development-selected results. They can identify candidates for a separately locked follow-up and show whether exploration, reward variation and greedy accuracy move together. They do not establish test improvement, a causal mechanism, stable cross-seed superiority, or official SOTA.", "",
              "## Limits", ""]
    lines.extend(f"- {value}" for value in summary["limitations"])
    (output / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=HERE / "SCREEN_PROTOCOL.json")
    parser.add_argument("--state", type=Path, default=HERE / "screen/state.json")
    parser.add_argument("--root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE / "screen_summary")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    ensure_data_disk(args.output_dir)
    summary = make_summary(args.policy.resolve(), args.state.resolve(), args.root.resolve(), args.allow_partial)
    write_outputs(summary, args.output_dir.resolve())
    print(json.dumps({"status": summary["status"], "completed_arms": summary["completed_selected_arms"],
                      "planned_arms": summary["planned_arms"], "output_dir": str(args.output_dir.resolve()),
                      "test_access": False}, indent=2))


if __name__ == "__main__":
    main()
