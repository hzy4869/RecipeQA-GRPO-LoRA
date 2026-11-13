#!/usr/bin/env python3
"""CPU-only summaries of real pilot evidence, throughput, and budget components.

  python scripts/summarize_pilot.py --run-dir runs/pilot_sft runs/pilot_exact \
    runs/pilot_pair --output results/pilot_summary.json

Reads canonical root logs only. Raw attempt directories are used exclusively to
list known completed-attempt wall costs, never pooled into canonical step/group
statistics. Missing evidence remains pending; no model-effect judgment is made.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median


def read_json(path, missing):
    if not path.is_file():
        missing.append(path.name)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_rows(path, missing):
    if not path.is_file():
        missing.append(path.name)
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object: {path}:{line_number}")
                rows.append(row)
    return rows


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def quantile(values, q):
    values = sorted(values)
    location = (len(values) - 1) * q
    low, high = math.floor(location), math.ceil(location)
    return values[low] + (location - low) * (values[high] - values[low])


def timing_stats(values):
    values = [float(value) for value in values if positive(value)]
    return {"status": "observed" if values else "pending", "n": len(values),
            "median_seconds": median(values) if values else None,
            "p90_seconds": quantile(values, .9) if values else None,
            "mean_seconds": mean(values) if values else None,
            "min_seconds": min(values) if values else None, "max_seconds": max(values) if values else None}


def attempt_name(row):
    return row.get("attempt_id") or "legacy_untracked"


def summarize_run(run_dir, *, discard_first=2, minimum_steps=20,
                  sft_formal_steps=256, rl_formal_steps=1024,
                  sft_save_steps=64, rl_save_steps=256):
    root = Path(run_dir).resolve()
    missing, issues, warnings = [], [], []
    config = read_json(root / "resolved_config.json", missing)
    outcome = read_json(root / "train_outcome.json", missing)
    method = config.get("method")
    is_sft = method == "sft"
    is_rl = method in ("grpo_exact", "grpo_pair")
    if not (is_sft or is_rl):
        issues.append("unknown_or_missing_method")
    rows = {name: read_rows(root / f"{name}.jsonl", missing) for name in (
        "step_times", "trainer_metrics", "gradient_audit")}
    for name in (("actual_prompt_visits",) if is_sft else ("reward_groups", "rollouts") if is_rl else ()):
        rows[name] = read_rows(root / f"{name}.jsonl", missing)
    times = rows["step_times"]
    by_step = {}
    for row in times:
        step = row.get("global_step")
        if type(step) is not int or step < 1:
            issues.append("step_times_has_invalid_global_step")
        elif step in by_step:
            issues.append(f"ambiguous_duplicate_optimizer_step:{step}")
        else:
            by_step[step] = row
    signatures = {row.get("run_signature_sha256") for records in rows.values() for row in records if row.get("run_signature_sha256")}
    if config.get("run_signature_sha256"):
        signatures.add(config["run_signature_sha256"])
    if len(signatures) > 1:
        issues.append("mixed_run_signatures_in_canonical_logs")
    ordered_steps = sorted(by_step)
    attempts = defaultdict(list)
    segments = []
    for step in ordered_steps:
        attempt = attempt_name(by_step[step])
        attempts[attempt].append(step)
        if not segments or segments[-1] != attempt:
            segments.append(attempt)
    if len(set(segments)) != len(segments):
        issues.append("interleaved_attempts_in_canonical_optimizer_steps")
    if "legacy_untracked" in attempts and len(attempts) > 1:
        issues.append("cannot_disambiguate_mixed_tracked_and_untracked_attempts")
    expected_last = outcome.get("global_step") if type(outcome.get("global_step")) is int else max(ordered_steps, default=0)
    full_step_coverage = bool(expected_last) and set(ordered_steps) == set(range(1, expected_last + 1))
    if not full_step_coverage:
        issues.append("canonical_optimizer_step_coverage_incomplete")
    if len(by_step) < minimum_steps:
        issues.append("fewer_than_minimum_complete_pilot_optimizer_steps")
    stable_steps = {step for sequence in attempts.values() for step in sequence[discard_first:]}
    step_values = [by_step[step].get("optimizer_step_seconds") for step in ordered_steps if step in stable_steps]
    intervals, no_save_intervals, post_save_intervals = [], [], []
    skipped_cross_attempt_intervals = 0
    trainer = config.get("trainer", {})
    cli = config.get("cli", {})
    pilot_save_steps = trainer.get("save_steps", cli.get("save_steps"))
    for previous_step, step in zip(ordered_steps, ordered_steps[1:]):
        previous, current = by_step[previous_step], by_step[step]
        if attempt_name(previous) != attempt_name(current):
            skipped_cross_attempt_intervals += 1
            continue
        if step not in stable_steps or step != previous_step + 1:
            continue
        a, b = previous.get("elapsed_seconds"), current.get("elapsed_seconds")
        if positive(a) and positive(b) and b > a:
            interval = b - a
            intervals.append(interval)
            if type(pilot_save_steps) is int and pilot_save_steps > 0:
                (post_save_intervals if previous_step % pilot_save_steps == 0 else no_save_intervals).append(interval)
    optimizer_timing = timing_stats(step_values)
    wall_timing = timing_stats(intervals)
    nonsave_timing = timing_stats(no_save_intervals)
    postsave_timing = timing_stats(post_save_intervals)
    grad_counts = Counter(row.get("global_step_before_update") for row in rows["gradient_audit"])
    if any(type(step) is not int or count != 1 for step, count in grad_counts.items()):
        issues.append("ambiguous_or_invalid_gradient_step_records")
    if set(grad_counts) != set(range(expected_last)):
        issues.append("canonical_gradient_step_coverage_incomplete")
    n_nonzero = 0
    for row in rows["gradient_audit"]:
        norm = row.get("l2_norm")
        if type(norm) not in (float, int) or not math.isfinite(norm) or norm < 0:
            issues.append("nonfinite_or_invalid_gradient_norm")
        n_nonzero += int(positive(norm) and row.get("nonzero_gradient_tensors", 0) > 0)
    micro = trainer.get("per_device_train_batch_size", 1)
    accumulation = trainer.get("gradient_accumulation_steps", cli.get("gradient_accumulation_steps"))
    group_size = trainer.get("num_generations", cli.get("num_generations")) if is_rl else None
    generation_batch = trainer.get("generation_batch_size", cli.get("generation_batch_size")) if is_rl else None
    sampling = {"single_process_assumption": True, "optimizer_steps_observed": len(by_step),
                "global_step_reported_or_last_observed": expected_last,
                "minimum_timing_pilot_steps": minimum_steps, "minimum_step_count_observed": full_step_coverage and len(by_step) >= minimum_steps}
    signal = None
    if is_sft:
        visits = rows.get("actual_prompt_visits", [])
        expected_visits = expected_last * micro * accumulation if type(micro) is int and type(accumulation) is int else None
        sampling.update({"actual_prompt_visit_records": len(visits), "expected_prompt_visits": expected_visits,
                         "visit_count_matches": len(visits) == expected_visits,
                         "unique_visit_ids": len({row["visit_id"] for row in visits if row.get("visit_id") is not None}),
                         "visit_note": "Visits are recorded after actual training_step forward/backward; prefetched collator batches are excluded by the training implementation."})
        if expected_visits is None or len(visits) != expected_visits:
            issues.append("sft_visit_count_does_not_match_optimizer_budget")
    elif is_rl:
        groups = rows.get("reward_groups", [])
        rollouts = rows.get("rollouts", [])
        groups_per_update = generation_batch // group_size if type(generation_batch) is int and type(group_size) is int and group_size > 0 and generation_batch % group_size == 0 else None
        expected_groups = expected_last * groups_per_update if groups_per_update is not None else None
        expected_answers = expected_groups * group_size if expected_groups is not None else None
        group_ids = [row.get("group_id") for row in groups]
        if any(value is None for value in group_ids) or len(set(group_ids)) != len(group_ids):
            issues.append("missing_or_duplicate_reward_group_ids")
        n_rewarded = 0
        exact_nonconstant = pair_nonconstant = 0
        n_valid = 0
        for row in groups:
            exact, pair = row.get("exact_rewards"), row.get("pair_rewards")
            if not isinstance(exact, list) or not isinstance(pair, list) or len(exact) != group_size or len(pair) != group_size:
                issues.append("reward_group_missing_complete_G_candidates")
                continue
            n_rewarded += len(exact)
            exact_nonconstant += len(set(exact)) > 1
            pair_nonconstant += len(set(pair)) > 1
            n_valid += row.get("valid_count", 0)
            if len(set(exact)) > 1 and len(set(pair)) == 1:
                issues.append("impossible_exact_only_reward_variation")
        actual_keys = [(row.get("group_id"), row.get("sample_index")) for row in rollouts]
        if len(set(actual_keys)) != len(actual_keys):
            issues.append("duplicate_rollout_group_sample_keys")
        groups_by_step = Counter(row.get("global_step_before_update") for row in groups)
        if groups_per_update is not None and groups_by_step != Counter({step: groups_per_update for step in range(expected_last)}):
            issues.append("reward_group_steps_do_not_match_canonical_updates")
        sampling.update({"G": group_size, "generation_batch_size": generation_batch,
                         "independent_groups_per_optimizer_update": groups_per_update,
                         "actual_prompt_groups": len(groups), "expected_prompt_groups": expected_groups,
                         "actual_rewarded_candidates": n_rewarded, "actual_raw_rollout_records": len(rollouts),
                         "expected_candidates": expected_answers,
                         "group_count_matches": len(groups) == expected_groups,
                         "answer_count_matches": n_rewarded == expected_answers and len(rollouts) == expected_answers})
        if len(groups) != expected_groups or n_rewarded != expected_answers or len(rollouts) != expected_answers:
            issues.append("grpo_group_or_candidate_count_does_not_match_optimizer_budget")
        signal = {"n_candidate_groups": len(groups), "exact_nonconstant_groups": exact_nonconstant,
                  "pair_nonconstant_groups": pair_nonconstant,
                  "exact_nonconstant_rate": exact_nonconstant / len(groups) if groups else None,
                  "pair_nonconstant_rate": pair_nonconstant / len(groups) if groups else None,
                  "valid_candidate_rate": n_valid / n_rewarded if n_rewarded else None,
                  "interpretation": "Descriptive on-policy training candidates, not a same-bank comparison between trained policies or proof of reasoning improvement."}
    # Detect reward/visit/gradient rows accidentally drawn from another attempt
    # at the same canonical update. Multiple sequential attempts are legitimate.
    for name, records in rows.items():
        if name in ("step_times", "trainer_metrics"):
            continue
        for row in records:
            step = row.get("global_step_before_update")
            reference = by_step.get(step + 1) if type(step) is int else None
            if reference is not None and row.get("attempt_id") and reference.get("attempt_id") and row["attempt_id"] != reference["attempt_id"]:
                issues.append(f"cross_attempt_canonical_record:{name}")
                break
    completed_attempts = []
    incomplete_attempts = []
    for path in sorted((root / "attempts").glob("*/attempt.json")):
        metadata = json.loads(path.read_text())
        attempt_outcome = path.parent / "train_outcome.json"
        if attempt_outcome.is_file():
            value = json.loads(attempt_outcome.read_text()).get("elapsed_seconds_including_preprocessing")
            if positive(value):
                completed_attempts.append({"attempt_id": metadata.get("attempt_id"), "wall_seconds": value})
                continue
        incomplete_attempts.append({"attempt_id": metadata.get("attempt_id"), "status": metadata.get("status", "unknown")})
    elapsed = outcome.get("elapsed_seconds_including_preprocessing")
    resumed = bool(outcome.get("resume_global_step", 0) or cli.get("resume"))
    wall_scope = "latest_invocation_only_after_resume" if resumed else "whole_original_invocation_including_preprocessing_and_finalization"
    runtime = outcome.get("metrics", {}).get("train_runtime")
    if not positive(runtime):
        runtime = next((row["train_runtime"] for row in reversed(rows["trainer_metrics"]) if positive(row.get("train_runtime"))), None)
    outside_trainer = max(0, elapsed - runtime) if positive(elapsed) and positive(runtime) else None
    formal_steps = sft_formal_steps if is_sft else rl_formal_steps if is_rl else None
    formal_save = sft_save_steps if is_sft else rl_save_steps
    step_median = optimizer_timing["median_seconds"]
    step_p90 = optimizer_timing["p90_seconds"]
    interval_median = wall_timing["median_seconds"]
    nonsave_median = nonsave_timing["median_seconds"]
    postsave_median = postsave_timing["median_seconds"]
    save_proxy = max(0, postsave_median - nonsave_median) if postsave_median is not None and nonsave_median is not None else None
    unsafe = any("ambiguous" in item or "duplicate" in item or "mixed_" in item or "interleaved" in item or "cross_attempt" in item for item in issues)
    extrapolation = {"status": "pending_ambiguous_logs" if unsafe else "provisional_insufficient_steps" if len(by_step) < minimum_steps or not full_step_coverage else "components_only" if step_median is not None and formal_steps is not None else "pending_timing",
                     "formal_optimizer_steps": formal_steps, "planned_checkpoint_every_steps": formal_save,
                     "planned_checkpoint_events": math.ceil(formal_steps / formal_save) if formal_steps else None,
                     "optimizer_only_median_projection_seconds": step_median * formal_steps if not unsafe and step_median is not None and formal_steps else None,
                     "optimizer_only_p90_sensitivity_seconds": step_p90 * formal_steps if not unsafe and step_p90 is not None and formal_steps else None,
                     "pilot_save_frequency_wall_interval_projection_seconds": interval_median * formal_steps if not unsafe and interval_median is not None and formal_steps else None,
                     "noncheckpoint_wall_median_projection_seconds": nonsave_median * formal_steps if not unsafe and nonsave_median is not None and formal_steps else None,
                     "checkpoint_overhead_proxy_seconds_per_event": save_proxy if not unsafe else None,
                     "latest_invocation_outside_trainer_seconds": outside_trainer,
                     "formula": "formal total = representative non-checkpoint wall seconds/update * formal updates + timed checkpoint cost * formal checkpoint events + one-time load/preprocess/finalization + all timed dev evaluations + end-point inference + restart margin",
                     "limitations": ["Optimizer-step timers exclude checkpoint saves, dev evaluation and may exclude data prefetch.",
                         "Elapsed adjacent-step intervals include intervening save/log/data costs but omit initial load and final save.",
                         "Pilot save frequency may be much higher than formal save frequency; its interval projection is a sensitivity, not the final estimate.",
                         "Save overhead is only a difference-of-medians proxy, confounded by input/generation variability. Independently time saves/dev before quoting a full budget.",
                         "P90 times steps is a conservative latency sensitivity, not a probabilistic total-runtime confidence bound.",
                         "No missing dev/save/end-point cost is filled with zero, and no total budget or scientific model effect is automatically approved."]}
    status = "completed_evidence_summary" if outcome.get("status") == "completed" and full_step_coverage and not issues and not missing else "pending_or_inconsistent_evidence"
    return {"run_dir": str(root), "method": method, "status": status, "missing_files": missing,
            "issues": sorted(set(issues)), "warnings": warnings, "sampling": sampling,
            "timing": {"discard_first_optimizer_steps_per_attempt": discard_first,
                "canonical_steps_by_attempt": {name: {"n": len(values), "first": min(values), "last": max(values)} for name, values in attempts.items()},
                "skipped_cross_attempt_elapsed_intervals": skipped_cross_attempt_intervals,
                "stable_optimizer_step": optimizer_timing, "stable_elapsed_adjacent_step_interval": wall_timing,
                "stable_noncheckpoint_wall_interval": nonsave_timing, "stable_postcheckpoint_wall_interval": postsave_timing,
                "latest_invocation_wall_seconds": elapsed if positive(elapsed) else None, "latest_invocation_wall_scope": wall_scope,
                "trainer_runtime_latest_invocation_seconds": runtime,
                "known_completed_attempt_wall_sum_seconds": sum(item["wall_seconds"] for item in completed_attempts) if completed_attempts else None,
                "completed_attempt_costs_not_pooled_into_step_statistics": completed_attempts,
                "attempts_without_complete_wall_cost": incomplete_attempts,
                "rental_note": "Known process wall time is not platform-billed instance time; failed attempts, idle time, downloads and orchestration must remain in the separate billing ledger."},
            "memory": {"peak_allocated_bytes": max([row.get("peak_allocated_bytes", 0) for row in times] + [outcome.get("peak_allocated_bytes", 0)]) or None,
                       "peak_reserved_bytes": max([row.get("peak_reserved_bytes", 0) for row in times] + [outcome.get("peak_reserved_bytes", 0)]) or None},
            "gradient": {"n_recorded_updates": len(grad_counts), "n_nonzero_finite_gradient_updates": n_nonzero},
            "training_signal": signal, "formal_extrapolation": extrapolation,
            "scope": "Pilot implementation/throughput evidence only; not a proof of recovery correctness, generalization, or relative model quality."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--discard-first", type=int, default=2)
    parser.add_argument("--minimum-steps", type=int, default=20)
    parser.add_argument("--sft-formal-steps", type=int, default=256)
    parser.add_argument("--rl-formal-steps", type=int, default=1024)
    parser.add_argument("--sft-checkpoint-steps", type=int, default=64)
    parser.add_argument("--rl-checkpoint-steps", type=int, default=256)
    args = parser.parse_args(argv)
    if args.discard_first < 0 or any(value < 1 for value in (args.minimum_steps, args.sft_formal_steps, args.rl_formal_steps, args.sft_checkpoint_steps, args.rl_checkpoint_steps)):
        parser.error("Timing counts and production budgets must be positive; discard-first must be nonnegative")
    output = args.output.resolve()
    data_disk = Path("/root/autodl-tmp")
    if data_disk.is_dir() and not output.is_relative_to(data_disk.resolve()):
        parser.error("On the rented server --output must remain under /root/autodl-tmp")
    report = {"schema_version": 1, "runs": [summarize_run(path, discard_first=args.discard_first,
                minimum_steps=args.minimum_steps, sft_formal_steps=args.sft_formal_steps,
                rl_formal_steps=args.rl_formal_steps, sft_save_steps=args.sft_checkpoint_steps,
                rl_save_steps=args.rl_checkpoint_steps) for path in args.run_dir]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": [{"method": run["method"], "status": run["status"],
                      "optimizer_steps": run["sampling"]["optimizer_steps_observed"], "issues": run["issues"]} for run in report["runs"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
