#!/usr/bin/env python3
"""CPU-only E1 learning curves from actual canonical logs and dev predictions.

Required CLI: --run-dir RUN ... --output-dir DIR. Optional --pipeline-state FILE
and --bin-updates 32. Exit 0: valid artifacts (possibly partial); 2: no drawable
observations; 1: contradictory inputs. Inspect manifest completeness flags.

Training-loop elapsed wall excludes dev and preprocessing, is not GPU kernel
time or equal FLOPs. Multiple-attempt elapsed cannot be safely accumulated here:
the wall axis is refused for that run rather than hiding restart/rollback costs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recipeqa.metrics import score_order


class Evidence:
    def __init__(self):
        self.sources = {}
        self.unavailable = []

    def read(self, path, *, lines=False, optional=False):
        path = Path(path).resolve()
        if not path.is_file():
            if optional:
                return [] if lines else {}
            raise ValueError(f"Required evidence missing: {path}")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if str(path) in self.sources and self.sources[str(path)]["sha256"] != digest:
            raise ValueError(f"Evidence changed during analysis: {path}")
        self.sources[str(path)] = {"path": str(path), "sha256": digest, "bytes": len(raw)}
        return [json.loads(line) for line in raw.decode().splitlines() if line.strip()] if lines else json.loads(raw)

    def absent(self, run, item, reason, severity="optional"):
        self.unavailable.append({"run": run, "item": item, "reason": reason, "severity": severity})


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def probability(value):
    return finite(value) and 0 <= value <= 1


def optional_average(values):
    return mean(values) if values and all(finite(value) for value in values) else None


def load_run(path, evidence, pipeline_path=None, bin_updates=32):
    path = Path(path).resolve()
    name = path.name
    root = path.parent.parent
    config = evidence.read(path / "resolved_config.json")
    method = config.get("method")
    if method not in ("sft", "grpo_exact", "grpo_pair"):
        raise ValueError(f"Unknown method for {path}")
    cfg = config.get("trainer", {})
    cli = config.get("cli", {})
    formal_steps = cli.get("max_steps")
    if type(formal_steps) is not int or formal_steps <= 0 or formal_steps % 4:
        raise ValueError(f"Cannot identify fixed four checkpoint steps for {name}")
    expected_points = [formal_steps * k // 4 for k in (1, 2, 3, 4)]
    times = evidence.read(path / "step_times.jsonl", lines=True)
    timeline = {}
    for row in times:
        step = row.get("global_step")
        if type(step) is not int or step <= 0 or step in timeline or not finite(row.get("elapsed_seconds")) or row["elapsed_seconds"] <= 0:
            raise ValueError(f"Invalid/duplicate timing step in {name}")
        timeline[step] = row
    if not timeline or set(timeline) != set(range(1, max(timeline) + 1)):
        raise ValueError(f"Canonical timing coverage is not contiguous for {name}")
    signatures = {row.get("run_signature_sha256") for row in times if row.get("run_signature_sha256")}
    if config.get("run_signature_sha256"):
        signatures.add(config["run_signature_sha256"])
    if len(signatures) > 1:
        raise ValueError(f"Mixed run signatures: {name}")
    attempts = {row.get("attempt_id", "legacy_untracked") for row in times}
    wall_available = len(attempts) == 1
    if not wall_available:
        evidence.absent(name, "development_wall_axis", "Multiple attempts: callback clocks reset and discarded/restart costs cannot be aligned from canonical elapsed alone", "required")
    elif any(timeline[b]["elapsed_seconds"] <= timeline[a]["elapsed_seconds"] for a, b in zip(sorted(timeline), sorted(timeline)[1:])):
        raise ValueError(f"Nonmonotone elapsed clock in one attempt: {name}")
    before = "actual_prompt_visits" if method == "sft" else "reward_groups"
    visits = evidence.read(path / f"{before}.jsonl", lines=True)
    counts = Counter()
    for row in visits:
        step = row.get("global_step_before_update")
        if type(step) is not int or step < 0 or step + 1 not in timeline:
            raise ValueError(f"Cannot align {before} update with canonical timing: {name}")
        recorded_attempt = row.get("attempt_id")
        if recorded_attempt and recorded_attempt != timeline[step + 1].get("attempt_id"):
            raise ValueError(f"Cross-attempt {before} record: {name}/{step}")
        counts[step + 1] += 1
    g = cfg.get("num_generations", cli.get("num_generations"))
    generation_batch = cfg.get("generation_batch_size", cli.get("generation_batch_size"))
    micro = cfg.get("per_device_train_batch_size", 1)
    accum = cfg.get("gradient_accumulation_steps", cli.get("gradient_accumulation_steps"))
    expected_per_step = micro * accum if method == "sft" and type(micro) is int and type(accum) is int else generation_batch // g if type(g) is int and g > 0 and type(generation_batch) is int and generation_batch % g == 0 else None
    if expected_per_step is None or any(counts[step] != expected_per_step for step in timeline):
        raise ValueError(f"Actual prompt exposure counts disagree with configured updates: {name}")
    cumulative = {}
    total = 0
    for step in sorted(timeline):
        total += counts[step]
        cumulative[step] = total
    best = evidence.read(path / "best_checkpoint.json", optional=True)
    state = evidence.read(pipeline_path or root / "metadata/formal_training_pipeline.json", optional=True)
    candidates = best.get("candidates", state.get("development_candidates", {}).get(name, []))
    selected = best.get("selected", state.get("selected", {}).get(name))
    points = []
    seen_steps = set()
    for candidate in sorted(candidates, key=lambda item: item["step"]):
        step = candidate.get("step")
        if step not in expected_points or step in seen_steps or step not in timeline:
            raise ValueError(f"Unexpected/duplicate development checkpoint: {name}/{step}")
        seen_steps.add(step)
        predictions_path = Path(candidate["predictions"])
        if not predictions_path.is_absolute():
            predictions_path = root / predictions_path
        predictions = evidence.read(predictions_path, lines=True)
        if len(predictions) != 200 or len({str(row["recipe_id"]) for row in predictions}) != 200:
            raise ValueError(f"Development points require 200 independent recipes: {name}/{step}")
        if any(row.get("technical_error") or str(Path(row["checkpoint"]).resolve()) != str(Path(candidate["checkpoint"]).resolve()) for row in predictions):
            raise ValueError(f"Technical failure/checkpoint identity mismatch in dev point: {name}/{step}")
        scores = [score_order(row["raw_response"], row["display_perm"], row.get("target_order", [0, 1, 2, 3])) for row in predictions]
        metrics = {metric: mean(row[metric] for row in scores) for metric in ("exact", "pair", "middle_pair", "valid")}
        if any(not probability(candidate.get(metric)) or abs(candidate[metric] - metrics[metric]) > 1e-8 for metric in ("exact", "pair", "valid")):
            raise ValueError(f"Stored dev selection metrics disagree with raw predictions: {name}/{step}")
        points.append({"step": step, "actual_prompt_visits": cumulative[step],
                       "loop_wall_minutes": timeline[step]["elapsed_seconds"] / 60 if wall_available else None,
                       "n_recipes": 200, "selected": bool(selected and selected.get("step") == step), **metrics})
    dev_complete = seen_steps == set(expected_points) and wall_available and selected is not None
    if selected is not None and selected.get("step") != max(points, key=lambda point: (point["exact"], point["pair"], -point["step"]))["step"]:
        raise ValueError(f"Selection receipt disagrees with fixed dev rule: {name}")
    if not dev_complete:
        evidence.absent(name, "four_development_points", f"Expected steps {expected_points}; observed {sorted(seen_steps)}; selected receipt present={selected is not None}; wall axis available={wall_available}", "required")
    outcome = evidence.read(path / "train_outcome.json", optional=True)
    label = {"sft": "SFT-cont", "grpo_exact": "Exact-GRPO", "grpo_pair": "Pair-GRPO"}[method] + f" (seed {cli.get('seed', 'unknown')})"
    result = {"run_dir": str(path), "name": name, "label": label, "method": method,
              "expected_checkpoint_steps": expected_points, "development_points": points,
              "required_dev_curves_complete": dev_complete,
              "wall_axis": "Training-loop elapsed wall (min); excludes preprocessing/dev; includes prior save/log/data gaps, not current save/finalization",
              "actual_total_prompt_visits_observed": total,
              "full_invocation_wall_seconds_separate": outcome.get("elapsed_seconds_including_preprocessing"),
              "full_invocation_scope": "latest invocation only if resumed; not platform billed time",
              "signal_bins": [], "required_signal_curves_complete": method == "sft"}
    if method == "sft":
        return result
    telemetry_rows = evidence.read(path / "trainer_metrics.jsonl", lines=True, optional=True)
    telemetry = defaultdict(dict)
    for row in telemetry_rows:
        step = row.get("global_step")
        if step not in timeline:
            continue
        for metric in ("completions/mean_length", "completions/clipped_ratio"):
            if metric in row:
                if metric in telemetry[step] and telemetry[step][metric] != row[metric]:
                    raise ValueError(f"Conflicting telemetry at update {step}: {name}/{metric}")
                telemetry[step][metric] = row[metric]
    rollouts = evidence.read(path / "rollouts.jsonl", lines=True, optional=True)
    lengths = defaultdict(list)
    rollout_keys = set()
    for row in rollouts:
        key = row.get("group_id"), row.get("sample_index")
        if key in rollout_keys:
            raise ValueError(f"Duplicate raw rollout: {name}/{key}")
        rollout_keys.add(key)
        step = row.get("global_step_before_update")
        if type(step) is not int or step + 1 not in timeline:
            raise ValueError(f"Unaligned raw rollout: {name}")
        if row.get("attempt_id") and row["attempt_id"] != timeline[step + 1].get("attempt_id"):
            raise ValueError(f"Cross-attempt raw rollout: {name}")
        lengths[step + 1].append(row.get("completion_tokens"))
    by_step = defaultdict(list)
    group_ids = set()
    for row in visits:
        key = row.get("group_id")
        if key is None or key in group_ids:
            raise ValueError(f"Missing/duplicate reward group: {name}")
        group_ids.add(key)
        exact, pair = row.get("exact_rewards"), row.get("pair_rewards")
        if not isinstance(exact, list) or not isinstance(pair, list) or len(exact) != g or len(pair) != g or not all(value in (0, 1) for value in exact) or not all(probability(value) for value in pair):
            raise ValueError(f"Invalid complete reward group: {name}/{key}")
        if type(row.get("valid_count")) is not int or not 0 <= row["valid_count"] <= g:
            raise ValueError(f"Missing valid candidate count: {name}/{key}")
        by_step[row["global_step_before_update"] + 1].append(row)
    for first in range(1, max(timeline) + 1, bin_updates):
        last = min(first + bin_updates - 1, max(timeline))
        chunk = [row for step in range(first, last + 1) for row in by_step[step]]
        n_candidates = len(chunk) * g
        raw_lengths = [value for step in range(first, last + 1) for value in lengths[step]]
        if len(raw_lengths) == n_candidates and all(finite(value) and value >= 0 for value in raw_lengths):
            length = mean(raw_lengths)
            length_source = "raw completion_tokens"
        else:
            length = optional_average([telemetry[step].get("completions/mean_length") for step in range(first, last + 1)])
            length_source = "recorded TRL completions/mean_length" if length is not None else None
        clipped = optional_average([telemetry[step].get("completions/clipped_ratio") for step in range(first, last + 1)])
        if clipped is not None and not probability(clipped):
            raise ValueError(f"Invalid reported clipped ratio: {name}")
        result["signal_bins"].append({"first_update": first, "last_update": last,
            "actual_prompt_visits": cumulative[last], "n_groups": len(chunk), "n_candidates": n_candidates,
            "exact_nonconstant_rate": mean(len(set(row["exact_rewards"])) > 1 for row in chunk),
            "pair_nonconstant_rate": mean(len(set(row["pair_rewards"])) > 1 for row in chunk),
            "valid_rate": sum(row["valid_count"] for row in chunk) / n_candidates,
            "exact_mean_reward": sum(sum(row["exact_rewards"]) for row in chunk) / n_candidates,
            "pair_mean_reward": sum(sum(row["pair_rewards"]) for row in chunk) / n_candidates,
            "completion_mean_tokens": length, "length_source": length_source,
            "reported_clipped_ratio": clipped})
    for field in ("completion_mean_tokens", "reported_clipped_ratio"):
        if any(row[field] is None for row in result["signal_bins"]):
            evidence.absent(name, field, "Some actual bins lack this recorded metric; no zero or length-based truncation substitute generated")
    result["required_signal_curves_complete"] = max(timeline) == formal_steps and bool(result["signal_bins"])
    if not result["required_signal_curves_complete"]:
        evidence.absent(name, "full_signal_trajectory", f"Observed {max(timeline)}/{formal_steps} updates", "required")
    return result


def save(fig, name, out, plt, outputs):
    for ext in ("svg", "pdf"):
        path = out / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", metadata={"Creator": "RecipeQA recorded training dynamics"})
        outputs.append(str(path))
    plt.close(fig)


def figures(runs, out, bin_updates, outputs):
    os.environ["MPLCONFIGDIR"] = str(out / ".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "text.usetex": False})
    drawable = [run for run in runs if any(point["loop_wall_minutes"] is not None for point in run["development_points"])]
    if drawable:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for ax, metric in zip(axes.flat, ("exact", "pair", "middle_pair", "valid")):
            for index, run in enumerate(drawable):
                points = [point for point in run["development_points"] if point["loop_wall_minutes"] is not None]
                x = [point["loop_wall_minutes"] for point in points]
                ax.plot(x, [point[metric] for point in points], "o-", label=run["label"], markersize=4)
                for point in points:
                    if metric == "exact":
                        ax.annotate(f"n={point['actual_prompt_visits']}", (point["loop_wall_minutes"], point[metric]),
                                    xytext=(2, 5 + index * 7), textcoords="offset points", fontsize=6)
                    if point["selected"]:
                        ax.scatter([point["loop_wall_minutes"]], [point[metric]], marker="*", s=130, facecolors="none", edgecolors="black", linewidths=.8, zorder=5)
            ax.set_title({"exact": "Full ordering accuracy", "pair": "Correct pair fraction", "middle_pair": "Middle relation (2 < 3)", "valid": "Valid answer rate"}[metric])
            ax.set_xlabel("Training-loop elapsed wall (min)")
            ax.set_ylabel("Fixed dev200 recipe macro")
            ax.set_ylim(-.025, 1.12 if metric == "exact" else 1.025)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
        fig.suptitle("Recorded development checkpoints; n = actual prompt visits, star = selected checkpoint\nWall excludes preprocessing/dev and is not equal FLOPs or GPU kernel-active time", fontsize=10)
        save(fig, "development_vs_training_wall", out, plt, outputs)
    policies = [run for run in runs if run["method"] != "sft" and run["signal_bins"]]
    if policies:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        specifications = (("exact_nonconstant_rate", "Exact reward: nonconstant groups"),
            ("pair_nonconstant_rate", "Pair reward: nonconstant groups"), ("valid_rate", "Valid candidate fraction"),
            ("completion_mean_tokens", "Mean completion length (tokens)"),
            ("reported_clipped_ratio", "Recorded TRL clipped ratio"))
        for ax, (field, title) in zip(list(axes.flat)[:5], specifications):
            for run in policies:
                rows = run["signal_bins"]
                if any(row[field] is not None for row in rows):
                    ax.plot([row["actual_prompt_visits"] for row in rows], [row[field] if row[field] is not None else float("nan") for row in rows], "o-", markersize=3, label=run["label"])
            ax.set_title(title)
            if not ax.lines:
                ax.text(.5, .5, "Metric not recorded", transform=ax.transAxes,
                        ha="center", va="center", color="#666666")
            if field != "completion_mean_tokens":
                ax.set_ylim(-.025, 1.025)
            else:
                ax.set_ylim(bottom=0)
        for run in policies:
            for metric, style in (("exact_mean_reward", "-"), ("pair_mean_reward", "--")):
                rows = run["signal_bins"]
                axes[1, 2].plot([row["actual_prompt_visits"] for row in rows], [row[metric] for row in rows], style,
                                label=f"{run['label']}: {'Exact' if metric.startswith('exact') else 'Pair'}")
        axes[1, 2].set_title("Mean on-policy outcome rewards")
        axes[1, 2].set_ylim(-.025, 1.025)
        for ax in axes.flat:
            ax.set_xlabel("Actual prompt visits")
            ax.spines[["top", "right"]].set_visible(False)
            if ax.lines:
                ax.legend(frameon=False, fontsize=6, loc="best")
        fig.suptitle(f"On-policy diagnostics: non-overlapping bins of at most {bin_updates} updates\nEach branch samples its own candidates; these curves are not the identical W bank or held-out accuracy", fontsize=10)
        save(fig, "on_policy_dynamics", out, plt, outputs)
    return matplotlib.__version__


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pipeline-state", type=Path)
    parser.add_argument("--bin-updates", type=int, default=32)
    args = parser.parse_args(argv)
    if args.bin_updates < 1:
        parser.error("--bin-updates must be positive")
    if len({str(path.resolve()) for path in args.run_dir}) != len(args.run_dir):
        parser.error("Repeated --run-dir")
    out = args.output_dir.resolve()
    if Path("/root/autodl-tmp").is_dir() and not out.is_relative_to(Path("/root/autodl-tmp").resolve()):
        parser.error("Remote outputs must stay on /root/autodl-tmp")
    evidence = Evidence()
    runs = []
    for path in args.run_dir:
        try:
            runs.append(load_run(path, evidence, args.pipeline_state, args.bin_updates))
        except FileNotFoundError as error:
            evidence.absent(path.name, "run", str(error), "required")
        except ValueError as error:
            # Missing evidence may occur in an honest pre-completion preview;
            # contradictory evidence must fail rather than create a false curve.
            if str(error).startswith("Required evidence missing:"):
                evidence.absent(path.name, "run", str(error), "required")
            else:
                raise
    out.mkdir(parents=True, exist_ok=True)
    outputs = []
    version = figures(runs, out, args.bin_updates, outputs)
    required_dev = len(runs) == len(args.run_dir) and all(run["required_dev_curves_complete"] for run in runs)
    required_signal = len(runs) == len(args.run_dir) and all(run["required_signal_curves_complete"] for run in runs)
    # Record the same-exposure view from actual checkpoint observations only.
    # No interpolation is defensible where the four time supports do not overlap.
    exposure_checkpoints = []
    for visits in (512, 1024, 1536, 2048):
        observed = [{"run": run["name"], **point} for run in runs
                    for point in run["development_points"]
                    if point["actual_prompt_visits"] == visits]
        exposure_checkpoints.append({"actual_prompt_visits": visits, "observations": observed,
            "all_requested_runs_observed": len(observed) == len(args.run_dir)})
    payload = {"schema_version": 1, "runs": runs, "unavailable": evidence.unavailable,
               "required_dev_curves_complete": required_dev, "required_signal_curves_complete": required_signal,
               "common_exposure_checkpoints": exposure_checkpoints,
               "bin_updates": args.bin_updates, "source_hashes": list(evidence.sources.values()),
               "notes": ["Development curves use the actual fixed 200-recipe predictions used for checkpoint selection, not test data.",
                   "Training-loop elapsed excludes preprocessing/dev/current save/finalization; full invocation wall is reported separately. It is neither hardware-active GPU time nor equal compute.",
                   "Four observed checkpoint time supports can be disjoint across methods. Connecting lines are visual guides, not additional measurements; no equal-time interpolation or extrapolation is performed.",
                   "The common-exposure JSON contains only observed checkpoints at 512/1024/1536/2048 actual visits, under the same fixed dev selection protocol; equal maximum exposure is not equal compute.",
                   "On-policy rates are group/candidate-weighted non-overlapping actual bins; no spline/synthetic observations or missing-zero substitution.",
                   "Multiple-attempt cost clocks are refused; canonical update/exposure diagnostics may still be available."]}
    (out / "training_dynamics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    manifest = {**{key: value for key, value in payload.items() if key != "runs"}, "outputs": outputs,
                "matplotlib_version": version, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "scoring_module_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "src/recipeqa/metrics.py").read_bytes()).hexdigest(),
                "run_development_completeness": [{"run": run["name"], "expected_steps": run["expected_checkpoint_steps"],
                    "observed_steps": [point["step"] for point in run["development_points"]], "complete": run["required_dev_curves_complete"]} for run in runs]}
    (out / "figures_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output_dir": str(out), "generated_files": len(outputs), "required_dev_curves_complete": required_dev,
                      "required_signal_curves_complete": required_signal, "unavailable": evidence.unavailable}, ensure_ascii=False))
    return 0 if outputs else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError) as error:
        raise SystemExit(f"Contradictory dynamics evidence: {error}") from error
