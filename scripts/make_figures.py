#!/usr/bin/env python3
"""Make E0/E2/E3 scientific SVG/PDF figures from actual saved summaries.

  python scripts/make_figures.py --summary results/e0/summary.json \
    results/e3/summary.json --output-dir results/figures \
    --label group_001=SFT-cont --label group_002=Exact-GRPO

E3 correctness matrices also require per_record.csv beside their summary.json.
Missing experiments are reported explicitly; no mock/placeholder research data
is generated. Each figure is exported as SVG and PDF. NumPy/Matplotlib caches
are confined to output-dir/.mplconfig; no website or dashboard is created.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import permutations
import json
import math
import os
from pathlib import Path
import sys

PAIRS = ("1_2", "1_3", "1_4", "2_3", "2_4", "3_4")
PRESENTATIONS = tuple(permutations(range(4)))


def finite_number(value, description):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing/nonfinite {description}: {value!r}")
    return float(value)


def probability(value, description):
    value = finite_number(value, description)
    if not 0 <= value <= 1:
        raise ValueError(f"Out-of-range probability for {description}: {value}")
    return value


def load_groups(paths, labels):
    groups = []
    inputs = []
    seen = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            raise ValueError(f"Repeated --summary file: {path}")
        seen.add(path)
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload.get("groups"), list) or not payload["groups"]:
            raise ValueError(f"No actual scored groups in {path}")
        inputs.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for group in payload["groups"]:
            group_id = group.get("group_id")
            if not isinstance(group_id, str):
                raise ValueError(f"Summary group lacks group_id: {path}")
            identity = group.get("identity", {})
            label = labels.get(f"{path.parent.name}:{group_id}", labels.get(group_id))
            if label is None:
                label = str(identity.get("run_id") or identity.get("checkpoint") or identity.get("model_id") or group_id)
                if identity.get("seed") is not None:
                    label += f" (seed {identity['seed']})"
            groups.append({"source_path": path, "group_id": group_id, "label": label,
                           "identity": identity, "data": group})
    return groups, inputs


def local_data(groups):
    heatmap = []
    distributions = []
    for item in groups:
        summary = item["data"].get("scores", {})
        n = summary.get("n_scored", 0)
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"No scored E0/E2 predictions for {item['label']}")
        macro = summary.get("recipe_macro", {})
        heatmap.append([probability(macro.get(f"pair_{pair}"), f"{item['label']} pair {pair}") for pair in PAIRS])
        bins = summary.get("pair_bins", {})
        row = [probability(bins.get(str(k), {}).get("rate_all_scored"), f"{item['label']} q bin {k}") for k in range(7)]
        row.append(probability(summary.get("invalid", {}).get("rate_all_scored"), f"{item['label']} invalid rate"))
        if not math.isclose(sum(row), 1, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"Pair bins plus invalid must sum to 1: {item['label']}")
        distributions.append(row)
    return heatmap, distributions


def orbit_correctness_data(groups):
    """Require exact same recipe/question coverage across displayed models."""
    cache = {}
    model_rows = []
    common_questions = None
    for item in groups:
        path = item["source_path"].parent / "per_record.csv"
        if not path.is_file():
            raise ValueError(f"E3 requires actual per-record predictions: missing {path}")
        if path not in cache:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                cache[path] = list(csv.DictReader(stream))
        lookup = {}
        for row in cache[path]:
            if row.get("group_id") != item["group_id"]:
                continue
            if row.get("technical_error") or row.get("exact") == "":
                raise ValueError(f"Technical missingness in E3 for {item['label']}")
            display = tuple(json.loads(row["display_perm"]))
            if display not in PRESENTATIONS:
                raise ValueError(f"Invalid display permutation in {path}")
            key = row["recipe_id"], row["qid"]
            if (key, display) in lookup:
                raise ValueError(f"Duplicate E3 display: {item['label']} {key} {display}")
            exact = probability(float(row["exact"]), "binary orbit correctness")
            if exact not in (0, 1):
                raise ValueError("Orbit exact must be binary")
            lookup[key, display] = int(exact)
        questions = {key for key, _ in lookup}
        if not questions:
            raise ValueError(f"No E3 rows found for {item['label']}")
        expected_n = item["data"].get("orbit", {}).get("n_complete_questions")
        if len(questions) != expected_n or len(lookup) != 24 * len(questions):
            raise ValueError(f"E3 is not a complete 24-display design for {item['label']}")
        if common_questions is not None and questions != common_questions:
            raise ValueError("Cannot compare E3 models with different recipe/question sets")
        common_questions = questions
        model_rows.append(lookup)
    ordered = sorted(common_questions or ())
    matrices = [[[lookup[key, display] for display in PRESENTATIONS] for key in ordered] for lookup in model_rows]
    return ordered, matrices


def position_data(groups):
    raw, centered = [], []
    common_questions = None
    for item in groups:
        orbit = item["data"].get("orbit", {})
        question_rows = orbit.get("per_question", [])
        questions = {(str(row["recipe_id"]), str(row["qid"])) for row in question_rows}
        if not questions or len(questions) != orbit.get("n_complete_questions") or any(row.get("n_displays") != 24 for row in question_rows):
            raise ValueError(f"Position comparison needs complete actual orbit identities for {item['label']}")
        if common_questions is not None and questions != common_questions:
            raise ValueError("Cannot compare position matrices from different recipe/question sets")
        common_questions = questions
        matrix = item["data"].get("rank_position", {})
        rate = matrix.get("recipe_macro_rates")
        center = matrix.get("recipe_macro_row_centered")
        if not isinstance(rate, list) or len(rate) != 4 or not isinstance(center, list) or len(center) != 4:
            raise ValueError(f"Missing 4x4 rank-position metrics for {item['label']}")
        if any(not isinstance(row, list) or len(row) != 4 for row in (*rate, *center)):
            raise ValueError(f"Invalid rank-position shape for {item['label']}")
        checked_rate = [[probability(value, "rank-position accuracy") for value in row] for row in rate]
        checked_center = [[finite_number(value, "row-centered rank-position accuracy") for value in row] for row in center]
        for row, normalized in zip(checked_rate, checked_center):
            expected = [value - sum(row) / 4 for value in row]
            if any(abs(a - b) > 1e-8 for a, b in zip(expected, normalized)):
                raise ValueError(f"Row-centering mismatch for {item['label']}")
        raw.append(checked_rate)
        centered.append(checked_center)
    return raw, centered


def draw_grid(ax, matrix, *, cmap, vmin, vmax):
    image = ax.pcolormesh(matrix, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", rasterized=False)
    ax.invert_yaxis()
    ax.set_xlim(0, len(matrix[0]))
    ax.set_ylim(len(matrix), 0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def annotate_grid(ax, matrix, *, centered=False, color_bound=1):
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            text = f"{value:+.2f}" if centered else f"{value:.2f}"
            dark = abs(value) > .65 * color_bound if centered else value > .63
            ax.text(x + .5, y + .5, text, ha="center", va="center", fontsize=9,
                    color="white" if dark else "black")


def save(fig, stem, out, plt, outputs):
    for suffix in ("svg", "pdf"):
        path = out / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", metadata={"Creator": "RecipeQA analysis: make_figures.py"})
        outputs.append(str(path.resolve()))
    plt.close(fig)


def plot_local(groups, out, plt, outputs, prefix="e2"):
    heatmap, distribution = local_data(groups)
    labels = [item["label"] for item in groups]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, max(3.7, .45 * len(groups) + 2.1)), layout="constrained", gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    image = draw_grid(ax, heatmap, cmap="Blues", vmin=0, vmax=1)
    annotate_grid(ax, heatmap)
    ax.set_xticks([index + .5 for index in range(6)], [pair.replace("_", "<") for pair in PAIRS])
    ax.set_yticks([index + .5 for index in range(len(groups))], labels)
    ax.set_xlabel("Relation between selected target ranks")
    ax.set_title("Six relations (recipe macro)")
    colorbar = fig.colorbar(image, ax=ax, fraction=.05, pad=.025)
    colorbar.set_label("Accuracy")
    colorbar.solids.set_rasterized(False)
    width = .8 / len(groups)
    for index, (label, values) in enumerate(zip(labels, distribution)):
        x = [position - .4 + width * (index + .5) for position in range(8)]
        axes[1].bar(x, values, width=width, label=label)
    axes[1].set_xticks(range(8), [f"{k}/6" for k in range(7)] + ["Invalid"])
    axes[1].set_xlabel("Correct directed relations")
    axes[1].set_ylabel("Fraction of all scored questions")
    axes[1].set_ylim(0, min(1, max(max(row) for row in distribution) * 1.2 + .02))
    axes[1].set_title("Prediction distribution (question micro)")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    save(fig, f"{prefix}_relations_and_distribution", out, plt, outputs)


def plot_orbit(groups, out, plt, outputs):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    questions, matrices = orbit_correctness_data(groups)
    n = len(questions)
    fig, axes = plt.subplots(1, len(groups), figsize=(4.1 * len(groups) + .4, max(4, min(10, .09 * n + 2.2))), squeeze=False, layout="constrained")
    for column, (item, matrix) in enumerate(zip(groups, matrices)):
        ax = axes[0][column]
        draw_grid(ax, matrix, cmap=ListedColormap(["#f1f1f1", "#1769a6"]), vmin=0, vmax=1)
        ax.set_xticks([index + .5 for index in range(24)], ["".join(str(value + 1) for value in display) for display in PRESENTATIONS], rotation=90, fontsize=6.5)
        tick_rows = list(range(0, n, max(1, n // 6)))
        if n - 1 not in tick_rows:
            tick_rows.append(n - 1)
        ax.set_yticks([row + .5 for row in tick_rows], [str(row + 1) for row in tick_rows])
        ax.set_ylabel("Recipe index (identical order across models)" if column == 0 else "")
        ax.set_xlabel("Input permutation of selected target images")
        ax.set_title(item["label"])
    fig.suptitle(f"Full display orbit: {n} recipes x 24 presentations", fontsize=12)
    axes[0][-1].legend(handles=[Patch(facecolor="#1769a6", label="Correct"), Patch(facecolor="#f1f1f1", edgecolor="#999999", label="Incorrect / invalid")],
                      loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False, fontsize=8)
    save(fig, "e3_orbit_correctness", out, plt, outputs)
    path = out / "e3_orbit_row_order.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["recipe_index", "recipe_id", "qid"])
        writer.writerows((index + 1, recipe, qid) for index, (recipe, qid) in enumerate(questions))
    outputs.append(str(path.resolve()))


def plot_position(groups, out, plt, outputs):
    raw, centered = position_data(groups)
    bound = max(abs(value) for matrix in centered for row in matrix for value in row)
    bound = max(bound, .01)  # Shared nondegenerate scale when every value is zero.
    fig, axes = plt.subplots(2, len(groups), figsize=(3.65 * len(groups) + 1, 6.8), squeeze=False, layout="constrained")
    for column, item in enumerate(groups):
        for row, matrices in enumerate((raw, centered)):
            ax = axes[row][column]
            image = draw_grid(ax, matrices[column], cmap="Blues" if row == 0 else "RdBu", vmin=0 if row == 0 else -bound, vmax=1 if row == 0 else bound)
            annotate_grid(ax, matrices[column], centered=row == 1, color_bound=bound if row == 1 else 1)
            ax.set_xticks([.5, 1.5, 2.5, 3.5], list("ABCD"))
            ax.set_yticks([.5, 1.5, 2.5, 3.5], [1, 2, 3, 4])
            ax.set_xlabel("Displayed input position")
            ax.set_ylabel("Selected target rank" if column == 0 else "")
            if row == 0:
                ax.set_title(item["label"])
            if column == len(groups) - 1:
                colorbar = fig.colorbar(image, ax=list(axes[row]), fraction=.03, pad=.025)
                colorbar.set_label("Recipe-macro accuracy" if row == 0 else "Accuracy minus row mean")
                colorbar.solids.set_rasterized(False)
    fig.suptitle("Target rank and input position: raw accuracy (top), row centered (bottom)", fontsize=12)
    save(fig, "e3_rank_position", out, plt, outputs)


def plot_bank(groups, out, plt, outputs):
    names = ("both_vary", "pair_only_varies", "neither_varies")
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    width = .8 / len(groups)
    for index, item in enumerate(groups):
        bank = item["data"]["candidate_bank"]
        n = bank.get("n_groups")
        categories = bank.get("categories", {})
        if not isinstance(n, int) or n <= 0 or any(type(categories.get(name)) is not int or categories[name] < 0 for name in names) or sum(categories[name] for name in names) != n:
            raise ValueError(f"Invalid actual candidate-bank counts: {item['label']}")
        ax.bar([position - .4 + width * (index + .5) for position in range(3)], [categories[name] / n for name in names], width, label=item["label"])
    ax.set_xticks(range(3), ["Both rewards vary", "Only Pair varies", "Neither varies"])
    ax.set_ylabel("Fraction of identical candidate groups")
    ax.set_ylim(0, 1.02)
    ax.set_title("Same candidates, two verifiable rewards")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "e1_same_candidate_signal", out, plt, outputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", action="append", default=[], help="group_001=Name or parent_directory:group_001=Name")
    args = parser.parse_args(argv)
    labels = {}
    for assignment in args.label:
        if "=" not in assignment:
            parser.error("--label must use key=Name")
        key, name = assignment.split("=", 1)
        if not key or not name:
            parser.error("--label requires a nonempty key and label")
        labels[key] = name
    groups, inputs = load_groups(args.summary, labels)
    local = [item for item in groups if "scores" in item["data"] and "orbit" not in item["data"]]
    orbits = [item for item in groups if "orbit" in item["data"]]
    banks = [item for item in groups if "candidate_bank" in item["data"]]
    if not (local or orbits or banks):
        parser.error("No E0/E2, E3, or candidate-bank results found; no figures generated")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(out / ".mplconfig")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Matplotlib is required; install it in the project data-disk virtual environment before running make_figures.py") from error
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "text.usetex": False, "axes.titleweight": "medium"})
    outputs = []
    unavailable = []
    if local:
        plot_local(local, out, plt, outputs)
    else:
        unavailable.append({"figure": "E0/E2 full-test relations", "reason": "No full-test scores group supplied; orbit subset is not substituted for full test"})
    if orbits:
        # Matrix and position figures can each be absent independently; failure
        # is recorded, never replaced by fabricated cells/placeholder heatmaps.
        for name, callback in (("E3 correctness matrix", plot_orbit), ("E3 target-rank position", plot_position)):
            try:
                callback(orbits, out, plt, outputs)
            except (ValueError, KeyError, FileNotFoundError) as error:
                unavailable.append({"figure": name, "reason": str(error)})
    else:
        unavailable.append({"figure": "E3", "reason": "No complete-orbit summary supplied"})
    if banks:
        plot_bank(banks, out, plt, outputs)
    else:
        unavailable.append({"figure": "E1 same-candidate signal", "reason": "No candidate-bank summary supplied"})
    manifest = {"schema_version": 1, "inputs": inputs, "outputs": outputs, "unavailable": unavailable,
                "groups": [{"source": str(item["source_path"]), "group_id": item["group_id"], "label": item["label"], "identity": item["identity"]} for item in groups],
                "matplotlib_version": matplotlib.__version__,
                "notes": ["No synthetic/placeholder observations are generated.", "Pair heatmap: recipe macro; q distribution: question micro including invalids.",
                          "All compared orbit models use identical recipe/question row order and 24 lexicographically ordered image permutations.",
                          "Position colors share a 0..1 raw scale and a common symmetric row-centered scale; matrix cells are descriptive, not independent samples."]}
    (out / "figures_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "generated_files": len(outputs), "unavailable": unavailable}, ensure_ascii=False))
    return 0 if outputs else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise SystemExit(f"Cannot create figures from supplied results: {error}") from error
