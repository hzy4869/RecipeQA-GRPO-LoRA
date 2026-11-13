#!/usr/bin/env python3
"""Prepare preregistered input/parser sanity controls, with NO model execution.

Production example:
  python scripts/prepare_sanity_controls.py --root /root/autodl-tmp/rl_recipeqa

All files stay on the data disk. The resulting analytic-control JSON contains
mathematical/computed baselines, never invented model predictions or results.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import html
from itertools import permutations
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recipeqa.metrics import SCALAR_METRICS, aggregate_scores, score_order

DATA_DISK = Path("/root/autodl-tmp")
LABELS = "ABCD"


def confined(path: Path, root: Path) -> Path:
    path, root = path.expanduser().resolve(), root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Refusing path outside data root {root}: {path}")
    return path


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = confined(path.with_name(path.name + ".part"), path.parent)
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def blank_record(record: dict, blank_relative_path: str, blank_sha256: str, parent_sha256: str) -> dict:
    result = deepcopy(record)
    result.update({
        "manifest_set": "monitor20_blank", "control_type": "identical_white_images",
        "control_intent": "Remove all photograph information while preserving recipe text, fixed display mapping, IDs and targets; a drop alone does not establish grounding",
        "control_parent_manifest_sha256": parent_sha256,
        "control_origin_image_paths_sha256": digest({
            "display": record["image_paths_display"], "chronological": record.get("image_paths_chronological"),
        }),
        "control_blank_sha256": blank_sha256,
        "image_paths_display": [blank_relative_path] * 4,
        "image_paths_chronological": [blank_relative_path] * 4,
        "image_status_display": ["ok"] * 4,
        "image_status_chronological": ["ok"] * 4,
        "technical_status": "ok",
    })
    return result


def analytic_controls(rows: list[dict]) -> dict:
    """Deterministic fixed-layout copy baseline + exact uniform expectation."""
    if not rows:
        raise ValueError("Analytic controls require nonempty input")
    scored = [{
        "qid": row["qid"], "recipe_id": row["recipe_id"], "display_perm": row["display_perm"],
        "target_order": row.get("target_order", [0, 1, 2, 3]),
        "metrics": score_order("A B C D", row["display_perm"], row.get("target_order", [0, 1, 2, 3])),
    } for row in rows]
    summary = aggregate_scores(scored)
    # Enumerate the finite answer space once. Uniform displayed-label orders are
    # uniform semantic orders under every valid presentation/target permutation.
    finite = [score_order(order) for order in permutations(range(4))]
    expectation = {metric: str(sum(Fraction(row[metric]) for row in finite) / 24) for metric in SCALAR_METRICS}
    # Pair scores are floats in the shared scorer; use integer correct-pair
    # counts for an exact rational expression instead of Fraction(float).
    expectation["pair"] = str(sum(Fraction(row["pair_bin"], 6) for row in finite) / 24)
    return {
        "n_questions": len(rows), "n_recipes": len({row["recipe_id"] for row in rows}),
        "copy_display_order": {
            "kind": "computed_deterministic_parser_baseline_NOT_model_inference", "response": "A B C D",
            "micro": summary["micro"], "recipe_macro": summary["recipe_macro"],
            "pair_bins": summary["pair_bins"], "invalid": summary["invalid"],
        },
        "uniform_random_legal_permutation": {
            "kind": "exact_analytic_expectation_NOT_sampled_predictions_NOT_model_results",
            "assumptions": "For each question choose one of all 24 valid displayed-label permutations uniformly, independently of pixels, text, IDs and target",
            "n_permutations": 24,
            "expected_micro_and_recipe_macro_fractions": expectation,
            "expected_micro_and_recipe_macro": {metric: float(Fraction(value)) for metric, value in expectation.items()},
            "pair_bin_probability_fractions": {str(k): str(Fraction(sum(row["pair_bin"] == k for row in finite), 24)) for k in range(7)},
            "warning": "These are expected values, not a realized random run or measured model accuracy; finite-sample scores may vary",
        },
    }


def alternate_layout(record: dict) -> dict:
    """Reverse displayed positions, preserving image identities and target task."""
    result = deepcopy(record)
    original_perm = list(record["display_perm"])
    new_perm = original_perm[::-1]
    by_semantic = {semantic: path for semantic, path in zip(original_perm, record["image_paths_display"])}
    target = record.get("target_order", [0, 1, 2, 3])
    result.update({
        "manifest_set": "monitor20_hand_check_two_layouts",
        "presentation_id": f"{record['qid']}::p{''.join(map(str, new_perm))}",
        "display_perm": new_perm,
        "target_labels": [LABELS[new_perm.index(semantic)] for semantic in target],
        "image_paths_display": [by_semantic[semantic] for semantic in new_perm],
    })
    if "image_status_display" in record:
        result["image_status_display"] = record["image_status_display"][::-1]
    return result


def make_hand_check_packet(rows: list[dict], image_root: Path, output_dir: Path, allowed_root: Path) -> dict:
    from PIL import Image

    entries, private, manifest_rows = [], [], []
    assets = confined(output_dir / "two_layout" / "assets", allowed_root)
    assets.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(rows, 1):
        anonymous_id = f"Q{index:02d}"
        layouts = [deepcopy(record), alternate_layout(record)]
        panels = []
        private_layouts = []
        for layout_index, layout in enumerate(layouts, 1):
            layout["manifest_set"] = "monitor20_hand_check_two_layouts"
            manifest_rows.append(layout)
            figures = []
            for label, relative in zip(LABELS, layout["image_paths_display"]):
                source = confined(image_root / relative, image_root)
                asset_name = f"{anonymous_id.lower()}_l{layout_index}_{label.lower()}.jpg"
                target = confined(assets / asset_name, allowed_root)
                temporary = confined(target.with_name(target.name + ".part"), allowed_root)
                with Image.open(source) as opened:
                    image = opened.convert("RGB")
                    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    image.save(temporary, format="JPEG", quality=88)
                temporary.replace(target)
                figures.append(f'<figure><figcaption>{label}</figcaption><img src="assets/{asset_name}" alt="Image {label}"></figure>')
            panels.append(f'<div><h3>布局 {layout_index}</h3><div class="photos">{"".join(figures)}</div></div>')
            private_layouts.append({
                "layout": layout_index, "presentation_id": layout["presentation_id"],
                "display_perm": layout["display_perm"], "expected_target_labels": layout["target_labels"],
                "source_step_ids_by_display": [record["source_step_ids_chronological"][i] for i in layout["display_perm"]],
                "image_paths_display_sha256": digest(layout["image_paths_display"]),
            })
        recipe = "".join(f'<li><strong>{html.escape(str(step.get("title", "")))}</strong><p>{html.escape(str(step.get("body", "")))}</p></li>' for step in record["steps"])
        entries.append(f'<article><h2>{anonymous_id}</h2>{"".join(panels)}<details><summary>完整有序食谱</summary><ol>{recipe}</ol></details><p>手算布局1：________　布局2：________　核对人／日期：________</p></article>')
        private.append({"anonymous_id": anonymous_id, "qid": record["qid"], "recipe_id": record["recipe_id"], "layouts": private_layouts})
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>两布局人工映射核对</title><style>body{font-family:system-ui;max-width:1200px;margin:24px auto;padding:15px}article{border:1px solid #bbb;padding:16px;margin:22px 0}.photos{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}figure{margin:0}figcaption{font-weight:bold}img{max-width:100%;max-height:230px;object-fit:contain}li p{margin-top:4px}</style><h1>同题两布局人工映射核对</h1><p>无模型输出。先依据相同图片与食谱独立手算两布局标签，再由研究者用PRIVATE_reference.json核对。此页面不预填或生成人工标签。布局2只反转显示位置，语义顺序保持；文件名均为中性展示标签。</p>' + "".join(entries) + '</html>'
    write_text(confined(output_dir / "two_layout" / "hand_check.html", allowed_root), page)
    write_json(confined(output_dir / "two_layout" / "PRIVATE_reference.json", allowed_root), {
        "purpose": "PRIVATE source-grounded arithmetic reference, not human judgments or model results",
        "cases": private,
    })
    write_jsonl(confined(output_dir / "monitor20_two_layout.jsonl", allowed_root), manifest_rows)
    return {"questions": len(rows), "layouts_per_question": 2, "manifest_records": len(manifest_rows),
            "page": "two_layout/hand_check.html", "private_reference": "two_layout/PRIVATE_reference.json"}


def prepare_controls(root: Path, *, allowed_root: Path = DATA_DISK) -> dict:
    from PIL import Image

    root = confined(root, allowed_root)
    image_root = confined(root / "data" / "images", allowed_root)
    output_dir = confined(root / "data" / "controls", allowed_root)
    monitor_path = confined(root / "data" / "manifests" / "monitor.jsonl", allowed_root)
    test_path = confined(root / "data" / "manifests" / "test.jsonl", allowed_root)
    monitor_all, test = read_jsonl(monitor_path), read_jsonl(test_path)
    if len(monitor_all) < 20 or len(test) != 905:
        raise ValueError("Require at least 20 fixed monitor records and the complete 905-question test; no resampling/filtering")
    monitor = monitor_all[:20]
    if len({row["recipe_id"] for row in monitor}) != 20:
        raise ValueError("Monitor20 must contain 20 independent recipes")
    blank_path = confined(image_root / "_controls" / "blank.png", allowed_root)
    blank_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = confined(blank_path.with_name(blank_path.name + ".part"), allowed_root)
    Image.new("RGB", (256, 256), (255, 255, 255)).save(temporary, format="PNG")
    temporary.replace(blank_path)
    blank_sha = hashlib.sha256(blank_path.read_bytes()).hexdigest()
    monitor_sha = hashlib.sha256(monitor_path.read_bytes()).hexdigest()
    blank_rows = [blank_record(row, blank_path.relative_to(image_root).as_posix(), blank_sha, monitor_sha) for row in monitor]
    write_jsonl(confined(output_dir / "monitor20_blank.jsonl", allowed_root), blank_rows)
    analytic = {
        "scope": "Preregistered analytic/parser sanity controls; NO MODEL INFERENCE and NO HUMAN LABELS generated",
        "monitor_manifest_sha256": monitor_sha,
        "test_manifest_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "selection": "First 20 records in the frozen monitor manifest, in saved order; all 905 test records unchanged",
        "monitor20": analytic_controls(monitor), "test905": analytic_controls(test),
    }
    write_json(confined(output_dir / "analytic_controls.json", allowed_root), analytic)
    hand_check = make_hand_check_packet(monitor, image_root, output_dir, allowed_root)
    report = {
        "scope": "Preparation only: no GPU, model training, model prediction, human annotation or test-based selection",
        "blank_image": str(blank_path), "blank_image_sha256": blank_sha,
        "blank_manifest": str(output_dir / "monitor20_blank.jsonl"), "blank_questions": 20,
        "blank_preserved_fields": ["qid", "recipe_id", "presentation_id", "steps", "display_perm", "target_order", "target_labels"],
        "analytic_controls": str(output_dir / "analytic_controls.json"),
        "hand_check": hand_check,
        "interpretation": "Blank-image inference is run separately on a fixed selected W. A drop alone does not prove grounding; unexpectedly high performance calls for leakage and presentation checks, with uncertainty for n=20.",
    }
    write_json(confined(output_dir / "controls_report.json", allowed_root), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/rl_recipeqa"))
    args = parser.parse_args()
    print(json.dumps(prepare_controls(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
