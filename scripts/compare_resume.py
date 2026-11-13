#!/usr/bin/env python3
"""Read-only CPU comparison of a completed run and its latest resumed trajectory.

The comparison reads trusted checkpoints produced by this project. It never
loads a GPU model, edits checkpoints, or substitutes file ZIP hashes for tensors.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare_values(left, right, torch, np, atol, rtol):
    result = {"status": "exact_equal", "max_abs_difference": None, "compared_leaves": 0, "differences": []}
    priority = {"exact_equal": 0, "numerical_close": 1, "different": 2}

    def difference(status, path, detail, absolute=None):
        if priority[status] > priority[result["status"]]:
            result["status"] = status
        if absolute is not None:
            result["max_abs_difference"] = max(result["max_abs_difference"] or 0.0, absolute)
        if len(result["differences"]) < 20:
            result["differences"].append({"path": path, "status": status, "detail": detail, "max_abs_difference": absolute})

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                difference("different", path, {"only_old": [str(k) for k in a.keys() - b.keys()], "only_new": [str(k) for k in b.keys() - a.keys()]})
            for key in a.keys() & b.keys():
                walk(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                difference("different", path, "sequence type/length differs")
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}/{index}")
        elif torch.is_tensor(a) or torch.is_tensor(b) or isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            result["compared_leaves"] += 1
            a, b = torch.as_tensor(a), torch.as_tensor(b)
            if a.shape != b.shape or a.dtype != b.dtype:
                difference("different", path, f"shape/dtype: {a.shape}/{a.dtype} versus {b.shape}/{b.dtype}")
            elif not torch.equal(a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8)):
                delta = float((a.to(torch.float64) - b.to(torch.float64)).abs().max()) if a.numel() else 0.0
                finite = math.isfinite(delta)
                close = finite and torch.is_floating_point(a) and torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
                difference("numerical_close" if close else "different", path, "tensor contents differ", delta if finite else None)
        else:
            result["compared_leaves"] += 1
            if type(a) is not type(b) or a != b:
                numeric = isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number))
                delta = abs(float(a) - float(b)) if numeric else None
                close = numeric and isinstance(a, (float, np.floating)) and math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)
                difference("numerical_close" if close else "different", path, {"old": str(a), "new": str(b)}, delta if delta is not None and math.isfinite(delta) else None)
    walk(left, right, "root")
    if result["status"] == "exact_equal":
        result["max_abs_difference"] = 0.0
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    from safetensors.torch import load_file

    root = args.run_dir.resolve()
    report = {"status": "pending", "run_dir": str(root), "atol": args.atol, "rtol": args.rtol,
              "sections": {}, "missing_or_unreadable": [],
              "equality_definition": "Tensor exact_equal checks matching dtype/shape and raw element bytes after CPU loading. It does not claim equal ZIP serialization bytes. Numerical_close uses the stated tolerances.",
              "scope": "Recovery reproducibility only; numerical differences are reported, not an RL efficacy or automatic scientific failure judgment."}
    attempts = []
    for path in (root / "attempts").glob("*"):
        if not (path / "superseded_before_resume").is_dir():
            continue
        try:
            metadata = load_json(path / "attempt.json")
            if type(metadata.get("resume_global_step")) is not int:
                raise ValueError("Missing resume_global_step")
            attempts.append((path, metadata))
        except (OSError, ValueError, TypeError) as error:
            report["missing_or_unreadable"].append(f"{path}: {error}")
    if not attempts:
        report["missing_or_unreadable"].append("No resumed attempt with superseded_before_resume archive")
    else:
        attempt, metadata = max(attempts, key=lambda pair: pair[1].get("started_utc", ""))
        old = attempt / "superseded_before_resume"
        report.update(attempt_id=attempt.name, resume_global_step=metadata["resume_global_step"])

        def section(name, loader):
            try:
                left, right = loader()
                report["sections"][name] = compare_values(left, right, torch, np, args.atol, args.rtol)
            except Exception as error:
                report["sections"][name] = {"status": "pending", "reason": f"{type(error).__name__}: {error}"}
                report["missing_or_unreadable"].append(f"{name}: {error}")

        try:
            before, after = load_json(old / "train_outcome.json"), load_json(root / "train_outcome.json")
            final_step = after["global_step"]
            if final_step != before["global_step"] or final_step <= metadata["resume_global_step"]:
                raise ValueError("Old/new final steps must match and exceed the resume step")
            def fingerprints():
                values = before["final_adapter_fingerprint"], after["final_adapter_fingerprint"]
                if any(not isinstance(value, str) or len(value) != 64 for value in values):
                    raise ValueError("Missing final adapter runtime fingerprint")
                return values
            section("final_adapter_runtime_fingerprint", fingerprints)

            def weights(path):
                safe = path / "adapter_model.safetensors"
                return load_file(str(safe), device="cpu") if safe.exists() else torch.load(path / "adapter_model.bin", map_location="cpu", weights_only=True)
            section("final_adapter_tensors", lambda: (weights(old / "adapter_final"), weights(root / "adapter_final")))
            for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                section(filename, lambda name=filename: (
                    torch.load(old / f"checkpoint-{final_step}" / name, map_location="cpu", weights_only=False),
                    torch.load(root / f"checkpoint-{final_step}" / name, map_location="cpu", weights_only=False)))

            def selected(filename, step_field, fields, inclusive=False):
                selections = []
                for base in (old, root):
                    rows = [row for row in load_rows(base / filename) if all(field in row for field in fields)
                            and (row[step_field] > metadata["resume_global_step"] if inclusive else row[step_field] >= metadata["resume_global_step"])
                            and row[step_field] <= (final_step if inclusive else final_step - 1)]
                    expected_steps = set(range(metadata["resume_global_step"] + int(inclusive), final_step + int(inclusive)))
                    if {row[step_field] for row in rows} != expected_steps:
                        raise ValueError(f"Missing resumed step coverage in {base / filename}")
                    selections.append([{field: row[field] for field in fields} for row in rows])
                return selections
            visit_file = "actual_prompt_visits.jsonl" if (old / "actual_prompt_visits.jsonl").exists() else "reward_groups.jsonl"
            visit_fields = ("global_step_before_update", "qid", "presentation_id", "visit_id", "display_perm") + (("input_hash",) if visit_file == "actual_prompt_visits.jsonl" else ())
            section("resumed_prompt_visits", lambda: selected(visit_file, "global_step_before_update", visit_fields))
            section("resumed_gradients", lambda: selected("gradient_audit.jsonl", "global_step_before_update", ("global_step_before_update", "l2_norm", "nonzero_gradient_tensors", "trainable_gradient_tensors")))
            section("resumed_learning_rates", lambda: selected("trainer_metrics.jsonl", "global_step", ("global_step", "learning_rate"), inclusive=True))
        except (OSError, ValueError, KeyError, TypeError) as error:
            report["missing_or_unreadable"].append(f"Run outcomes: {error}")
        statuses = [item["status"] for name, item in report["sections"].items() if name != "final_adapter_runtime_fingerprint"]
        if statuses and not report["missing_or_unreadable"]:
            report["status"] = "different" if "different" in statuses else "numerical_close" if "numerical_close" in statuses else "exact_equal"
            fingerprint_status = report["sections"]["final_adapter_runtime_fingerprint"]["status"]
            if report["status"] == "exact_equal" and fingerprint_status != "exact_equal":
                report["status"] = "different"
                report["fingerprint_inconsistency"] = "Saved tensors/states match but runtime fingerprints differ"
            report["all_sections_exact_equal"] = all(item["status"] == "exact_equal" for item in report["sections"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output.resolve()), "sections": {key: value["status"] for key, value in report["sections"].items()}}))
    return 1 if report["status"] == "pending" else 0


if __name__ == "__main__":
    sys.exit(main())
