#!/usr/bin/env python3
"""Compare two genuinely separate greedy prediction jobs on fixed inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recipeqa.metrics import score_order
from recipeqa.modeling import sha256_file, write_json


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compare(left, right, expected=10):
    if left.resolve() == right.resolve():
        raise ValueError("Two independent output files are required; resuming one file is not a repeat")
    fields = ("run_id", "model_id", "checkpoint", "seed", "recipe_id", "qid", "presentation_id",
              "display_perm", "target_order", "target_labels", "raw_response", "score", "input_audit",
              "model_weights_sha256", "prompt_kind", "sample_index", "sample_seed", "decoding",
              "input_tokens", "completion_tokens", "reached_token_limit", "last_token_id")
    differences = []
    groups = []
    configs = []
    for path in (left, right):
        records = rows(path)
        indexed = {}
        for record in records:
            key = (str(record["qid"]), str(record["presentation_id"]), record.get("sample_index", 0))
            if key in indexed or record.get("technical_error"):
                differences.append({"file": str(path), "input": key, "issue": "duplicate_or_technical_failure"})
            missing = [field for field in fields if field not in record]
            if missing:
                differences.append({"file": str(path), "input": key, "missing": missing})
            if record.get("decoding", {}).get("do_sample") is not False or record.get("prompt_kind") != "answer":
                differences.append({"file": str(path), "input": key, "issue": "not_answer_only_greedy"})
            if record.get("score") != score_order(record.get("raw_response", ""), record["display_perm"], record.get("target_order", [0, 1, 2, 3])):
                differences.append({"file": str(path), "input": key, "issue": "score_or_parsed_order_not_reproducible"})
            indexed[key] = record
        if len(records) != expected or len(indexed) != expected:
            differences.append({"file": str(path), "issue": "wrong_count", "records": len(records), "unique": len(indexed), "expected": expected})
        groups.append(indexed)
        config = json.loads(path.with_suffix(path.suffix + ".config.json").read_text())
        config["cli"] = {key: value for key, value in config["cli"].items() if key != "output"}
        configs.append(config)
    if not configs[0].get("weight_identity") or configs[0] != configs[1]:
        differences.append({"issue": "configuration_or_actual_weight_identity_differs"})
    if groups[0].keys() != groups[1].keys():
        differences.append({"issue": "input_coverage_differs"})
    for key in sorted(groups[0].keys() & groups[1].keys()):
        changed = [field for field in fields if groups[0][key].get(field) != groups[1][key].get(field)]
        if changed:
            differences.append({"input": key, "different_fields": changed})
    return {"status": "passed" if not differences else "different", "expected_inputs": expected,
            "files": {str(path): sha256_file(path) for path in (left, right)}, "compared_fields": fields,
            "differences": differences, "interpretation": "Fixed development greedy repeat only; not proof of all-input or cross-training bitwise determinism. A difference pauses E3; backend is never changed here."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--expected-inputs", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compare(args.left, args.right, args.expected_inputs)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {"status": "missing_or_invalid_evidence", "error": f"{type(error).__name__}: {error}"}
    write_json(args.output, result)
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
