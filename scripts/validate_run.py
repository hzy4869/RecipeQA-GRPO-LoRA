#!/usr/bin/env python3
"""Fail-closed coverage and training-artifact gates; never infer completeness from scores.

Examples:
  python scripts/validate_run.py predictions --manifest manifests/test.jsonl \
    --predictions runs/base/test.jsonl --run-id base --checkpoint /absolute/snapshot \
    --expected-questions 905 --expected-recipes 597 --output results/base_coverage.json
  python scripts/validate_run.py predictions --manifest manifests/orbit.jsonl \
    --predictions runs/pair/orbit.jsonl --require-orbit --expected-questions 60 \
    --output results/orbit_coverage.json
  python scripts/validate_run.py training --run-dir runs/pilot_pair --min-steps 20 \
    --output runs/pilot_pair/artifact_validation.json

Exit code 0 means the requested gate passed; 1 means evidence is missing or
contradictory. A training-artifact gate is not an actual interrupted-process
recovery test, a learning/generalization result, or whole-project completion.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import permutations
import json
import math
from pathlib import Path
import re
import sys
from typing import Any
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recipeqa.metrics import score_order
from recipeqa.modeling import ensure_data_disk, hash_json, model_weight_identity, sha256_file, write_json


ALL_LAYOUTS = set(permutations(range(4)))


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: record must be a JSON object")
            rows.append(row)
    return rows


def permutation(value) -> tuple[int, ...] | None:
    if isinstance(value, (list, tuple)) and len(value) == 4:
        if all(type(item) is int for item in value) and set(value) == set(range(4)):
            return tuple(value)
    return None


def _issue(issues: list, code: str, **details) -> None:
    issues.append({"code": code, **details})


def _sample_key(qid, presentation_id, sample_index) -> tuple:
    return str(qid), str(presentation_id), sample_index


def _key_payload(key: tuple) -> dict:
    return {"qid": key[0], "presentation_id": key[1], "sample_index": key[2]}


def validate_prediction_coverage(
    manifest_rows: list[dict], prediction_rows: list[dict], *, samples_per_input: int = 1,
    prompt_kind: str = "answer", expected_run_id: str | None = None,
    expected_checkpoint: str | None = None, expected_seed: int | None = None,
    expected_questions: int | None = None, expected_recipes: int | None = None,
    require_orbit: bool = False,
) -> dict:
    """Require every declared input/sample in every observed model identity group.

    Manifest IDs alone are insufficient: recipe, actual display mapping, target
    order, and target labels are cross-checked. Technical failures do not enter
    model-invalid counts and do not satisfy successful coverage.
    """
    issues = []
    warnings = []
    expected = {}
    input_rows = {}
    recipe_ids, question_keys = set(), set()
    if samples_per_input < 1:
        _issue(issues, "invalid_samples_per_input", value=samples_per_input)
    for number, row in enumerate(manifest_rows, 1):
        if not all(row.get(field) is not None and str(row.get(field)) for field in ("qid", "recipe_id", "presentation_id")):
            _issue(issues, "malformed_manifest_identity", row=number)
            continue
        display = permutation(row.get("display_perm"))
        target = permutation(row.get("target_order", [0, 1, 2, 3]))
        if display is None or target is None:
            _issue(issues, "malformed_manifest_permutation", row=number)
            continue
        target_labels = [chr(ord("A") + display.index(semantic)) for semantic in target]
        if row.get("target_labels") != target_labels:
            _issue(issues, "manifest_target_labels_mismatch", row=number, qid=row["qid"])
        input_key = (str(row["qid"]), str(row["presentation_id"]))
        if input_key in input_rows:
            _issue(issues, "duplicate_manifest_input", qid=input_key[0], presentation_id=input_key[1])
            continue
        input_rows[input_key] = row
        recipe_ids.add(str(row["recipe_id"]))
        question_keys.add((str(row["recipe_id"]), str(row["qid"])))
        for index in range(max(samples_per_input, 0)):
            expected[_sample_key(*input_key, index)] = row
    if not expected:
        _issue(issues, "empty_expected_coverage")
    if expected_questions is not None and len(question_keys) != expected_questions:
        _issue(issues, "manifest_question_count_mismatch", expected=expected_questions, actual=len(question_keys))
    if expected_recipes is not None and len(recipe_ids) != expected_recipes:
        _issue(issues, "manifest_recipe_count_mismatch", expected=expected_recipes, actual=len(recipe_ids))
    if require_orbit:
        by_question = defaultdict(list)
        for row in input_rows.values():
            by_question[str(row["recipe_id"]), str(row["qid"])].append(row)
        for (recipe, qid), rows in by_question.items():
            layouts = {tuple(row["display_perm"]) for row in rows}
            if len(rows) != 24 or layouts != ALL_LAYOUTS:
                _issue(issues, "manifest_incomplete_orbit", recipe_id=recipe, qid=qid,
                       n_rows=len(rows), n_unique_layouts=len(layouts))
            if len({tuple(row.get("target_order", [0, 1, 2, 3])) for row in rows}) != 1:
                _issue(issues, "manifest_orbit_target_changed", recipe_id=recipe, qid=qid)

    grouped = defaultdict(list)
    for number, row in enumerate(prediction_rows, 1):
        if not all(isinstance(row.get(field), str) and row[field] for field in ("run_id", "checkpoint")):
            _issue(issues, "malformed_prediction_group", row=number)
            continue
        seed = row.get("seed")
        if type(seed) is not int:
            _issue(issues, "missing_or_invalid_prediction_seed", row=number)
        group = (row["run_id"], row["checkpoint"], seed)
        grouped[group].append((number, row))
        if expected_run_id is not None and row["run_id"] != expected_run_id:
            _issue(issues, "unexpected_run_id", row=number, expected=expected_run_id, actual=row["run_id"])
        if expected_checkpoint is not None and row["checkpoint"] != expected_checkpoint:
            _issue(issues, "unexpected_checkpoint", row=number, expected=expected_checkpoint, actual=row["checkpoint"])
        if expected_seed is not None and seed != expected_seed:
            _issue(issues, "unexpected_seed", row=number, expected=expected_seed, actual=seed)
    if not grouped:
        _issue(issues, "no_prediction_groups")
    if expected_run_id is None or expected_checkpoint is None:
        warnings.append("Model groups were partly inferred from records. An entire absent model cannot be detected unless its run/checkpoint is declared in a separate invocation.")

    reports = []
    for (run_id, checkpoint, seed), numbered_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        group_issues = []
        seen = Counter()
        successful = set()
        failed = set()
        n_invalid = n_success = n_technical = 0
        bank_ids = defaultdict(set)
        for number, row in numbered_rows:
            if not all(row.get(field) is not None and str(row.get(field)) for field in ("qid", "presentation_id", "recipe_id")):
                _issue(group_issues, "malformed_prediction_identity", row=number)
                continue
            index = row.get("sample_index", 0)
            if type(index) is not int or index < 0:
                _issue(group_issues, "invalid_sample_index", row=number, value=index)
                continue
            key = _sample_key(row["qid"], row["presentation_id"], index)
            seen[key] += 1
            reference = expected.get(key)
            if reference is None:
                _issue(group_issues, "unexpected_prediction_key", row=number, **_key_payload(key))
            else:
                fields = ("recipe_id", "display_perm", "target_labels")
                for field in fields:
                    expected_value = reference[field]
                    actual_value = row.get(field)
                    if field == "recipe_id":
                        expected_value, actual_value = str(expected_value), str(actual_value)
                    if actual_value != expected_value:
                        _issue(group_issues, "prediction_manifest_mismatch", row=number, field=field,
                               expected=expected_value, actual=actual_value, **_key_payload(key))
                if row.get("target_order", [0, 1, 2, 3]) != reference.get("target_order", [0, 1, 2, 3]):
                    _issue(group_issues, "prediction_manifest_mismatch", row=number, field="target_order", **_key_payload(key))
            if row.get("prompt_kind", "answer") != prompt_kind:
                _issue(group_issues, "unexpected_prompt_kind", row=number, expected=prompt_kind,
                       actual=row.get("prompt_kind", "answer"))
            if samples_per_input > 1:
                bank_id = row.get("candidate_group_id")
                if not isinstance(bank_id, str) or not bank_id:
                    _issue(group_issues, "missing_candidate_group_id", row=number)
                else:
                    bank_ids[key[:2]].add(bank_id)
            technical = bool(row.get("technical_error")) or row.get("status") == "technical_error"
            if technical:
                n_technical += 1
                failed.add(key)
                continue
            if not isinstance(row.get("raw_response"), str):
                _issue(group_issues, "missing_or_nonstring_raw_response", row=number)
                continue
            if permutation(row.get("display_perm")) is None or permutation(row.get("target_order", [0, 1, 2, 3])) is None:
                _issue(group_issues, "malformed_prediction_permutation", row=number)
                continue
            n_success += 1
            successful.add(key)
            if prompt_kind == "answer":
                score = score_order(row["raw_response"], row["display_perm"], row.get("target_order", [0, 1, 2, 3]))
                n_invalid += 1 - score["valid"]
        duplicates = [key for key, count in seen.items() if count > 1]
        for key in sorted(duplicates):
            _issue(group_issues, "duplicate_prediction_key", count=seen[key], **_key_payload(key))
        missing = sorted(expected.keys() - successful)
        if missing:
            _issue(group_issues, "missing_successful_predictions", count=len(missing))
        if n_technical:
            _issue(group_issues, "unresolved_technical_failures", count=n_technical)
        for input_key, ids in bank_ids.items():
            if len(ids) != 1:
                _issue(group_issues, "candidate_input_split_across_groups", qid=input_key[0], presentation_id=input_key[1])
        reverse_bank = defaultdict(set)
        for input_key, ids in bank_ids.items():
            for bank_id in ids:
                reverse_bank[bank_id].add(input_key)
        for bank_id, inputs in reverse_bank.items():
            if len(inputs) != 1:
                _issue(group_issues, "candidate_group_mixes_inputs", candidate_group_id=bank_id)
        reports.append({
            "identity": {"run_id": run_id, "checkpoint": checkpoint, "seed": seed},
            "passed": not group_issues, "expected_predictions": len(expected), "n_rows": len(numbered_rows),
            "n_unique_recorded_keys": len(seen), "n_unique_successful_keys": len(successful),
            "n_successful_model_responses": n_success, "n_technical_failure_rows": n_technical,
            "n_invalid_model_responses": n_invalid if prompt_kind == "answer" else None,
            "missing_successful_keys": [_key_payload(key) for key in missing],
            "unresolved_technical_keys": [_key_payload(key) for key in sorted(failed)],
            "unexpected_keys": [_key_payload(key) for key in sorted(seen.keys() - expected.keys())],
            "duplicate_keys": [_key_payload(key) for key in sorted(duplicates)], "issues": group_issues,
        })
    return {
        "schema_version": 1, "gate": "prediction_manifest_coverage", "passed": not issues and bool(reports) and all(group["passed"] for group in reports),
        "manifest": {"rows": len(manifest_rows), "unique_inputs": len(input_rows), "questions": len(question_keys),
                     "recipes": len(recipe_ids), "samples_per_input": samples_per_input, "expected_predictions_per_group": len(expected),
                     "complete_24_layouts_required": require_orbit},
        "prompt_kind": prompt_kind, "groups": reports, "issues": issues, "warnings": warnings,
        "denominator_rule": "Every expected input/sample must have exactly one successful inference record. Invalid model text is scored; technical failures are excluded from invalid counts and fail coverage.",
    }


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _valid_hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def validate_training_artifacts(run_dir: str | Path, *, min_steps: int = 1,
                                min_nonzero_gradient_steps: int = 1) -> dict:
    root = Path(run_dir)
    issues, warnings, checks = [], [], {}
    loaded = {}
    for name in ("resolved_config.json", "train_outcome.json", "trainable_parameters.json", "run_signature.json"):
        path = root / name
        if not _nonempty(path):
            _issue(issues, "missing_required_artifact", path=name)
            continue
        try:
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded[name], dict):
                raise ValueError("expected JSON object")
        except (ValueError, TypeError) as error:
            _issue(issues, "malformed_required_artifact", path=name, reason=str(error))
    config = loaded.get("resolved_config.json", {})
    outcome = loaded.get("train_outcome.json", {})
    inventory = loaded.get("trainable_parameters.json", {})
    signature = loaded.get("run_signature.json", {})
    if not _valid_hash(signature.get("sha256")) or signature.get("sha256") != hash_json(signature.get("payload")):
        _issue(issues, "missing_or_inconsistent_run_signature")
    for name, payload in (("resolved_config", config), ("train_outcome", outcome)):
        if payload.get("run_signature_sha256") != signature.get("sha256") or not signature.get("sha256"):
            _issue(issues, "artifact_run_signature_mismatch", artifact=name)
    method = config.get("method")
    if method not in ("sft", "grpo_exact", "grpo_pair"):
        _issue(issues, "unknown_or_missing_training_method", value=method)
    if outcome.get("status") != "completed":
        _issue(issues, "training_not_completed", status=outcome.get("status"))
    steps = outcome.get("global_step")
    if type(steps) is not int or steps < min_steps:
        _issue(issues, "insufficient_optimizer_steps", actual=steps, minimum=min_steps)
    initial_hash = outcome.get("initial_adapter_fingerprint")
    final_hash = outcome.get("final_adapter_fingerprint")
    changed = _valid_hash(initial_hash) and _valid_hash(final_hash) and initial_hash != final_hash
    checks["adapter_weights_changed"] = changed
    if not changed:
        _issue(issues, "no_verified_adapter_weight_change")
    if initial_hash != config.get("initial_adapter_fingerprint"):
        _issue(issues, "initial_adapter_hash_disagrees_with_resolved_config")
    if outcome.get("effective_adapter_update") is False:
        _issue(issues, "runtime_reported_ineffective_update")
    checks["runtime_visual_freeze_guard_passed"] = outcome.get("visual_weights_unchanged") is True and _valid_hash(config.get("initial_visual_fingerprint"))
    if not checks["runtime_visual_freeze_guard_passed"]:
        _issue(issues, "missing_visual_freeze_guard_evidence")
    if not _valid_hash(outcome.get("initial_visual_fingerprint")) or not _valid_hash(outcome.get("final_visual_fingerprint")):
        _issue(issues, "missing_exported_visual_weight_fingerprints")
    elif outcome["final_visual_fingerprint"] != config.get("initial_visual_fingerprint") or outcome["initial_visual_fingerprint"] != config.get("initial_visual_fingerprint"):
        _issue(issues, "visual_weight_fingerprint_changed")
    trainer = config.get("trainer", {})
    common = {"shuffle_dataset": False, "train_sampling_strategy": "sequential", "num_train_epochs": 1,
              "per_device_train_batch_size": 1, "save_only_model": False}
    specific = ({"completion_only_loss": True, "packing": False, "max_length": None} if method == "sft" else
                {"beta": 0.0, "loss_type": "grpo", "scale_rewards": "group", "num_iterations": 1,
                 "use_vllm": False, "mask_truncated_completions": False})
    for key, expected_value in {**common, **specific}.items():
        if key not in trainer or trainer[key] != expected_value:
            _issue(issues, "resolved_training_condition_mismatch", field=key, expected=expected_value, actual=trainer.get(key))
    for name in ("python", "torch", "transformers", "trl", "peft", "accelerate", "datasets"):
        if not isinstance(config.get("versions", {}).get(name), str):
            _issue(issues, "missing_package_version", package=name)
    if not _valid_hash(config.get("manifest_sha256")):
        _issue(issues, "missing_manifest_fingerprint")
    parameters = inventory.get("parameters", [])
    if not isinstance(parameters, list) or not parameters:
        _issue(issues, "missing_trainable_parameter_inventory")
        parameters = []
    for parameter in parameters:
        name = parameter.get("name", "") if isinstance(parameter, dict) else ""
        if "lora_" not in name or ".language_model.layers." not in name:
            _issue(issues, "unapproved_trainable_parameter", name=name)
    if type(inventory.get("trainable_parameters")) is not int or inventory.get("trainable_parameters", 0) <= 0:
        _issue(issues, "no_trainable_parameters")

    gradient_rows = []
    gradient_path = root / "gradient_audit.jsonl"
    if not _nonempty(gradient_path):
        _issue(issues, "missing_required_artifact", path="gradient_audit.jsonl")
    else:
        try:
            gradient_rows = load_jsonl(gradient_path)
        except ValueError as error:
            _issue(issues, "malformed_gradient_audit", reason=str(error))
    observed_steps = set()
    gradient_step_counts = Counter()
    nonzero_steps = set()
    for number, row in enumerate(gradient_rows, 1):
        step = row.get("global_step_before_update")
        norm = row.get("l2_norm")
        tensors = row.get("trainable_gradient_tensors")
        nonzero = row.get("nonzero_gradient_tensors")
        if type(step) is not int or step < 0 or (type(steps) is int and step >= steps):
            _issue(issues, "invalid_gradient_step", row=number, value=step)
            continue
        observed_steps.add(step)
        gradient_step_counts[step] += 1
        if not isinstance(norm, (float, int)) or not math.isfinite(norm) or norm < 0:
            _issue(issues, "invalid_gradient_norm", row=number, value=norm)
        if type(tensors) is not int or tensors <= 0 or type(nonzero) is not int or not 0 <= nonzero <= tensors:
            _issue(issues, "invalid_gradient_tensor_counts", row=number)
        elif isinstance(norm, (int, float)) and math.isfinite(norm) and norm > 0 and nonzero > 0:
            nonzero_steps.add(step)
    if type(steps) is int and steps > 0 and observed_steps != set(range(steps)):
        _issue(issues, "incomplete_gradient_step_coverage", expected_steps=steps,
               missing_steps=sorted(set(range(steps)) - observed_steps))
    if len(nonzero_steps) < min_nonzero_gradient_steps:
        _issue(issues, "insufficient_nonzero_gradient_steps", actual=len(nonzero_steps), minimum=min_nonzero_gradient_steps)
    checks["gradient_steps_recorded"] = len(observed_steps)
    checks["nonzero_gradient_steps"] = len(nonzero_steps)
    if any(count != 1 for count in gradient_step_counts.values()):
        _issue(issues, "duplicate_canonical_gradient_steps", steps=sorted(step for step, count in gradient_step_counts.items() if count > 1))

    final_adapter = root / "adapter_final"
    if not _nonempty(final_adapter / "adapter_config.json"):
        _issue(issues, "missing_final_adapter_config")
    if not any(_nonempty(final_adapter / name) for name in ("adapter_model.safetensors", "adapter_model.bin")):
        _issue(issues, "missing_final_adapter_weights")
    checkpoints = []
    for checkpoint in sorted(root.glob("checkpoint-*")):
        if not checkpoint.is_dir():
            continue
        result = {"path": str(checkpoint), "issues": []}
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_config.json")
        for name in required:
            path = checkpoint / name
            if not _nonempty(path):
                _issue(result["issues"], "missing_checkpoint_state", path=name)
        if not any(_nonempty(checkpoint / name) for name in ("adapter_model.safetensors", "adapter_model.bin")):
            _issue(result["issues"], "missing_checkpoint_adapter_weights")
        state_path = checkpoint / "trainer_state.json"
        if _nonempty(state_path):
            try:
                state = json.loads(state_path.read_text())
                checkpoint_step = int(checkpoint.name.split("-")[-1])
                result["global_step"] = state.get("global_step")
                if state.get("global_step") != checkpoint_step:
                    _issue(result["issues"], "checkpoint_step_mismatch")
                if type(steps) is int and checkpoint_step > steps:
                    _issue(result["issues"], "checkpoint_newer_than_outcome")
            except (ValueError, AttributeError) as error:
                _issue(result["issues"], "malformed_checkpoint_state", reason=str(error))
        for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
            path = checkpoint / name
            if _nonempty(path):
                if not zipfile.is_zipfile(path):
                    _issue(result["issues"], "unexpected_checkpoint_serialization", path=name)
                else:
                    try:
                        with zipfile.ZipFile(path) as archive:
                            damaged = archive.testzip()
                            if damaged is not None:
                                _issue(result["issues"], "checkpoint_crc_error", path=name, member=damaged)
                            if not any(member.endswith("data.pkl") for member in archive.namelist()):
                                _issue(result["issues"], "checkpoint_missing_serialized_state", path=name)
                    except (OSError, zipfile.BadZipFile) as error:
                        _issue(result["issues"], "checkpoint_archive_error", path=name, reason=str(error))
        audit_path = checkpoint / "recipeqa_checkpoint_audit.json"
        if not _nonempty(audit_path):
            _issue(result["issues"], "missing_checkpoint_identity_audit")
        else:
            try:
                audit = json.loads(audit_path.read_text())
                if audit.get("run_signature_sha256") != signature.get("sha256"):
                    _issue(result["issues"], "checkpoint_run_signature_mismatch")
                state_hashes = audit.get("state_sha256", {})
                state_names = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"}
                if set(state_hashes) != state_names:
                    _issue(result["issues"], "missing_checkpoint_state_content_hashes")
                for name in state_names:
                    if _nonempty(checkpoint / name) and state_hashes.get(name) != sha256_file(checkpoint / name):
                        _issue(result["issues"], "checkpoint_state_content_hash_mismatch", path=name)
                if audit.get("adapter_identity") != model_weight_identity(checkpoint, adapter=True):
                    _issue(result["issues"], "checkpoint_adapter_content_hash_mismatch")
            except (OSError, ValueError, TypeError) as error:
                _issue(result["issues"], "malformed_checkpoint_identity_audit", reason=str(error))
        result["passed"] = not result["issues"]
        checkpoints.append(result)
        if not result["passed"]:
            _issue(issues, "checkpoint_validation_failed", checkpoint=checkpoint.name)
    if not checkpoints:
        _issue(issues, "no_full_checkpoint_available")
    warnings.append("This artifact gate does not execute Trainer resume or prove optimizer/scheduler/RNG/sample-position restoration. An independent exit/restart pilot is still required.")
    return {"schema_version": 1, "gate": "training_artifacts_only", "passed": not issues,
            "run_dir": str(root.resolve()), "method": method, "optimizer_steps": steps,
            "checks": checks, "checkpoints": checkpoints, "issues": issues, "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    predictions = subparsers.add_parser("predictions", help="Require exact expected-manifest coverage per run/checkpoint/seed")
    predictions.add_argument("--manifest", type=Path, required=True)
    predictions.add_argument("--predictions", type=Path, nargs="+", required=True)
    predictions.add_argument("--samples-per-input", type=int, default=1)
    predictions.add_argument("--prompt-kind", choices=("answer", "process"), default="answer")
    predictions.add_argument("--run-id")
    predictions.add_argument("--checkpoint")
    predictions.add_argument("--seed", type=int)
    predictions.add_argument("--expected-questions", type=int)
    predictions.add_argument("--expected-recipes", type=int)
    predictions.add_argument("--require-orbit", action="store_true")
    predictions.add_argument("--output", type=Path, required=True)
    training = subparsers.add_parser("training", help="Validate completed-run artifacts; not a functional recovery test")
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--min-steps", type=int, default=1)
    training.add_argument("--min-nonzero-gradient-steps", type=int, default=1)
    training.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        ensure_data_disk(args.output)
        if args.command == "predictions":
            manifest = load_jsonl(args.manifest)
            rows = [row for path in args.predictions for row in load_jsonl(path)]
            report = validate_prediction_coverage(
                manifest, rows, samples_per_input=args.samples_per_input, prompt_kind=args.prompt_kind,
                expected_run_id=args.run_id, expected_checkpoint=args.checkpoint, expected_seed=args.seed,
                expected_questions=args.expected_questions, expected_recipes=args.expected_recipes,
                require_orbit=args.require_orbit,
            )
            report["source_files"] = {"manifest": str(args.manifest.resolve()),
                                      "predictions": [str(path.resolve()) for path in args.predictions]}
        else:
            report = validate_training_artifacts(args.run_dir, min_steps=args.min_steps,
                                                  min_nonzero_gradient_steps=args.min_nonzero_gradient_steps)
    except (OSError, ValueError, TypeError, KeyError) as error:
        report = {"schema_version": 1, "gate": args.command, "passed": False,
                  "issues": [{"code": "validation_input_error", "reason": f"{type(error).__name__}: {error}"}]}
    write_json(args.output, report)
    print(json.dumps({"passed": report["passed"], "gate": report["gate"], "report": str(args.output.resolve()),
                      "n_issues": len(report.get("issues", [])) + sum(len(group.get("issues", [])) for group in report.get("groups", []))}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
