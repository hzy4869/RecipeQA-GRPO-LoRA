#!/usr/bin/env python3
"""Read-only real-LoRA gradient geometry on a fixed TRAINING prefix.

No optimizer, backward accumulation, parameter assignment, update, generation,
or held-out selection occurs. All 24 canonical EOS-complete scores are computed
first. Detached coefficients then combine one candidate gradient at a time.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recipeqa.metrics import score_order
from recipeqa.modeling import (
    InputConfig, adapter_fingerprint, encode_messages, ensure_data_disk, hash_json,
    load_images, load_model, load_processor, model_weight_identity, package_versions,
    prepare_record, read_jsonl, sha256_file, trainable_inventory, write_json,
)
from score_permutations import CANDIDATES, candidate_suffixes, probability_summary, repeat_inputs, score_candidates

TRAIN_VISITS_SHA256 = "e502e427cdfb0306e9ee8e581ad86ae159f0a7d96de1049716d81bb961622e20"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def coefficient_table(q, exact, pair):
    """First derivatives with respect to unnormalized canonical log scores."""
    require(len(q) == len(exact) == len(pair) == 24, "Need complete 24-action distribution")
    require(abs(sum(q) - 1) < 1e-9 and all(0 <= p <= 1 for p in q), "Invalid closed policy")
    require(sum(exact) == 1 and set(exact) <= {0, 1}, "Need one verified exact target")
    j_exact = sum(p * r for p, r in zip(q, exact))
    j_pair = sum(p * r for p, r in zip(q, pair))
    return {
        "exact": [p * (r - j_exact) for p, r in zip(q, exact)],
        "pair": [p * (r - j_pair) for p, r in zip(q, pair)],
        "closed_ce_ascent": [r - p for r, p in zip(exact, q)],
        "raw_sft_sequence_ascent": list(exact),
        "log_canonical_mass": list(q),
    }


def validate_train(records, limit):
    require(len(records) == 2048, "Expected the original 2048-visit training manifest")
    require(1 <= limit <= 4, "Read-only gradient diagnostic is capped at four fixed training questions")
    selected = records[:limit]
    require(len({row["recipe_id"] for row in selected}) == limit, "First training visits are not distinct recipe clusters")
    for row in selected:
        require(row.get("source_split") == "train" and row.get("manifest_set") == "train_visits",
                "Gradient diagnosis must use the original training prefix, never development/test targets")
    return selected


def differentiable_logp(model, encoded, candidate_ids):
    """Same answer/EOS positions as scorer, with autograd enabled."""
    import torch

    require(torch.is_grad_enabled(), "Differentiable scoring was entered under no_grad/inference_mode")
    full = repeat_inputs(encoded, [candidate_ids])
    count = len(candidate_ids)
    prefix = encoded["input_ids"].shape[1]
    output = model(**full, use_cache=False, return_dict=True, logits_to_keep=count + 1)
    require(output.logits.shape[1] == count + 1, "Actual model ignored selected-logit argument")
    logits = output.logits[:, :-1, :]
    targets = full["input_ids"][:, prefix:]
    require(logits.shape[:2] == targets.shape, "Differentiable answer logit/target misalignment")
    values = logits.float().log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    scalar = values.sum()
    require(scalar.requires_grad, "Sequence score is disconnected from trainable LoRA")
    return scalar, values.detach().cpu().tolist()[0]


def parameter_fingerprint(model):
    """Hash every actual in-memory parameter, including frozen base weights."""
    import torch

    digest = hashlib.sha256()
    count = 0
    for name, parameter in model.named_parameters():
        digest.update(json.dumps([name, str(parameter.dtype), list(parameter.shape)]).encode())
        raw = parameter.detach().contiguous().view(-1).view(torch.uint8)
        for begin in range(0, raw.numel(), 8 * 1024 * 1024):
            digest.update(raw[begin:begin + 8 * 1024 * 1024].cpu().numpy().tobytes())
        count += parameter.numel()
    return {"sha256": digest.hexdigest(), "parameter_elements": count,
            "scope": "all actual in-memory named parameters, including frozen base/vision and trainable LoRA"}


def inner_product(a, b):
    """Float64 products/reduction in small chunks, no giant double copies."""
    import torch

    total = torch.zeros((), dtype=torch.float64, device=a.device)
    for begin in range(0, a.numel(), 1024 * 1024):
        total += torch.dot(a[begin:begin + 1024 * 1024].double(), b[begin:begin + 1024 * 1024].double())
    return float(total.item())


def vector_summary(buffers, q_gold, atol, rtol):
    import torch

    norms2 = {name: inner_product(vector, vector) for name, vector in buffers.items()}
    require(all(math.isfinite(value) for value in norms2.values()), "Non-finite parameter gradient")
    comparisons = {}
    for left, right in (("exact", "pair"), ("closed_ce_ascent", "raw_sft_sequence_ascent"),
                        ("pair", "raw_sft_sequence_ascent")):
        value = inner_product(buffers[left], buffers[right])
        denom = math.sqrt(norms2[left] * norms2[right])
        comparisons[f"{left}__{right}"] = {"dot": value, "cosine": value / denom if denom > 0 else None}
    identity_checks = {}
    for name, left, expected in (
        ("exact_equals_qgold_times_closed_ce", buffers["exact"], q_gold * buffers["closed_ce_ascent"]),
        ("closed_ce_equals_raw_sft_minus_log_canonical_mass", buffers["closed_ce_ascent"],
         buffers["raw_sft_sequence_ascent"] - buffers["log_canonical_mass"]),
    ):
        error = left - expected
        maximum_error = float(error.abs().max().item())
        maximum_expected = float(expected.abs().max().item())
        error_norm2 = inner_product(error, error)
        expected_norm2 = inner_product(expected, expected)
        relative_l2 = math.sqrt(error_norm2 / expected_norm2) if expected_norm2 > 0 else None
        maximum_gate = maximum_error <= atol + rtol * maximum_expected
        # An absolute floor alone can accept a very wrong direction when the
        # entire gradient is small. Require relative agreement independently.
        # When the expected vector is exactly zero, only an exactly zero error
        # is accepted; there is no meaningful relative norm to report.
        relative_gate = relative_l2 <= rtol if relative_l2 is not None else error_norm2 == 0
        passed = maximum_gate and relative_gate
        require(passed, f"Parameter-gradient identity failed: {name}; relative_l2={relative_l2}")
        identity_checks[name] = {"passed": passed, "max_absolute_error": maximum_error,
                                 "max_absolute_expected": maximum_expected, "relative_l2_error": relative_l2,
                                 "absolute_tolerance": atol, "relative_tolerance": rtol,
                                 "relative_l2_tolerance": rtol, "maximum_error_gate_passed": maximum_gate,
                                 "relative_l2_gate_passed": relative_gate}
    return {"gradient_norms": {name: math.sqrt(value) for name, value in norms2.items()},
            "comparisons": comparisons,
            "gold_probability_derivative_along_pair_parameter_gradient": comparisons["exact__pair"]["dot"],
            "identities": identity_checks,
            "coordinate_system": "actual shared trainable language-LoRA parameters, Euclidean raw gradient; no optimizer/preconditioner"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=1, choices=(1, 2, 3, 4))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logp-atol", type=float, default=.01)
    parser.add_argument("--identity-atol", type=float, default=1e-6)
    parser.add_argument("--identity-rtol", type=float, default=1e-4)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--recipe-token-budget", type=int, default=1536)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    return parser.parse_args()


def main():
    args = arguments()
    ensure_data_disk(args.model_path, args.adapter_path, args.manifest, args.image_root, args.output)
    require(sha256_file(args.manifest) == TRAIN_VISITS_SHA256, "Original fixed training manifest SHA differs")
    require(0 < args.logp_atol <= .1
            and math.isfinite(args.identity_atol) and args.identity_atol > 0
            and math.isfinite(args.identity_rtol) and args.identity_rtol > 0, "Invalid numerical tolerances")
    records = validate_train(read_jsonl(args.manifest), args.limit)
    output = Path(args.output)
    config_path = output.with_suffix(output.suffix + ".config.json")
    completion_path = output.with_suffix(output.suffix + ".completion.json")
    require(not output.exists() and not config_path.exists() and not completion_path.exists(),
            "Choose a fresh output; no partial diagnostic restart or weight overwrite is allowed")
    output.parent.mkdir(parents=True, exist_ok=True)
    import torch

    config = InputConfig(args.min_pixels, args.max_pixels, args.recipe_token_budget, args.max_input_tokens, 32)
    processor = load_processor(args.model_path, args.revision, config)
    model = load_model(args.model_path, args.revision, adapter_path=args.adapter_path, trainable=True, seed=args.seed)
    # The frozen helper intentionally leaves trainable models for Trainer placement.
    # This diagnostic has no Trainer, so placement/eval are explicit.
    model.to("cuda")
    model.eval()
    model.config.use_cache = False
    inventory = trainable_inventory(model)
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = tuple(parameter for _, parameter in named_parameters)
    require(len(parameters) == len(inventory["parameters"]), "Trainable LoRA inventory differs")
    require(all(parameter.grad is None for parameter in parameters), "Unexpected existing parameter .grad buffers")
    all_before = parameter_fingerprint(model)
    adapter_before = adapter_fingerprint(model)
    provenance = {
        "schema_version": 1, "run_id": args.run_id, "post_hoc_exploration": True,
        "kind": "read_only_training_prefix_parameter_gradient_diagnostic", "cli": vars(args),
        "input": asdict(config), "versions": package_versions(),
        "manifest_sha256": sha256_file(args.manifest), "selected_records_sha256": hash_json(records),
        "weight_identity": {"base": model_weight_identity(args.model_path), "adapter": model_weight_identity(args.adapter_path, adapter=True)},
        "in_memory_parameters_before": all_before, "adapter_fingerprint_before": adapter_before,
        "trainable_inventory": inventory, "candidate_texts": CANDIDATES,
        "objective": "Exact and Pair closed-24 expected rewards; closed-24 CE ascent; raw canonical sequence log likelihood ascent; log canonical mass",
        "mode": "model.eval(), LoRA trainable, autograd.grad only, use_cache=False, no optimizer or updates",
        "gradient_aggregation_dtype": "float64; original model/autograd gradient precision unchanged",
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in
                          (Path(__file__), HERE / "score_permutations.py", ROOT / "src/recipeqa/modeling.py", ROOT / "src/recipeqa/metrics.py")},
    }
    signature = hash_json(provenance)
    write_json(config_path, provenance)
    score_args = SimpleNamespace(microbatch=1, batch_atol=args.logp_atol, tie_atol=1e-8)
    for index, record in enumerate(records):
        prepared, audit = prepare_record(record, processor, args.image_root, config)
        candidates = candidate_suffixes(processor, prepared["prompt"])
        encoded = encode_messages(processor, prepared["prompt"], load_images(record, args.image_root)).to(model.device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        token_logps, references = score_candidates(model, encoded, candidates, score_args, literal_check=index == 0)
        scores = [sum(values) for values in token_logps]
        summary = probability_summary(scores, " ".join(record["target_labels"]), audit["prompt_text_hash"])
        require(summary["logZ_canonical_24"] <= 1e-4, "Canonical string mass exceeds one")
        rewards = [score_order(candidate["text"], record["display_perm"], record["target_order"]) for candidate in candidates]
        exact, pair = [score["exact"] for score in rewards], [score["pair"] for score in rewards]
        require(CANDIDATES[exact.index(1)] == " ".join(record["target_labels"]), "Training gold labels contradict source mapping")
        coefficients = coefficient_table(summary["q24"], exact, pair)
        buffers = {name: torch.zeros(inventory["trainable_parameters"], dtype=torch.float64, device=model.device)
                   for name in coefficients}
        candidate_checks = []
        original_gradient_dtypes = set()
        for candidate_index, candidate in enumerate(candidates):
            sequence, checked_token_logps = differentiable_logp(model, encoded, candidate["token_ids"])
            delta = max(abs(a - b) for a, b in zip(checked_token_logps, token_logps[candidate_index]))
            require(delta <= args.logp_atol, f"Differentiable/nograd forward mismatch on candidate {candidate_index}: {delta}")
            require(math.isfinite(float(sequence.detach().item())), "Non-finite differentiable sequence likelihood")
            gradients = torch.autograd.grad(sequence, parameters, retain_graph=False, create_graph=False, allow_unused=False)
            offset = 0
            for parameter, gradient in zip(parameters, gradients):
                count = parameter.numel()
                original_gradient_dtypes.add(str(gradient.dtype))
                # Strong SFT can give q_gold close to one. Accumulate in double
                # so raw-SFT minus logZ does not lose its small closed-CE signal
                # merely because both large component buffers were FP32 sums.
                flat = gradient.detach().reshape(-1).double()
                require(bool(torch.isfinite(flat).all().item()), "Non-finite actual LoRA gradient")
                for name, buffer in buffers.items():
                    coefficient = coefficients[name][candidate_index]
                    if coefficient:
                        buffer[offset:offset + count].add_(flat, alpha=coefficient)
                offset += count
            require(offset == inventory["trainable_parameters"], "Parameter gradient flattening differs from inventory")
            candidate_checks.append({"candidate_index": candidate_index, "text": candidate["text"],
                                     "max_token_logp_difference_grad_vs_nograd": delta,
                                     "differentiable_sequence_logp": float(sequence.detach().item())})
            del sequence, gradients
        geometry = vector_summary(buffers, summary["q24_gold"], args.identity_atol, args.identity_rtol)
        require(all(parameter.grad is None for parameter in parameters), "autograd.grad unexpectedly accumulated parameter .grad")
        require(adapter_fingerprint(model) == adapter_before, "Trainable adapter weights changed during read-only diagnosis")
        torch.cuda.synchronize()
        result = {
            "run_id": args.run_id, "qid": record["qid"], "recipe_id": record["recipe_id"],
            "presentation_id": record["presentation_id"], "source_split": "train", "manifest_set": "train_visits",
            "configuration_sha256": signature, "input_audit": audit, "display_perm": record["display_perm"],
            "target_order": record["target_order"], "target_labels": record["target_labels"],
            "candidates": [{**candidate, "token_logps_including_eos": values, "logp_including_eos": sum(values),
                            "exact_reward_rescored": reward["exact"], "pair_reward_rescored": reward["pair"]}
                           for candidate, values, reward in zip(candidates, token_logps, rewards)],
            "canonical_map": summary, "detached_gradient_coefficients": coefficients,
            "parameter_gradient_geometry": geometry, "numerical_reference_checks": references,
            "candidate_grad_forward_checks": candidate_checks,
            "gradient_aggregation_dtype": "torch.float64",
            "original_autograd_gradient_dtypes": sorted(original_gradient_dtypes),
            "adapter_unchanged": True, "parameter_grad_buffers_all_none": True,
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "completed_utc": datetime.now(timezone.utc).isoformat(), "technical_error": None,
            "limits": ["First fixed training records only; not a representative prevalence estimate.",
                       "Raw Euclidean LoRA gradient geometry, not an Adam-preconditioned or clipped update.",
                       "Canonical closed policy differs from old unrestricted GRPO; no optimizer step or held-out improvement measured.",
                       "Raw SFT gradient means canonical complete-sequence log likelihood ascent, before token-count loss normalization."]}
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, allow_nan=False) + "\n")
        print(json.dumps({"done": index + 1, "planned": args.limit, "qid": record["qid"],
                          "parameter_exact_pair_cosine": geometry["comparisons"]["exact__pair"]["cosine"],
                          "seconds": result["wall_seconds"]}), flush=True)
        del buffers, encoded
    all_after = parameter_fingerprint(model)
    require(all_after == all_before, "Actual in-memory model parameters changed; diagnostic is not read-only")
    write_json(completion_path, {"status": "complete_read_only_parameter_geometry", "configuration_sha256": signature,
                                "rows": len(records), "output_sha256": sha256_file(output),
                                "in_memory_parameters_before": all_before, "in_memory_parameters_after": all_after,
                                "all_model_parameters_unchanged": True, "optimizer_constructed": False, "updates": 0})


if __name__ == "__main__":
    main()
