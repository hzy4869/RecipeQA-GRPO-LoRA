#!/usr/bin/env python3
"""Explicit two-pass reference-refresh derivative of frozen coverage trainer.

Keeps EOS24 objective/autograd/Adam math; adds actual midpoint reference
scoring with complete optimizer/grad/mode/RNG continuity, then final fixed
training-bank scoring. New policy, namespace and original20h commonclock.
No heuristic reward, score filtering, checkpoint selection, resume or old edits.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "exploratory/principled_ordering_20261105"))
from refresh_contract import (TRAIN_SOURCE_PATHS, SCORING_SOURCE_PATHS, WALL_SECONDS,
    validate_child, remaining_seconds)
_LOCKED_POLICY = None
from recipeqa.metrics import score_order
from recipeqa.modeling import (
    InputConfig, adapter_fingerprint, encode_messages, ensure_data_disk,
    frozen_vision_fingerprint, hash_json, load_images, load_model, load_processor,
    model_weight_identity, package_versions, prepare_record, read_jsonl,
    sha256_file, trainable_inventory, write_json,
)
from score_permutations import CANDIDATES, candidate_suffixes, score_candidates
from gradient_alignment import differentiable_logp

METHODS = ("raw_sft", "exact", "exact_kl", "rloo_kl")
ACCUMULATION = 4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def log_policy(scores):
    require(len(scores) == 24 and all(math.isfinite(value) for value in scores), "Need24finite complete canonical scores")
    maximum = max(scores)
    logz = maximum + math.log(math.fsum(math.exp(value - maximum) for value in scores))
    logq = [value - logz for value in scores]
    q = [math.exp(value) for value in logq]
    require(abs(math.fsum(q) - 1) <= 1e-9, "Closed policy normalization failed")
    return {"q": q, "logq": logq, "logZ": logz}


def categorical_draws(q, count, rng):
    require(count == 8, "Training RLOO uses exactly8 independent draws")
    require(len(q) == 24 and all(value >= 0 and math.isfinite(value) for value in q)
            and abs(math.fsum(q) - 1) <= 1e-9, "Invalid sampling policy")
    cumulative, total = [], 0.0
    for probability in q:
        total += probability
        cumulative.append(total)
    cumulative[-1] = 1.0  # roundoff convention, not reward-conditioned sampling
    uniforms = [rng.random() for _ in range(count)]
    indices = [bisect_right(cumulative, value) for value in uniforms]
    require(all(0 <= index < 24 and q[index] > 0 for index in indices), "Categorical sampler selected zero mass")
    return indices, uniforms


def policy_coefficients(method, policy, exact_rewards, reference_logq, beta, samples=None):
    """Detached coefficients of grad(raw sequence logp), before /C.

    For KL the explicit derivative of logq contributes a constant whose exact
    expectation is zero. Thus u=r-beta*log(q/qref) gives an unbiased RLOO score
    estimator. RLOO advantages sum to zero, cancelling every logZ derivative.
    """
    require(method in METHODS and len(exact_rewards) == 24 and sum(exact_rewards) == 1
            and set(exact_rewards) <= {0, 1}, "Need one verified exact target")
    require(len(reference_logq) == 24 and all(math.isfinite(value) for value in reference_logq), "Bad fixed reference logq")
    require(math.isfinite(beta) and beta >= 0, "Invalid KL coefficient")
    q, logq = policy["q"], policy["logq"]
    require(len(q) == len(logq) == 24 and all(math.isfinite(value) and value >= 0 for value in q)
            and all(math.isfinite(value) for value in logq) and abs(math.fsum(q) - 1) <= 1e-9, "Incomplete or invalid closed policy")
    gold = exact_rewards.index(1)
    active_beta = beta if method in ("exact_kl", "rloo_kl") else 0.0
    u = [reward - active_beta * (current - reference)
         for reward, current, reference in zip(exact_rewards, logq, reference_logq)]
    require(all(math.isfinite(value) for value in u), "Non-finite regularized outcome score")
    mean_u = math.fsum(p * value for p, value in zip(q, u))
    coefficients = [0.0] * 24
    advantages = None
    if method == "raw_sft":
        coefficients[gold] = 1.0
        require(samples is None, "Raw SFT does not sample")
    elif method in ("exact", "exact_kl"):
        coefficients = [p * (value - mean_u) for p, value in zip(q, u)]
        require(samples is None, "Enumerated objectives do not sample")
    else:
        require(isinstance(samples, list) and len(samples) >= 2
                and all(type(index) is int and 0 <= index < 24 for index in samples), "RLOO needs independent categorical samples")
        sample_u = [u[index] for index in samples]
        total = math.fsum(sample_u)
        advantages = ([0.0] * len(samples) if len(set(sample_u)) == 1 else
                      [value - (total - value) / (len(samples) - 1) for value in sample_u])
        for index, advantage in zip(samples, advantages):
            coefficients[index] += advantage / len(samples)
        require(abs(math.fsum(advantages)) <= 1e-10 * max(1.0, max(abs(value) for value in advantages)),
                "Leave-one-out advantages do not sum to zero")
    if method != "raw_sft":
        require(abs(math.fsum(coefficients)) <= 1e-9 * max(1.0, max(abs(value) for value in coefficients)),
                "Closed-policy coefficient sum is not zero")
    kl = math.fsum(p * (current - reference) for p, current, reference in zip(q, logq, reference_logq))
    return {"coefficients": coefficients, "u": u, "mean_u": mean_u, "advantages": advantages,
            "exact_expected_reward": q[gold], "kl_q_reference": kl,
            "regularized_expected_reward": q[gold] - active_beta * kl,
            "closed_ce_nll": -logq[gold], "gold_index": gold, "active_beta": active_beta}


def validate_budget(records, limit, max_steps, unique_count, passes):
    require((unique_count, passes, limit, max_steps) in ((4,1,4,1),(4,2,8,2),(256,2,512,128)),
            "Only fixed ordinary/path smoke or two-pass formal budgets")
    require(len(records) == unique_count and len({(str(r['qid']),str(r['presentation_id'])) for r in records}) == unique_count,
            "Unique input manifest differs; repeated visits are separate from question identity")
    for row in records:
        require(row.get('source_split') == 'train' and row.get('technical_status') == 'ok'
                and row.get('manifest_set') not in ('val','test','confirmation','orbit','process'), "Only original undamaged train input")
        require(len(row.get('target_labels',[])) == 4 and set(row['target_labels']) == set('ABCD'), "Bad canonical gold")
    return records


def append(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def pixel_identities(records, image_root):
    root = Path(image_root).resolve()
    result = {}
    for record in records:
        for relative in record["image_paths_display"]:
            path = (root / relative).resolve()
            require(path.is_relative_to(root) and path.is_file(), "Selected image escaped or is missing")
            if relative not in result:
                result[relative] = sha256_file(path)
    return result


def sources():
    require(_LOCKED_POLICY is not None, "Refresh policy must be loaded before source binding")
    require(set(_LOCKED_POLICY["sources_sha256"]) == set(TRAIN_SOURCE_PATHS), "Whole reviewed source closure required")
    result = {name: sha256_file(ROOT / name) for name in TRAIN_SOURCE_PATHS}
    require(result == _LOCKED_POLICY["sources_sha256"], "Frozen refresh source changed")
    return result


def cpu_gradient_norm(parameters):
    import torch

    norms = torch.stack([parameter.grad.detach().float().norm() for parameter in parameters])
    values = norms.cpu().tolist()
    return {"l2_norm": math.sqrt(math.fsum(value * value for value in values)),
            "trainable_gradient_tensors": len(values), "nonzero_gradient_tensors": sum(value > 0 for value in values)}


def arguments():
    global _LOCKED_POLICY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parsed = parser.parse_args()
    path = Path(parsed.policy).resolve()
    require(path.is_relative_to(ROOT), "Refresh policy escaped data disk")
    _LOCKED_POLICY = json.loads(path.read_text())
    cli = validate_child(_LOCKED_POLICY)
    require(os.environ.get("RECIPEQA_REFRESH_OUTER_WATCHDOG"), "Outer native process-group watchdog required")
    remaining_seconds(_LOCKED_POLICY)
    for name, digest in _LOCKED_POLICY["artifacts_sha256"].items():
        actual = (ROOT / name).resolve()
        require(actual.is_relative_to(ROOT) and sha256_file(actual) == digest, "Frozen preexisting artifact changed: " + name)
    validate_authorization(_LOCKED_POLICY, path)
    return argparse.Namespace(**cli, policy=str(path), arm=_LOCKED_POLICY["arm"], mode=_LOCKED_POLICY["mode"])


def main():
    args = arguments()
    ensure_data_disk(args.model_path, args.adapter_path, args.manifest, args.image_root, args.output_dir)
    if args.reference_cache:
        ensure_data_disk(args.reference_cache)
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TMPDIR", "TORCH_HOME"):
        require(bool(os.environ.get(name)), "Missing data-disk environment: " + name)
        ensure_data_disk(os.environ[name])
    require(sha256_file(args.manifest) == args.expected_manifest_sha256, "Explicitly bound manifest SHA differs")
    require(math.isfinite(args.learning_rate) and args.learning_rate > 0 and args.save_steps > 0
            and 0 < args.wall_seconds <= WALL_SECONDS and args.logp_atol == .01, "Invalid optimization/runtime settings")
    if args.method in ("exact_kl", "rloo_kl"):
        require(args.beta is not None and math.isfinite(args.beta) and args.beta > 0, "Named KL method needs explicit positive beta; no silent unregularized fallback")
    else:
        require(args.beta is None or args.beta == 0, "Unregularized methods do not accept a hidden beta")
    beta = args.beta or 0.0
    policy_sha = sha256_file(args.policy)
    records = validate_budget(read_jsonl(args.manifest), args.limit, args.max_steps, args.unique_count, args.passes)
    max_steps = args.limit // ACCUMULATION
    output = Path(args.output_dir)
    require(not output.exists(), "Fresh output directory required; no automatic resume or overwrite")
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + min(args.wall_seconds, remaining_seconds(_LOCKED_POLICY))
    state = {"status": "preparing", "method": args.method, "arm": args.arm, "mode": args.mode, "started_utc": now(), "global_step": 0,
             "actual_prompt_visits": 0, "no_resume": True, "test_access": False, "post_hoc_exploration": True}
    state_path = output / "state.json"
    write_json(state_path, state)

    def update(**changes):
        state.update(changes, updated_utc=now())
        write_json(state_path, state)

    def budget():
        require(time.monotonic() < deadline, "Refresh child/common wall budget exhausted; preserve partial evidence, no resume")
        remaining_seconds(_LOCKED_POLICY)
        require(sha256_file(args.policy) == policy_sha, "Frozen child policy changed")

    try:
        import torch
        import numpy as np

        config = InputConfig(args.min_pixels, args.max_pixels, args.recipe_token_budget, args.max_input_tokens, 32)
        bound_sources = sources()
        pixels = pixel_identities(records, args.image_root)
        initial_identity = {"base": model_weight_identity(args.model_path), "adapter": model_weight_identity(args.adapter_path, adapter=True)}
        require(initial_identity == _LOCKED_POLICY["initial_identity"], "Fixed SFT768 initial weights changed")
        reference_spec = {"schema_version": 1, "policy": "canonical24_full_sequence_plusEOS_temperature1",
                          "initial_identity": initial_identity, "input": asdict(config), "revision": args.revision,
                          "manifest_sha256": args.expected_manifest_sha256, "selected_records_sha256": hash_json(records),
                          "candidate_texts": CANDIDATES, "pixel_sha256": pixels,
                          "numerical_reference_atol": args.logp_atol,
                          "scoring_source_sha256": {key: bound_sources[key] for key in SCORING_SOURCE_PATHS},
                          "versions": package_versions()}
        processor = load_processor(args.model_path, args.revision, config)
        model = load_model(args.model_path, args.revision, adapter_path=args.adapter_path, trainable=True, seed=args.seed)
        model.to("cuda")
        model.eval()  # deliberate deterministic policy; gradients remain enabled
        model.config.use_cache = False
        inventory = trainable_inventory(model)
        require(inventory["trainable_parameters"] == 17432576 and len(inventory["parameters"]) == 392,
                "LoRA dimensions differ from the shared17,432,576-parameter392-tensor protocol")
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        initial_adapter = adapter_fingerprint(model)
        initial_vision = frozen_vision_fingerprint(model)
        score_args = SimpleNamespace(microbatch=1, batch_atol=args.logp_atol, tie_atol=1e-8)
        reference_path = Path(args.reference_cache) if args.reference_cache else output / "reference_policy.jsonl"
        reference_config_path = Path(str(reference_path) + ".config.json")
        reference_completion_path = Path(str(reference_path) + ".completion.json")
        reference_forward_candidates = 0

        def prepare(record):
            prepared, audit = prepare_record(record, processor, args.image_root, config)
            candidates = candidate_suffixes(processor, prepared["prompt"])
            encoded = encode_messages(processor, prepared["prompt"], load_images(record, args.image_root)).to(model.device)
            return audit, candidates, encoded

        require(bool(args.reference_cache) and reference_config_path.is_file() and reference_completion_path.is_file(), "Sealed q0 cache and sidecars required")
        cached_spec = json.loads(reference_config_path.read_text())
        require(cached_spec == json.loads(json.dumps(reference_spec)), "q0 cache differs in actual weights/input/pixels/versions/scoring")
        completion = json.loads(reference_completion_path.read_text())
        require(completion.get('status') == 'complete_fixed_initial_reference' and completion.get('rows') == args.unique_count
                and completion.get('cache_sha256') == sha256_file(reference_path), "q0 cache incomplete or changed")
        reference = read_jsonl(reference_path)
        require(len(reference) == args.unique_count, "Incomplete unique q0 policy")
        for index, (record, ref) in enumerate(zip(records, reference)):
            require(ref["index"] == index and ref["qid"] == record["qid"] and ref["presentation_id"] == record["presentation_id"]
                    and ref["reference_spec_sha256"] == hash_json(reference_spec), "Reference prefix/signature differs")
            require(len(ref["candidates"]) == len(ref["q"]) == len(ref["logq"]) == len(ref["token_logps"]) == 24,
                    "Incomplete reference actions")
            require(ref["source_split"] == "train" and all(ref[field] == record[field] for field in
                    ("recipe_id", "display_perm", "target_order", "target_labels")), "Reference verifier metadata differs")
            require(tuple(candidate["text"] for candidate in ref["candidates"]) == CANDIDATES, "Reference action text/order changed")
            for candidate, values in zip(ref["candidates"], ref["token_logps"]):
                require(len(values) == candidate["token_count"] == len(candidate["token_ids"])
                        and candidate["token_ids"][-1] == candidate["eos_token_id"]
                        and all(math.isfinite(value) and value <= 1e-6 for value in values), "Reference suffix/EOS token scores incomplete")
            rescored = log_policy([sum(values) for values in ref["token_logps"]])
            require(all(abs(a - b) <= 1e-9 for a, b in zip(rescored["logq"], ref["logq"]))
                    and all(abs(a - b) <= 1e-9 for a, b in zip(rescored["q"], ref["q"])), "Reference q/logq does not match its raw token scores")
            exact = [score_order(candidate["text"], record["display_perm"], record["target_order"])["exact"]
                     for candidate in ref["candidates"]]
            require(exact == ref["exact_rewards"] and sum(exact) == 1
                    and ref["gold_candidate_index"] == exact.index(1), "Reference cached rewards differ from source mapping")
        reference_hash = sha256_file(reference_path)
        token_counts = {candidate["token_count"] for ref in reference for candidate in ref["candidates"]}
        require(len(token_counts) == 1, "Sequence objective normalization C is not constant across all selected inputs/actions")
        canonical_token_count = next(iter(token_counts))
        require(canonical_token_count == 5, "Refresh fixed canonical C=5 includingEOS required")
        signature_payload = {"schema_version": 2, "method": args.method, "arm": args.arm, "mode": args.mode,
                             "unique_count": args.unique_count, "passes": args.passes, "refresh_boundary_step": args.refresh_boundary_step,
                             "visit_sequence_sha256": hash_json([{"visit_index": i, "pass_index": i // args.unique_count, "index_within_pass": i % args.unique_count, "record_sha256": hash_json(records[i % args.unique_count])} for i in range(args.limit)]),
                             "authorization_sha256": sha256_file(ROOT / _LOCKED_POLICY["authorization_path"]), "cli": vars(args), "resolved_beta": beta,
                             "locked_policy_sha256": policy_sha, "refresh_mode": _LOCKED_POLICY["mode"],
                             "reference_lineage_sha256": sha256_file(ROOT / _LOCKED_POLICY["q0_reference_lineage"]),
                             "generated_reference_receipt_sha256": sha256_file(ROOT / _LOCKED_POLICY["generated_reference_receipt"]),
                             "batch_clock_started_utc": _LOCKED_POLICY["batch_clock_started_utc"],
                             "absolute_deadline_utc": _LOCKED_POLICY["absolute_deadline_utc"],
                             "actual_updates": max_steps, "accumulation": ACCUMULATION, "input": asdict(config),
                             "canonical_completion_token_count": canonical_token_count,
                             "initial_identity": initial_identity, "initial_adapter_fingerprint": initial_adapter,
                             "initial_visual_fingerprint": initial_vision, "manifest_sha256": args.expected_manifest_sha256,
                             "selected_records_sha256": hash_json(records), "pixel_sha256": pixels,
                             "reference_policy_sha256": reference_hash, "reference_spec_sha256": hash_json(reference_spec),
                             "sources_sha256": bound_sources, "versions": package_versions(),
                             "optimizer": {"name": "AdamW", "lr": args.learning_rate, "betas": [.9, .999], "eps": 1e-8,
                                           "weight_decay": 0.0, "max_grad_norm": 1.0, "schedule": "constant", "fresh_state": True},
                             "mode": "eval with LoRA autograd; no KV cache; all sequence objectives divided by canonical token count",
                             "sampling": "RLOO only:CPU random.Random(seed),8 independent categorical with replacement; duplicates retained; no resampling"}
        signature = hash_json(signature_payload)
        write_json(output / "run_signature.json", {"sha256": signature, "payload": signature_payload})
        write_json(output / "trainable_parameters.json", inventory)
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, betas=(.9, .999), eps=1e-8, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        require(not optimizer.state, "Optimizer did not start fresh")
        rng = random.Random(args.seed)
        actual_visits = []
        checkpoints = []
        current_policy_forward_candidates = 0
        gradient_forward_candidates = 0
        total_backward_candidates = 0
        active_reference = reference
        active_reference_hash = reference_hash
        reference_generation = 0
        refresh_forward_candidates = 0
        final_bank_forward_candidates = 0
        refresh_event = None
        torch.cuda.reset_peak_memory_stats()

        def immutable():
            budget()
            require(sources() == bound_sources and sha256_file(args.manifest) == args.expected_manifest_sha256
                    and sha256_file(reference_path) == reference_hash, "Bound scientific source/data/reference changed during training")

        def save_checkpoint(step, directory):
            immutable()
            require(not directory.exists(), "Never overwrite an existing checkpoint")
            directory.mkdir()
            model.save_pretrained(directory, safe_serialization=True)
            processor.save_pretrained(directory)
            torch.save(optimizer.state_dict(), directory / "optimizer.pt")
            torch.save(scheduler.state_dict(), directory / "scheduler.pt")
            torch.save({"python_global": random.getstate(), "sampling_rng": rng.getstate(), "numpy": np.random.get_state(),
                        "cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}, directory / "rng_state.pth")
            write_json(directory / "trainer_state.json", {"global_step": step, "epoch": step / max_steps,
                                                         "actual_prompt_visits": len(actual_visits), "run_signature_sha256": signature,
                                                         "resume_supported": False})
            identity = model_weight_identity(directory, adapter=True)
            write_json(directory / "finite_policy_checkpoint_audit.json", {"global_step": step, "adapter_identity": identity,
                       "run_signature_sha256": signature, "reference_policy_sha256": reference_hash, "active_reference_policy_sha256": active_reference_hash, "reference_generation": reference_generation,
                       "state_sha256": {name: sha256_file(directory / name) for name in
                                        ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")}})
            checkpoints.append({"step": step, "path": str(directory), "adapter_identity": identity})

        update(status="training", run_signature_sha256=signature, reference_policy_sha256=reference_hash)
        for step in range(max_steps):
            immutable()
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=False)
            # zero_grad(False) does not allocate missing .grad tensors on step0.
            # Explicit zeros ensure even all-constant RLOO batches execute Adam.
            for parameter in parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
            group_active = []
            for visit_index in range(step * ACCUMULATION, (step + 1) * ACCUMULATION):
                unique_index = visit_index % args.unique_count
                record, ref = records[unique_index], active_reference[unique_index]
                audit, candidates, encoded = prepare(record)
                require(audit["input_hash"] == ref["input_audit"]["input_hash"] and candidates == ref["candidates"],
                        "Actual training input/grammar differs from fixed reference")
                token_logps, checks = score_candidates(model, encoded, candidates, score_args, literal_check=visit_index == 0)
                current_policy_forward_candidates += 24 + len(checks)
                scores = [sum(values) for values in token_logps]
                policy = log_policy(scores)
                require(policy["logZ"] <= 1e-4, "Current canonical mass exceeds one")
                rewards = [score_order(candidate["text"], record["display_perm"], record["target_order"])["exact"] for candidate in candidates]
                require(sum(rewards) == 1 and CANDIDATES[rewards.index(1)] == " ".join(record["target_labels"]), "Source gold mapping differs")
                samples, uniforms = categorical_draws(policy["q"], 8, rng) if args.method == "rloo_kl" else (None, None)
                terms = policy_coefficients(args.method, policy, rewards, ref["logq"], beta, samples)
                count = candidates[0]["token_count"]
                backward_checks = []
                for candidate_index, coefficient in enumerate(terms["coefficients"]):
                    if coefficient == 0:
                        continue
                    budget()
                    scalar, verified_logps = differentiable_logp(model, encoded, candidates[candidate_index]["token_ids"])
                    difference = max(abs(a - b) for a, b in zip(verified_logps, token_logps[candidate_index]))
                    require(difference <= args.logp_atol, "Differentiable/current no-grad likelihood mismatch")
                    loss = -coefficient * scalar / (count * ACCUMULATION)
                    require(bool(torch.isfinite(loss).item()), "Non-finite weighted training loss")
                    loss.backward()
                    backward_checks.append({"candidate_index": candidate_index, "max_token_logp_difference": difference})
                    total_backward_candidates += 1
                    gradient_forward_candidates += 1
                    del scalar, loss
                active = any(value != 0 for value in terms["coefficients"])
                group_active.append(active)
                visit = {"visit_index": visit_index, "pass_index": visit_index // args.unique_count, "index_within_pass": unique_index,
                         "reference_row_index": unique_index, "reference_generation": reference_generation, "arm": args.arm,
                         "q0_reference_policy_sha256": reference_hash, "active_reference_policy_sha256": active_reference_hash,
                         "visit_id": record.get("visit_id"), "qid": record["qid"],
                         "recipe_id": record["recipe_id"], "presentation_id": record["presentation_id"],
                         "source_split": "train", "global_step_before_update": step,
                         "input_hash": audit["input_hash"], "input_audit": audit,
                         "display_perm": record["display_perm"], "target_order": record["target_order"], "target_labels": record["target_labels"],
                         "candidate_texts": CANDIDATES, "candidate_token_ids": [candidate["token_ids"] for candidate in candidates],
                         "sequence_logps": scores, "token_logps_including_eos": token_logps, **policy,
                         "canonical_mass": math.exp(policy["logZ"]),
                         "exact_rewards": rewards, "reference_logq": ref["logq"], "reference_policy_sha256": active_reference_hash,
                         "sampled_indices": samples, "sampling_uniforms": uniforms, "sampling_rng_state_sha256": hash_json(rng.getstate()),
                         **terms, "completion_token_count": count, "nonzero_coefficients": sum(value != 0 for value in terms["coefficients"]),
                         "group_active": active, "backward_checks": backward_checks, "numerical_reference_checks": checks,
                         "run_signature_sha256": signature, "consumption_point": "all_selected_candidate_backward_calls_returned",
                         "policy_gradient_scope": "current q24 over EOS-complete canonical strings; no free-token rollout"}
                append(output / "actual_prompt_visits.jsonl", visit)
                actual_visits.append((record["qid"], record["presentation_id"]))
                del encoded
            require(all(parameter.grad is not None for parameter in parameters), "Missing gradient tensors before planned Adam step")
            before_clip = cpu_gradient_norm(parameters)
            require(math.isfinite(before_clip["l2_norm"]), "Non-finite accumulated gradient")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            after_clip = cpu_gradient_norm(parameters)
            previous = [parameter.detach().clone() for parameter in parameters]
            optimizer.step()  # deliberately executes even when every group is inactive
            scheduler.step()
            difference_norms = torch.stack([(parameter.detach() - old).float().norm() for parameter, old in zip(parameters, previous)])
            update_norm = float(difference_norms.norm().item())
            del previous
            entry = {"global_step_before_update": step, "global_step": step + 1, "arm": args.arm, "reference_generation": reference_generation,
                     "active_reference_policy_sha256": active_reference_hash,
                     "actual_prompt_visits": len(actual_visits), "active_groups": sum(group_active), "groups": ACCUMULATION,
                     "zero_reward_gradient_batch": before_clip["l2_norm"] == 0,
                     "before_clip": before_clip, "after_clip": after_clip, "actual_parameter_update_l2": update_norm,
                     "optimizer_step_executed": True, "optimizer_state_entries": len(optimizer.state),
                     "learning_rate": optimizer.param_groups[0]["lr"], "run_signature_sha256": signature,
                     "elapsed_seconds": time.monotonic() - step_started,
                     "cumulative_current_policy_candidate_forwards_including_numerical_checks": current_policy_forward_candidates,
                     "cumulative_gradient_candidate_forwards": gradient_forward_candidates,
                     "initial_reference_candidate_forwards_charged_this_run": reference_forward_candidates,
                     "cumulative_candidate_backwards": total_backward_candidates,
                     "cumulative_refreshed_reference_candidate_forwards": refresh_forward_candidates}
            append(output / "step_times.jsonl", entry)
            update(global_step=step + 1, actual_prompt_visits=len(actual_visits), last_step=entry)
            if (step + 1) % args.save_steps == 0 or step + 1 == max_steps:
                save_checkpoint(step + 1, output / f"checkpoint-{step + 1}")
            if args.refresh_boundary_step == step + 1:
                refresh_event, active_reference = build_boundary_reference(step + 1)
                active_reference_hash = refresh_event["reference_policy_sha256"]
                refresh_forward_candidates += refresh_event["candidate_forwards"]
                reference_generation = 1
                update(status="training", reference_generation=1, active_reference_policy_sha256=active_reference_hash, refresh_event=refresh_event)
            print(json.dumps({"step": step + 1, "planned": max_steps, "method": args.method,
                              "visits": len(actual_visits), "gradient_norm": before_clip["l2_norm"],
                              "active_groups": sum(group_active), "seconds": entry["elapsed_seconds"]}), flush=True)
        immutable()
        require(actual_visits == [(records[i % args.unique_count]["qid"], records[i % args.unique_count]["presentation_id"]) for i in range(args.limit)], "Actual two-pass fixed flow differs")
        final_bank = build_final_bank(max_steps)
        final_bank_forward_candidates = final_bank["candidate_forwards"]
        final_vision = frozen_vision_fingerprint(model)
        final_adapter = adapter_fingerprint(model)
        require(final_vision == initial_vision, "Frozen visual weights changed")
        require(pixel_identities(records, args.image_root) == pixels, "Selected training image bytes changed")
        outcome = {"status": "completed", "method": args.method, "arm": args.arm, "mode": args.mode, "global_step": max_steps,
                   "unique_count": args.unique_count, "passes": args.passes, "refresh_event": refresh_event, "final_train_bank": final_bank,
                   "actual_prompt_visits": len(actual_visits), "unique_questions": len({record["qid"] for record in records}),
                   "unique_recipes": len({record["recipe_id"] for record in records}), "actual_visit_sequence_verified": True,
                   "effective_adapter_update": final_adapter != initial_adapter,
                   "scientific_outcome": "effective_training" if final_adapter != initial_adapter else "no_effective_adapter_update",
                   "initial_adapter_fingerprint": initial_adapter,
                   "final_adapter_fingerprint": final_adapter, "initial_visual_fingerprint": initial_vision,
                   "final_visual_fingerprint": final_vision, "visual_weights_unchanged": True,
                   "elapsed_seconds_including_reference": time.monotonic() - started,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "candidate_forwards": reference_forward_candidates + current_policy_forward_candidates + gradient_forward_candidates + refresh_forward_candidates + final_bank_forward_candidates,
                   "refreshed_reference_candidate_forwards": refresh_forward_candidates, "final_bank_candidate_forwards": final_bank_forward_candidates,
                   "reference_candidate_forwards_charged_this_run": reference_forward_candidates,
                   "current_policy_candidate_forwards_including_numerical_checks": current_policy_forward_candidates,
                   "gradient_candidate_forwards": gradient_forward_candidates,
                   "candidate_backwards": total_backward_candidates,
                   "reference_policy_sha256": reference_hash, "run_signature_sha256": signature,
                   "actual_visits_sha256": sha256_file(output / "actual_prompt_visits.jsonl"),
                   "step_times_sha256": sha256_file(output / "step_times.jsonl"), "checkpoints": checkpoints,
                   "final_adapter_identity": model_weight_identity(output / f"checkpoint-{max_steps}", adapter=True),
                   "resume_supported": False, "test_access": False}
        write_json(output / "train_outcome.json", outcome)
        update(status="completed", finished_utc=now(), outcome=outcome)
    except BaseException as error:
        update(status="failed_stopped_no_resume", error=f"{type(error).__name__}: {error}", finished_utc=now(),
               elapsed_seconds=time.monotonic() - started)
        raise


if __name__ == "__main__":
    main()
