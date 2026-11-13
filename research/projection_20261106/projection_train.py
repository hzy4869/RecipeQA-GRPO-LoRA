#!/usr/bin/env python3
"""DRAFT ONLY: fixed-target forward-KL policy projection on24 canonical actions.

Adapted into a NEW file from the frozen finite_policy_train.py. Original sources
are never edited. Reuse their scorer, autograd sequence scorer, input code and
reference-cache signature. This prototype refuses draft/unfrozen policies.
No SSH/GPU execution or additional experimental budget is authorized by this file.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT / 'exploratory/principled_ordering_20261105'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(LEGACY))
from recipeqa.metrics import score_order
from recipeqa.modeling import (InputConfig, adapter_fingerprint, encode_messages, ensure_data_disk,
    frozen_vision_fingerprint, hash_json, load_images, load_model, load_processor,
    model_weight_identity, package_versions, prepare_record, read_jsonl, sha256_file,
    trainable_inventory, write_json)
from score_permutations import CANDIDATES, candidate_suffixes, score_candidates
from gradient_alignment import differentiable_logp
from finite_policy_train import (require, now, log_policy, validate_budget, append,
    pixel_identities, cpu_gradient_norm, sources as legacy_sources)
from projection_math import make_qstar, projection_terms, validate_token_count

ACCUMULATION = 4


def sources():
    result = legacy_sources()
    for path in (Path(__file__), HERE / 'projection_math.py', HERE / 'test_projection_math.py'):
        result[str(path.relative_to(ROOT))] = sha256_file(path)
    return result


def json_cli(args):
    """Keep the scientific signature JSON-native, including the policy path."""
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--method", choices=("forward_projection",), default="forward_projection")
    parser.add_argument("--policy", type=Path, required=True, help="Future separately frozen protocol; draft status is refused")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-cache", required=True, help="Existing unchanged complete fixed reference with signed sidecars; never regenerate here")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, help="Must be explicit forKL methods; standard objective coefficient, not reward shaping")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--save-steps", type=int, default=16)
    parser.add_argument("--wall-seconds", type=float, default=10800)
    parser.add_argument("--logp-atol", type=float, default=.01)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--recipe-token-budget", type=int, default=1536)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    return parser.parse_args()


def main():
    args = arguments()
    locked_policy = json.loads(args.policy.read_text())
    require(locked_policy.get('status') == 'locked_before_projection_training', 'DRAFT trainer: a new reviewed/frozen protocol and separate GPU budget are required')
    require(locked_policy.get('method') == 'forward_projection' and locked_policy.get('reward') == 'unique_complete_order_exact_0_1', 'Method/reward policy differs')
    scientific_cli = json_cli(args)
    scientific_cli.pop('policy')
    require(scientific_cli == locked_policy['cli'], 'Every CLI field including revision, input budgets and tolerances must match the new locked policy')
    require(locked_policy.get('mode') == args.mode, 'Mode differs from new locked policy')
    require(args.learning_rate == 5e-6, 'Keep original fixed LR5e-6 in both modes')
    if args.mode == 'formal':
        require(args.limit == 256 and args.max_steps == 64 and args.save_steps == 16, 'Formal policy requires256 visits/64 updates/save16')
    else:
        require(args.limit == 4 and args.max_steps == 1 and args.save_steps == 1, 'Separately locked smoke requires4 visits/1 update/save1')
    require(args.beta == 1.6363096678310884, 'Use the existing fixed positive beta; do not recalibrate')
    require(locked_policy.get('sources_sha256') == sources(), 'Reviewed source hashes do not match')
    policy_hash = sha256_file(args.policy)
    required_artifacts = {str(Path(value).resolve()) for value in (args.manifest, args.reference_cache,
                          args.reference_cache + '.config.json', args.reference_cache + '.completion.json')}
    require(required_artifacts.issubset(locked_policy['artifacts_sha256']), 'Lock manifest and existing reference cache/config/completion content hashes')
    for path, digest in locked_policy['artifacts_sha256'].items():
        ensure_data_disk(path)
        require(sha256_file(path) == digest, 'Bound original input/reference artifact changed: ' + path)
    ensure_data_disk(args.policy, args.model_path, args.adapter_path, args.manifest, args.image_root, args.output_dir)
    if args.reference_cache:
        ensure_data_disk(args.reference_cache)
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TMPDIR", "TORCH_HOME"):
        require(bool(os.environ.get(name)), "Missing data-disk environment: " + name)
        ensure_data_disk(os.environ[name])
    require(sha256_file(args.manifest) == args.expected_manifest_sha256, "Explicitly bound manifest SHA differs")
    require(math.isfinite(args.learning_rate) and args.learning_rate > 0 and args.save_steps > 0
            and 0 < args.wall_seconds <= 10800 and 0 < args.logp_atol <= .1, "Invalid optimization/runtime settings")
    if args.method == "forward_projection":
        require(args.beta is not None and math.isfinite(args.beta) and args.beta > 0, "Named KL method needs explicit positive beta; no silent unregularized fallback")
    else:
        require(args.beta is None or args.beta == 0, "Unregularized methods do not accept a hidden beta")
    beta = args.beta or 0.0
    records = validate_budget(read_jsonl(args.manifest), args.limit, args.max_steps)
    max_steps = args.limit // ACCUMULATION
    output = Path(args.output_dir)
    require(not output.exists(), "Fresh output directory required; no automatic resume or overwrite")
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + args.wall_seconds
    state = {"status": "preparing", "method": args.method, "mode": args.mode, "started_utc": now(), "global_step": 0,
             "actual_prompt_visits": 0, "no_resume": True, "test_access": False, "post_hoc_exploration": True}
    state_path = output / "state.json"
    write_json(state_path, state)

    def update(**changes):
        state.update(changes, updated_utc=now())
        write_json(state_path, state)

    def budget():
        require(time.monotonic() < deadline, "Finite-policy wall budget exhausted; preserve partial evidence, no automatic resume")

    try:
        import torch
        import numpy as np

        config = InputConfig(args.min_pixels, args.max_pixels, args.recipe_token_budget, args.max_input_tokens, 32)
        bound_sources = sources()
        pixels = pixel_identities(records, args.image_root)
        initial_identity = {"base": model_weight_identity(args.model_path), "adapter": model_weight_identity(args.adapter_path, adapter=True)}
        require(initial_identity == locked_policy['initial_identity'], 'Initial base/adapter differs from frozen SFT starting weights')
        reference_spec = {"schema_version": 1, "policy": "canonical24_full_sequence_plusEOS_temperature1",
                          "initial_identity": initial_identity, "input": asdict(config), "revision": args.revision,
                          "manifest_sha256": args.expected_manifest_sha256, "selected_records_sha256": hash_json(records),
                          "candidate_texts": CANDIDATES, "pixel_sha256": pixels,
                          "numerical_reference_atol": args.logp_atol,
                          "scoring_source_sha256": {key: value for key, value in legacy_sources().items() if not key.endswith("finite_policy_train.py")},
                          "versions": package_versions()}
        reference_path = Path(args.reference_cache) if args.reference_cache else output / "reference_policy.jsonl"
        reference_config_path = Path(str(reference_path) + ".config.json")
        reference_completion_path = Path(str(reference_path) + ".completion.json")
        reference_forward_candidates = 0

        def prepare(record):
            prepared, audit = prepare_record(record, processor, args.image_root, config)
            candidates = candidate_suffixes(processor, prepared["prompt"])
            encoded = encode_messages(processor, prepared["prompt"], load_images(record, args.image_root)).to(model.device)
            return audit, candidates, encoded

        if args.reference_cache:
            require(reference_config_path.is_file() and reference_completion_path.is_file(), "Reference cache lacks signed sidecars")
            cached_spec = json.loads(reference_config_path.read_text())
            require(cached_spec == json.loads(json.dumps(reference_spec)), "Reference cache has different weights/input/prefix/pixels/versions")
            completion = json.loads(reference_completion_path.read_text())
            require(completion.get("status") == "complete_fixed_initial_reference"
                    and completion.get("rows") == args.limit
                    and completion.get("cache_sha256") == sha256_file(reference_path), "Reference cache incomplete or changed")
            reference = read_jsonl(reference_path)
        require(len(reference) == args.limit, "Incomplete reference policy")
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
        qstars = tuple(make_qstar(ref['logq'], ref['exact_rewards'], beta) for ref in reference)
        qstar_path = output / 'frozen_qstar.jsonl'
        for record, ref, target in zip(records, reference, qstars):
            append(qstar_path, {'qid': record['qid'], 'recipe_id': record['recipe_id'], 'presentation_id': record['presentation_id'],
                   'source_split': 'train', 'qstar': list(target.q), 'qstar_logq': list(target.logq), 'reference_logq': list(target.reference_logq),
                   'exact_rewards': list(target.rewards), 'beta': target.beta, 'reference_policy_sha256': reference_hash,
                   'candidate_texts': CANDIDATES, 'canonical_completion_token_count_including_eos': canonical_token_count})
        qstar_hash = sha256_file(qstar_path)
        frozen_qstar_signature = hash_json([{'q': t.q, 'logq': t.logq, 'reference_logq': t.reference_logq, 'rewards': t.rewards, 'beta': t.beta} for t in qstars])
        # Complete CPU/source/cache/target and JSON-native-signature preflight
        # before the first processor/model allocation. The final signature adds
        # actual in-memory fingerprints after load; this preflight cannot claim
        # the real GPU backward or optimizer has been accepted yet.
        hash_json({'cli': json_cli(args), 'initial_identity': initial_identity, 'sources': bound_sources,
                   'frozen_qstar_signature': frozen_qstar_signature, 'reference_spec': reference_spec})
        budget()
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
        signature_payload = {"schema_version": 1, "method": args.method, "cli": json_cli(args), "resolved_beta": beta,
                             "actual_updates": max_steps, "accumulation": ACCUMULATION, "locked_policy_sha256": policy_hash,
                             "frozen_qstar_sha256": qstar_hash, "frozen_qstar_signature": frozen_qstar_signature, "input": asdict(config),
                             "canonical_completion_token_count": canonical_token_count,
                             "initial_identity": initial_identity, "initial_adapter_fingerprint": initial_adapter,
                             "initial_visual_fingerprint": initial_vision, "manifest_sha256": args.expected_manifest_sha256,
                             "selected_records_sha256": hash_json(records), "pixel_sha256": pixels,
                             "reference_policy_sha256": reference_hash, "reference_spec_sha256": hash_json(reference_spec),
                             "sources_sha256": bound_sources, "versions": package_versions(),
                             "optimizer": {"name": "AdamW", "lr": args.learning_rate, "betas": [.9, .999], "eps": 1e-8,
                                           "weight_decay": 0.0, "max_grad_norm": 1.0, "schedule": "constant", "fresh_state": True},
                             "mode": "eval with LoRA autograd; no KV cache; all sequence objectives divided by canonical token count",
                             "sampling": "None: all24 verified EOS-complete actions; frozen soft target derived once from existing qref and exact0/1 reward",
                             "objective": "beta KL(qstar || qtheta)/C; not raw sequence SFT and not a complete MPO implementation"}
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
        torch.cuda.reset_peak_memory_stats()

        def immutable():
            budget()
            for path, digest in locked_policy['artifacts_sha256'].items():
                require(sha256_file(path) == digest, 'Locked reference/data sidecar changed during training: ' + path)
            require(sources() == bound_sources and sha256_file(args.manifest) == args.expected_manifest_sha256
                    and sha256_file(reference_path) == reference_hash and sha256_file(args.policy) == policy_hash
                    and sha256_file(qstar_path) == qstar_hash
                    and hash_json([{'q': t.q, 'logq': t.logq, 'reference_logq': t.reference_logq, 'rewards': t.rewards, 'beta': t.beta} for t in qstars]) == frozen_qstar_signature, "Bound scientific source/data/reference changed during training")

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
                       "run_signature_sha256": signature, "reference_policy_sha256": reference_hash,
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
                record, ref = records[visit_index], reference[visit_index]
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
                samples, uniforms = None, None
                target = qstars[visit_index]
                require(tuple(rewards) == target.rewards and tuple(ref['logq']) == target.reference_logq and target.beta == beta,
                        'Frozen projection target provenance changed')
                terms = projection_terms(policy, target)
                count = validate_token_count(candidates)
                require(count == canonical_token_count, "Global EOS-complete loss scale C changed")
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
                visit = {"visit_index": visit_index, "visit_id": record.get("visit_id"), "qid": record["qid"],
                         "recipe_id": record["recipe_id"], "presentation_id": record["presentation_id"],
                         "source_split": "train", "global_step_before_update": step,
                         "input_hash": audit["input_hash"], "input_audit": audit,
                         "display_perm": record["display_perm"], "target_order": record["target_order"], "target_labels": record["target_labels"],
                         "candidate_texts": CANDIDATES, "candidate_token_ids": [candidate["token_ids"] for candidate in candidates],
                         "sequence_logps": scores, "token_logps_including_eos": token_logps, **policy,
                         "canonical_mass": math.exp(policy["logZ"]),
                         "exact_rewards": rewards, "reference_logq": ref["logq"], "reference_policy_sha256": reference_hash,
                         "sampled_indices": samples, "sampling_uniforms": uniforms, "sampling_rng_state_sha256": hash_json(rng.getstate()),
                         **terms, "completion_token_count": count, "nonzero_coefficients": sum(value != 0 for value in terms["coefficients"]),
                         "group_active": active, "backward_checks": backward_checks, "numerical_reference_checks": checks,
                         "run_signature_sha256": signature, "consumption_point": "all_selected_candidate_backward_calls_returned",
                         "policy_gradient_scope": "beta*(qstar-qtheta) ascent coefficients over24 EOS-complete canonical strings; forward-KL soft-target projection, no sampling",
                         "frozen_qstar_sha256": qstar_hash, "forward_projection_objective_after_C": terms["forward_projection_objective_before_C"] / count}
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
            entry = {"global_step_before_update": step, "global_step": step + 1,
                     "actual_prompt_visits": len(actual_visits), "active_groups": sum(group_active), "groups": ACCUMULATION,
                     "zero_reward_gradient_batch": before_clip["l2_norm"] == 0,
                     "before_clip": before_clip, "after_clip": after_clip, "actual_parameter_update_l2": update_norm,
                     "optimizer_step_executed": True, "optimizer_state_entries": len(optimizer.state),
                     "learning_rate": optimizer.param_groups[0]["lr"], "run_signature_sha256": signature,
                     "elapsed_seconds": time.monotonic() - step_started,
                     "cumulative_current_policy_candidate_forwards_including_numerical_checks": current_policy_forward_candidates,
                     "cumulative_gradient_candidate_forwards": gradient_forward_candidates,
                     "initial_reference_candidate_forwards_charged_this_run": reference_forward_candidates,
                     "cumulative_candidate_backwards": total_backward_candidates}
            append(output / "step_times.jsonl", entry)
            update(global_step=step + 1, actual_prompt_visits=len(actual_visits), last_step=entry)
            if (step + 1) % args.save_steps == 0 or step + 1 == max_steps:
                save_checkpoint(step + 1, output / f"checkpoint-{step + 1}")
            print(json.dumps({"step": step + 1, "planned": max_steps, "method": args.method,
                              "visits": len(actual_visits), "gradient_norm": before_clip["l2_norm"],
                              "active_groups": sum(group_active), "seconds": entry["elapsed_seconds"]}), flush=True)
        immutable()
        require(actual_visits == [(record["qid"], record["presentation_id"]) for record in records], "Actual fixed-prefix exposure differs")
        final_vision = frozen_vision_fingerprint(model)
        final_adapter = adapter_fingerprint(model)
        require(final_vision == initial_vision, "Frozen visual weights changed")
        require(pixel_identities(records, args.image_root) == pixels, "Selected training image bytes changed")
        model.save_pretrained(output / "adapter_final", safe_serialization=True)
        processor.save_pretrained(output / "adapter_final")
        outcome = {"status": "completed", "method": args.method, "global_step": max_steps,
                   "actual_prompt_visits": len(actual_visits), "unique_questions": len({record["qid"] for record in records}),
                   "unique_recipes": len({record["recipe_id"] for record in records}), "actual_visit_sequence_verified": True,
                   "effective_adapter_update": final_adapter != initial_adapter,
                   "scientific_outcome": "effective_training" if final_adapter != initial_adapter else "no_effective_adapter_update",
                   "initial_adapter_fingerprint": initial_adapter,
                   "final_adapter_fingerprint": final_adapter, "initial_visual_fingerprint": initial_vision,
                   "final_visual_fingerprint": final_vision, "visual_weights_unchanged": True,
                   "elapsed_seconds_including_reference": time.monotonic() - started,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                   "candidate_forwards": reference_forward_candidates + current_policy_forward_candidates + gradient_forward_candidates,
                   "reference_candidate_forwards_charged_this_run": reference_forward_candidates,
                   "current_policy_candidate_forwards_including_numerical_checks": current_policy_forward_candidates,
                   "gradient_candidate_forwards": gradient_forward_candidates,
                   "candidate_backwards": total_backward_candidates,
                   "reference_policy_sha256": reference_hash, "frozen_qstar_sha256": qstar_hash, "run_signature_sha256": signature,
                   "actual_visits_sha256": sha256_file(output / "actual_prompt_visits.jsonl"),
                   "step_times_sha256": sha256_file(output / "step_times.jsonl"), "checkpoints": checkpoints,
                   "final_adapter_identity": model_weight_identity(output / "adapter_final", adapter=True),
                   "resume_supported": False, "test_access": False}
        write_json(output / "train_outcome.json", outcome)
        update(status="completed", finished_utc=now(), outcome=outcome)
    except BaseException as error:
        update(status="failed_stopped_no_resume", error=f"{type(error).__name__}: {error}", finished_utc=now(),
               elapsed_seconds=time.monotonic() - started)
        raise


if __name__ == "__main__":
    main()
