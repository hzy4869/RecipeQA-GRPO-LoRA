#!/usr/bin/env python3
"""Independent CPU verifier for completed finite-policy runs; no trainer import.

Reconstructs policies, KL/Exact derivatives, repeated-draw RLOO, RNG trajectory,
exposure and artifact identities. Never repairs evidence or requires a gain.
"""
from __future__ import annotations

import argparse
from array import array
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import permutations
import json
import math
from pathlib import Path
import random
import struct
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recipeqa.metrics import score_order
from recipeqa.modeling import InputConfig, hash_json, model_weight_identity, package_versions, read_jsonl, sha256_file, write_json

ACTIONS = tuple(" ".join(values) for values in permutations("ABCD"))
METHODS = ("raw_sft", "exact", "exact_kl", "rloo_kl")
SOURCE_SHA = {
    "exploratory/principled_ordering_20261105/finite_policy_train.py": "34e29cc48744626dec27f99afdd5bdf34660130cf45f17e6d3ff205786a446a5",
    "exploratory/principled_ordering_20261105/score_permutations.py": "affeda80416145d2b6a134172e812d15d48a86d77c67128bd2e5f84e0c341819",
    "exploratory/principled_ordering_20261105/gradient_alignment.py": "1adff082850e0b83a9a70c83565638c02b8efeebdf0fb0ade49af937df9fe41b",
    "src/recipeqa/modeling.py": "5fedb7a60877e965861aa9174179880754cb0555c4167265fdd8c3781af39771",
    "src/recipeqa/metrics.py": "5287d44a3bb4ede3a44ce0a6625aee399260afecd42c984441a8e07492a2423a",
}
STATE_NAMES = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, atol=1e-10, rtol=1e-8):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isfinite(expected) and math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), message)


def vector_close(actual, expected, message, atol=1e-10):
    require(isinstance(actual, list) and len(actual) == len(expected), message + ": length")
    for a, b in zip(actual, expected):
        close(a, b, message, atol)


def load(path):
    return json.loads(Path(path).read_text())


def data_path(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT) and path.is_relative_to("/root/autodl-tmp"), "Path escaped project data disk: " + str(path))
    return path


def recompute_policy(token_logps, candidates):
    require(len(token_logps) == len(candidates) == 24, "Need complete 24-action token scores")
    require(tuple(row["text"] for row in candidates) == ACTIONS, "Candidate text/order differs")
    lengths, eos_ids, scores = set(), set(), []
    for values, candidate in zip(token_logps, candidates):
        ids = candidate["token_ids"]
        require(isinstance(ids, list) and ids and all(type(value) is int and value >= 0 for value in ids), "Bad token IDs")
        require(candidate["token_count"] == len(ids) == len(values)
                and ids[-1] == candidate["eos_token_id"], "Incomplete EOS-complete action")
        require(all(isinstance(value, (int, float)) and math.isfinite(value) and value <= 1e-6 for value in values),
                "Invalid token log probabilities")
        lengths.add(len(ids)); eos_ids.add(ids[-1]); scores.append(math.fsum(values))
    require(len(lengths) == len(eos_ids) == 1, "Unequal canonical suffix length or EOS IDs")
    peak = max(scores)
    partition = peak + math.log(math.fsum(math.exp(value - peak) for value in scores))
    require(partition <= 1e-4, "Canonical complete-string probability exceeds one")
    logs = [value - partition for value in scores]
    probabilities = [math.exp(value) for value in logs]
    close(math.fsum(probabilities), 1., "Independent softmax does not normalize")
    return {"scores": scores, "logq": logs, "q": probabilities, "logZ": partition, "token_count": next(iter(lengths))}


def independent_terms(method, q, logq, gold, ref_logq, beta, samples=None):
    """Different derivation: Exact derivative minus explicit centered KL derivative."""
    require(method in METHODS and len(q) == len(logq) == len(ref_logq) == 24 and 0 <= gold < 24, "Bad objective inputs")
    require(math.isfinite(beta) and (beta > 0 if method.endswith("_kl") else beta == 0), "Method/beta mismatch")
    ratio = [current - reference for current, reference in zip(logq, ref_logq)]
    divergence = math.fsum(probability * value for probability, value in zip(q, ratio))
    utilities = [float(index == gold) - beta * value for index, value in enumerate(ratio)]
    average = math.fsum(probability * value for probability, value in zip(q, utilities))
    if method == "raw_sft":
        require(samples is None, "Raw SFT must not sample")
        coefficients = [float(index == gold) for index in range(24)]
        advantages = None
    elif method in ("exact", "exact_kl"):
        require(samples is None, "Enumerated objective must not sample")
        coefficients = [q[gold] * (float(index == gold) - probability)
                        - beta * probability * (ratio[index] - divergence)
                        for index, probability in enumerate(q)]
        advantages = None
    else:
        require(isinstance(samples, list) and len(samples) == 8
                and all(type(index) is int and 0 <= index < 24 for index in samples), "Need all eight RLOO draws, including repeats")
        values = [utilities[index] for index in samples]
        advantages = ([0.] * 8 if len(set(values)) == 1 else
                      [value - math.fsum(other for j, other in enumerate(values) if j != i) / 7
                       for i, value in enumerate(values)])
        coefficients = [math.fsum(advantage for action, advantage in zip(samples, advantages) if action == index) / 8
                        for index in range(24)]
    return {"coefficients": coefficients, "u": utilities, "mean_u": average, "advantages": advantages,
            "exact_expected_reward": q[gold], "kl_q_reference": divergence,
            "regularized_expected_reward": q[gold] - beta * divergence,
            "closed_ce_nll": -logq[gold], "gold_index": gold, "active_beta": beta}


def replay_draws(q, rng):
    """Independent inverse-CDF replay, using strict upper boundaries."""
    require(len(q) == 24 and all(value >= 0 and math.isfinite(value) for value in q), "Bad replay policy")
    close(math.fsum(q), 1., "Replay policy does not normalize")
    uniforms = [rng.random() for _ in range(8)]
    indices = []
    for uniform in uniforms:
        total = 0.
        selected = None
        for index, probability in enumerate(q):
            total = 1. if index == 23 else total + probability
            if uniform < total:
                selected = index
                break
        require(selected is not None and q[selected] > 0, "Replay selected an unsupported action")
        indices.append(selected)
    return indices, uniforms


def check_policy_fields(row, computed):
    for key in ("q", "logq"):
        vector_close(row[key], computed[key], "Stored " + key + " differs from EOS-complete scores", 1e-9)
    close(row["logZ"], computed["logZ"], "Stored logZ differs", 1e-9)


def exact_labels(record):
    labels = [score_order(text, record["display_perm"], record["target_order"])["exact"] for text in ACTIONS]
    require(sum(labels) == 1 and ACTIONS[labels.index(1)] == " ".join(record["target_labels"]), "Source target mapping contradicts gold text")
    return labels


def audit_numerical_checks(checks, first, atol, token_logps):
    require(isinstance(checks, list) and len(checks) == 1, "Microbatch-one score must contain one numerical reference")
    check = checks[0]
    require(check["candidate_index"] == 0 and check["literal_all_position_logits"] is first,
            "Wrong literal-reference schedule")
    require(isinstance(check["max_token_logp_abs_difference"], (int, float))
            and 0 <= check["max_token_logp_abs_difference"] <= atol, "Numerical reference exceeded locked tolerance")
    values = check["reference_token_logps"]
    require(len(values) == len(token_logps[0]) and all(math.isfinite(value) for value in values), "Numerical reference token logps are incomplete")
    close(check["max_token_logp_abs_difference"], max(abs(a - b) for a, b in zip(values, token_logps[0])),
          "Numerical reference delta does not match saved token scores", 1e-9)


def tensor_fingerprint(directory, inventory):
    """Reproduce runtime adapter_fingerprint from safe tensor bytes, no pickle/Torch."""
    import hashlib

    path = directory / "adapter_model.safetensors"
    with path.open("rb") as handle:
        prefix = handle.read(8)
        require(len(prefix) == 8, "Truncated safetensors file")
        header_size = struct.unpack("<Q", prefix)[0]
        require(0 < header_size <= 16 * 1024**2, "Invalid safetensors header size")
        header = json.loads(handle.read(header_size))
        tensors = {name: value for name, value in header.items() if name != "__metadata__"}
        expected_names = {row["name"].replace(".default.", ".") for row in inventory}
        require(set(tensors) == expected_names, "Saved adapter has missing/additional LoRA tensors")
        intervals = sorted(entry["data_offsets"] for entry in tensors.values())
        require(intervals and intervals[0][0] == 0
                and all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
                and intervals[-1][1] == path.stat().st_size - 8 - header_size,
                "Safetensors data ranges overlap, have gaps or leave unchecked bytes")
        digest = hashlib.sha256()
        for row in inventory:
            name = row["name"].replace(".default.", ".")
            entry = tensors[name]
            require(entry["shape"] == row["shape"], "Saved LoRA tensor shape differs")
            begin, end = entry["data_offsets"]
            dtype = entry["dtype"]
            size = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8}.get(dtype)
            require(size is not None and 0 <= begin <= end and end - begin == row["numel"] * size,
                    "Saved LoRA tensor dtype/size differs")
            handle.seek(8 + header_size + begin)
            raw = handle.read(end - begin)
            require(len(raw) == end - begin, "Truncated LoRA tensor")
            if dtype == "F32":
                converted = raw
            elif dtype == "BF16":
                converted = b"".join(struct.pack("<I", value[0] << 16) for value in struct.iter_unpack("<H", raw))
            else:
                converted = b"".join(struct.pack("<f", value[0]) for value in struct.iter_unpack("<e" if dtype == "F16" else "<d", raw))
            values = array("f"); values.frombytes(converted)
            if sys.byteorder != "little":
                values.byteswap()
            require(all(math.isfinite(value) for value in values), "Non-finite saved LoRA tensor")
            digest.update(row["name"].encode()); digest.update(converted)
    return digest.hexdigest()


def audit_state_file(path):
    require(path.is_file() and path.stat().st_size > 0 and zipfile.is_zipfile(path), "Missing/unrecognized saved Torch state: " + str(path))
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None and any(name.endswith("data.pkl") for name in archive.namelist()),
                "Damaged or incomplete saved Torch state: " + str(path))


def audit_effective_update(initial_fingerprint, final_fingerprint, disclosed, scientific_outcome):
    effective = initial_fingerprint != final_fingerprint
    require(disclosed is effective and scientific_outcome
            == ("effective_training" if effective else "no_effective_adapter_update"), "Effective-update disclosure differs")
    return effective


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--expected-manifest-sha256", required=True)
    result.add_argument("--initial-adapter", type=Path, required=True)
    result.add_argument("--method", choices=METHODS, required=True)
    result.add_argument("--seed", type=int, required=True)
    result.add_argument("--limit", type=int, required=True)
    result.add_argument("--steps", type=int, required=True)
    result.add_argument("--beta", type=float, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def verify(args):
    root, manifest, initial, output = map(data_path, (args.run_dir, args.manifest, args.initial_adapter, args.output))
    require(ROOT.stat().st_dev != Path("/").stat().st_dev, "Project is not on a separately mounted data disk")
    require(args.limit in (4, 256) and args.steps == args.limit // 4, "Only fixed four-question smoke or 256-question trial is authorized")
    require(math.isfinite(args.beta) and (args.beta > 0 if args.method.endswith("_kl") else args.beta == 0), "Expected method/beta mismatch")
    require(sha256_file(manifest) == args.expected_manifest_sha256, "Expected manifest bytes differ")
    require(not output.exists(), "Verifier output exists; preserve prior audit")
    signature_file = root / "run_signature.json"
    signed = load(signature_file)
    signature, payload = signed["sha256"], signed["payload"]
    require(signature == hash_json(payload), "Run signature is internally inconsistent")
    cli = payload["cli"]
    require(payload["method"] == cli["method"] == args.method and cli["seed"] == args.seed
            and cli["limit"] == args.limit and payload["actual_updates"] == args.steps
            and payload["accumulation"] == 4 and payload["resolved_beta"] == args.beta, "Expected method/seed/budget/beta differs")
    require(cli["max_steps"] in (None, args.steps) and not cli["prepare_reference_only"], "Reference-only or truncated run is not training completion")
    require(cli["save_steps"] == (1 if args.steps == 1 else 16) and cli["logp_atol"] == .01,
            "Save interval or numerical tolerance differs from reviewed smoke/trial")
    require((cli["beta"] is None or cli["beta"] == 0) if args.beta == 0 else cli["beta"] == args.beta, "CLI KL coefficient differs")
    require(data_path(cli["manifest"]) == manifest and data_path(cli["adapter_path"]) == initial
            and data_path(cli["output_dir"]) == root, "CLI source/run paths differ")
    require(payload["manifest_sha256"] == cli["expected_manifest_sha256"] == args.expected_manifest_sha256, "Signed manifest differs")
    require(payload["sources_sha256"] == SOURCE_SHA, "Run used unreviewed scientific sources")
    require(payload["input"] == asdict(InputConfig()), "Finite run changed frozen image/recipe/completion budgets")
    require(payload["optimizer"] == {"name": "AdamW", "lr": cli["learning_rate"], "betas": [.9, .999],
            "eps": 1e-8, "weight_decay": 0.0, "max_grad_norm": 1.0, "schedule": "constant", "fresh_state": True}
            and math.isfinite(cli["learning_rate"]) and cli["learning_rate"] > 0, "Optimizer metadata differs from reviewed fresh-state objective")
    for name, digest in SOURCE_SHA.items():
        require(sha256_file(data_path(ROOT / name)) == digest, "Current scientific source differs: " + name)
    require(payload["versions"] == package_versions(), "Audit environment differs from recorded runtime versions")
    state, outcome = load(root / "state.json"), load(root / "train_outcome.json")
    require(state["status"] == outcome["status"] == "completed" and state["outcome"] == outcome, "Run/state did not complete consistently")
    require(outcome["method"] == args.method and outcome["global_step"] == state["global_step"] == args.steps
            and outcome["actual_prompt_visits"] == state["actual_prompt_visits"] == args.limit
            and outcome["run_signature_sha256"] == state["run_signature_sha256"] == signature,
            "Completed run identity or coverage differs")
    require(outcome["actual_visit_sequence_verified"] is True and outcome["resume_supported"] is False
            and outcome["test_access"] is False and state["test_access"] is False and state["no_resume"] is True,
            "Completion scope changed")
    records = read_jsonl(manifest)[:args.limit]
    require(len(records) == args.limit and len({(r["qid"], r["presentation_id"]) for r in records}) == args.limit,
            "Selected training prefix is incomplete or duplicated")
    require(all(row["source_split"] == "train" and row["technical_status"] == "ok"
                and row["manifest_set"] not in ("val", "test", "orbit", "process") for row in records), "Held-out/damaged input")
    require(hash_json(records) == payload["selected_records_sha256"], "Signed selected records differ")
    identities = {"base": model_weight_identity(data_path(cli["model_path"])),
                  "adapter": model_weight_identity(initial, adapter=True)}
    require(identities == payload["initial_identity"], "Actual initial base/adapter bytes differ")
    image_root = data_path(cli["image_root"])
    pixels = {}
    for row in records:
        for relative in row["image_paths_display"]:
            path = data_path(image_root / relative)
            require(path.is_relative_to(image_root), "Image escaped declared image root")
            pixels[relative] = sha256_file(path)
    require(pixels == payload["pixel_sha256"], "Actual training image bytes differ")
    inventory_record = load(root / "trainable_parameters.json")
    inventory = inventory_record["parameters"]
    require(len(inventory) == len({row["name"] for row in inventory}) == 392
            and inventory_record["trainable_parameters"] == sum(row["numel"] for row in inventory) == 17432576,
            "LoRA tensor inventory/element count differs")
    for row in inventory:
        require("lora_" in row["name"] and ".language_model.layers." in row["name"]
                and math.prod(row["shape"]) == row["numel"], "Unapproved LoRA parameter or shape")
    initial_fingerprint = tensor_fingerprint(initial, inventory)
    require(initial_fingerprint == payload["initial_adapter_fingerprint"] == outcome["initial_adapter_fingerprint"],
            "Initial in-memory adapter fingerprint does not match actual safe tensor bytes")
    for field in ("initial_visual_fingerprint", "final_visual_fingerprint"):
        require(outcome[field] == payload["initial_visual_fingerprint"] and isinstance(outcome[field], str)
                and len(outcome[field]) == 64, "Runtime frozen-vision fingerprint changed or is missing")
    require(outcome["visual_weights_unchanged"] is True, "Runtime visual-freeze guard failed")
    ref_path = data_path(cli["reference_cache"]) if cli["reference_cache"] else root / "reference_policy.jsonl"
    ref_config_path, ref_completion_path = Path(str(ref_path) + ".config.json"), Path(str(ref_path) + ".completion.json")
    spec, ref_completion = load(ref_config_path), load(ref_completion_path)
    reference_hash = sha256_file(ref_path)
    require(ref_completion["status"] == "complete_fixed_initial_reference" and ref_completion["rows"] == args.limit
            and ref_completion["cache_sha256"] == reference_hash
            and ref_completion["reference_spec_sha256"] == payload["reference_spec_sha256"] == hash_json(spec),
            "Reference cache lacks complete, matching signed sidecars")
    expected_spec = {"schema_version": 1, "policy": "canonical24_full_sequence_plusEOS_temperature1",
                     "initial_identity": identities, "input": payload["input"], "revision": cli["revision"],
                     "manifest_sha256": args.expected_manifest_sha256, "selected_records_sha256": hash_json(records),
                     "candidate_texts": list(ACTIONS), "pixel_sha256": pixels, "numerical_reference_atol": .01,
                     "scoring_source_sha256": {name: value for name, value in SOURCE_SHA.items() if not name.endswith("finite_policy_train.py")},
                     "versions": payload["versions"]}
    require(spec == expected_spec and payload["reference_policy_sha256"] == outcome["reference_policy_sha256"]
            == state["reference_policy_sha256"] == reference_hash, "Fixed reference identity differs")
    references, visits, steps = (read_jsonl(path) for path in (ref_path, root / "actual_prompt_visits.jsonl", root / "step_times.jsonl"))
    require(len(references) == len(visits) == args.limit and len(steps) == args.steps, "Reference/visit/update coverage is incomplete")
    reference_forwards = 0 if cli["reference_cache"] else 25 * args.limit
    current_forwards = backwards = 0
    rng = random.Random(args.seed)
    active_by_step = [0] * args.steps
    counts_by_step = []
    constant_count = None
    for index, (record, ref, visit) in enumerate(zip(records, references, visits)):
        expected_identity = tuple(record[field] for field in ("qid", "recipe_id", "presentation_id"))
        require(tuple(ref[field] for field in ("qid", "recipe_id", "presentation_id")) == expected_identity
                == tuple(visit[field] for field in ("qid", "recipe_id", "presentation_id")), "Reference/visit order or identity differs")
        require(ref["index"] == visit["visit_index"] == index and visit["visit_id"] == record.get("visit_id")
                and visit["global_step_before_update"] == index // 4, "Visit/sample/update boundary differs")
        require(ref["source_split"] == visit["source_split"] == "train", "Held-out row in actual trajectory")
        for field in ("display_perm", "target_order", "target_labels"):
            require(record[field] == ref[field] == visit[field], "Actual verifier source mapping differs")
        require(ref["reference_spec_sha256"] == hash_json(spec) and visit["reference_policy_sha256"] == reference_hash
                and visit["run_signature_sha256"] == signature, "Trajectory reference/run identity differs")
        require(ref["input_audit"] == visit["input_audit"] and visit["input_hash"] == ref["input_audit"]["input_hash"],
                "Actual input differs from fixed reference")
        audit = ref["input_audit"]
        require(tuple(audit[field] for field in ("qid", "recipe_id", "presentation_id")) == expected_identity
                and audit["input_tokens"] <= payload["input"]["max_input_tokens"]
                and audit["recipe_tokens_used"] <= payload["input"]["recipe_token_budget"]
                and audit["titles_retained"] == audit["steps"] == len(record["steps"]), "Input budget or title preservation failed")
        require(tuple(visit["candidate_texts"]) == ACTIONS
                and visit["candidate_token_ids"] == [candidate["token_ids"] for candidate in ref["candidates"]], "Actual action grammar differs")
        ref_policy = recompute_policy(ref["token_logps"], ref["candidates"])
        policy = recompute_policy(visit["token_logps_including_eos"], ref["candidates"])
        check_policy_fields(ref, ref_policy); check_policy_fields(visit, policy)
        vector_close(visit["sequence_logps"], policy["scores"], "Sequence score not token sum", 1e-6)
        vector_close(visit["reference_logq"], ref_policy["logq"], "Actual reference logq differs", 1e-9)
        close(visit["canonical_mass"], math.exp(policy["logZ"]), "Canonical mass differs")
        count = policy["token_count"]
        constant_count = count if constant_count is None else constant_count
        require(count == constant_count == payload["canonical_completion_token_count"] == visit["completion_token_count"],
                "Sequence loss divisor C changed")
        labels = exact_labels(record); gold = labels.index(1)
        require(ref["exact_rewards"] == visit["exact_rewards"] == labels
                and ref["gold_candidate_index"] == gold, "Cached/actual verifier reward differs")
        audit_numerical_checks(ref["numerical_checks"], index == 0, .01, ref["token_logps"])
        audit_numerical_checks(visit["numerical_reference_checks"], index == 0, .01, visit["token_logps_including_eos"])
        current_forwards += 25
        samples = None
        if args.method == "rloo_kl":
            samples, uniforms = replay_draws(policy["q"], rng)
            require(visit["sampled_indices"] == samples and visit["sampling_uniforms"] == uniforms,
                    "RLOO draws differ from exact seeded replay; no resampling is allowed")
        else:
            require(visit["sampled_indices"] is None and visit["sampling_uniforms"] is None, "Enumerated/raw method unexpectedly sampled")
        require(visit["sampling_rng_state_sha256"] == hash_json(rng.getstate()), "Sampling RNG state trajectory differs")
        terms = independent_terms(args.method, policy["q"], policy["logq"], gold, ref_policy["logq"], args.beta, samples)
        for field in ("coefficients", "u"):
            vector_close(visit[field], terms[field], "Independent " + field + " differs")
        require((visit["advantages"] is None) == (terms["advantages"] is None), "Wrong LOO baseline type")
        if terms["advantages"] is not None:
            vector_close(visit["advantages"], terms["advantages"], "Repeated-sample LOO advantages differ")
        for field in ("mean_u", "exact_expected_reward", "kl_q_reference", "regularized_expected_reward", "closed_ce_nll", "active_beta"):
            close(visit[field], terms[field], "Independent " + field + " differs")
        require(visit["gold_index"] == gold, "Wrong actual gold index")
        # Machine-zero selection is audited literally; no new approximation or
        # threshold changes which tiny stored coefficients trigger a backward.
        selected = [i for i, value in enumerate(visit["coefficients"]) if value != 0]
        require(visit["nonzero_coefficients"] == len(selected) and visit["group_active"] == bool(selected)
                and [check["candidate_index"] for check in visit["backward_checks"]] == selected,
                "Actual backward schedule does not match verified coefficients")
        require(all(isinstance(check["max_token_logp_difference"], (int, float))
                    and 0 <= check["max_token_logp_difference"] <= .01 for check in visit["backward_checks"]), "Gradient forward mismatch")
        require(visit["consumption_point"] == "all_selected_candidate_backward_calls_returned", "Visit is not a completed backward consumption")
        backwards += len(selected); active_by_step[index // 4] += bool(selected)
        if (index + 1) % 4 == 0:
            counts_by_step.append((current_forwards, backwards))
    for index, step in enumerate(steps):
        require(step["global_step_before_update"] == index and step["global_step"] == index + 1
                and step["actual_prompt_visits"] == (index + 1) * 4 and step["groups"] == 4
                and step["active_groups"] == active_by_step[index] and step["optimizer_step_executed"] is True
                and step["optimizer_state_entries"] == 392 and step["run_signature_sha256"] == signature, "Optimizer step coverage/identity differs")
        for position in ("before_clip", "after_clip"):
            gradient = step[position]
            require(gradient["trainable_gradient_tensors"] == 392 and 0 <= gradient["nonzero_gradient_tensors"] <= 392
                    and math.isfinite(gradient["l2_norm"]) and gradient["l2_norm"] >= 0, "Invalid recorded accumulated gradient")
        require(step["after_clip"]["l2_norm"] <= 1.0001, "Gradient clipping did not enforce the locked bound")
        require(step["zero_reward_gradient_batch"] == (step["before_clip"]["l2_norm"] == 0), "Wrong zero-gradient batch flag")
        require(math.isfinite(step["actual_parameter_update_l2"]) and step["actual_parameter_update_l2"] >= 0
                and math.isfinite(step["elapsed_seconds"]) and step["elapsed_seconds"] > 0, "Invalid update/time diagnostic")
        close(step["learning_rate"], cli["learning_rate"], "Learning rate changed")
        forwards, back = counts_by_step[index]
        require(step["cumulative_current_policy_candidate_forwards_including_numerical_checks"] == forwards
                and step["cumulative_gradient_candidate_forwards"] == step["cumulative_candidate_backwards"] == back
                and step["initial_reference_candidate_forwards_charged_this_run"] == reference_forwards, "Per-step compute counters differ")
    require(outcome["actual_visits_sha256"] == sha256_file(root / "actual_prompt_visits.jsonl")
            and outcome["step_times_sha256"] == sha256_file(root / "step_times.jsonl"), "Final trajectory hashes differ")
    counts = {"reference_candidate_forwards_charged_this_run": reference_forwards,
              "current_policy_candidate_forwards_including_numerical_checks": current_forwards,
              "gradient_candidate_forwards": backwards, "candidate_backwards": backwards,
              "candidate_forwards": reference_forwards + current_forwards + backwards}
    require(all(outcome[field] == value for field, value in counts.items()), "Final forward/backward counters differ")
    expected_checkpoint_steps = list(range(cli["save_steps"], args.steps + 1, cli["save_steps"]))
    if args.steps not in expected_checkpoint_steps:
        expected_checkpoint_steps.append(args.steps)
    require([item["step"] for item in outcome["checkpoints"]] == expected_checkpoint_steps
            and {path.name for path in root.glob("checkpoint-*") if path.is_dir()}
            == {f"checkpoint-{step}" for step in expected_checkpoint_steps}, "Checkpoint schedule differs")
    required = [signature_file, root / "state.json", root / "train_outcome.json", root / "trainable_parameters.json",
                root / "actual_prompt_visits.jsonl", root / "step_times.jsonl", ref_path, ref_config_path, ref_completion_path]
    last_fingerprint = None
    for item in outcome["checkpoints"]:
        step, directory = item["step"], data_path(item["path"])
        require(directory == root / f"checkpoint-{step}", "Checkpoint escaped the audited run")
        audit_path = directory / "finite_policy_checkpoint_audit.json"
        audit, saved = load(audit_path), load(directory / "trainer_state.json")
        identity = model_weight_identity(directory, adapter=True)
        require(identity == item["adapter_identity"] == audit["adapter_identity"]
                and audit["global_step"] == saved["global_step"] == step
                and audit["run_signature_sha256"] == saved["run_signature_sha256"] == signature
                and audit["reference_policy_sha256"] == reference_hash
                and saved["actual_prompt_visits"] == step * 4 and saved["resume_supported"] is False,
                "Checkpoint identity/state differs")
        close(saved["epoch"], step / args.steps, "Checkpoint epoch differs")
        require(set(audit["state_sha256"]) == set(STATE_NAMES), "Checkpoint state hash coverage is incomplete")
        for name in STATE_NAMES:
            path = directory / name
            require(sha256_file(path) == audit["state_sha256"][name], "Checkpoint state bytes differ: " + name)
            if name != "trainer_state.json":
                audit_state_file(path)
            required.append(path)
        last_fingerprint = tensor_fingerprint(directory, inventory)
        required.extend([audit_path, directory / "adapter_model.safetensors", directory / "adapter_config.json"])
    final_dir = root / "adapter_final"
    final_identity = model_weight_identity(final_dir, adapter=True)
    final_fingerprint = tensor_fingerprint(final_dir, inventory)
    require(final_identity == outcome["final_adapter_identity"] and final_fingerprint == outcome["final_adapter_fingerprint"]
            == last_fingerprint, "Final safe tensor bytes differ from runtime fingerprint or last checkpoint")
    effective = audit_effective_update(initial_fingerprint, final_fingerprint, outcome["effective_adapter_update"], outcome["scientific_outcome"])
    require(outcome["unique_questions"] == len({row["qid"] for row in records})
            and outcome["unique_recipes"] == len({row["recipe_id"] for row in records}), "Unique question/recipe counts differ")
    required.extend([final_dir / "adapter_model.safetensors", final_dir / "adapter_config.json"])
    return {"status": "complete_verified_finite_policy_run", "passed": True, "method": args.method, "seed": args.seed,
            "limit": args.limit, "optimizer_steps": args.steps, "beta": args.beta,
            "effective_adapter_update": effective, "ineffective_updates_are_scientifically_valid": True,
            "run_signature_sha256": signature, "forward_backward_counts": counts,
            "reference": {"path": str(ref_path), "sha256": reference_hash, "rows": len(references), "reused": bool(cli["reference_cache"])},
            "initial_adapter_fingerprint": initial_fingerprint, "final_adapter_fingerprint": final_fingerprint,
            "source_sha256": SOURCE_SHA, "manifest_sha256": args.expected_manifest_sha256,
            "required_artifacts": {str(path.relative_to(ROOT)): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in required},
            "provenance": {"verifier_sha256": sha256_file(__file__), "versions": package_versions(), "arguments": vars(args)},
            "limits": ["CPU verification recomputes mathematics and persisted artifacts, not actual VLM forwards or Adam tensor arithmetic.",
                       "Torch optimizer/scheduler/RNG archives are content-hashed and CRC-checked, never unpickled by this verifier.",
                       "Persisted LoRA tensors independently reproduce runtime adapter fingerprints; frozen vision relies on runtime guards plus unchanged initial base identity."]}


def main():
    args = parser().parse_args()
    try:
        result = verify(args)
    except BaseException as error:
        result = {"status": "failed_finite_policy_verification", "passed": False,
                  "error": f"{type(error).__name__}: {error}", "verifier_sha256": sha256_file(__file__)}
        if not args.output.exists():
            write_json(data_path(args.output), result)
        raise
    result["verified_utc"] = datetime.now(timezone.utc).isoformat()
    # Path arguments are provenance strings, never JSON-encoded Path objects.
    result["provenance"]["arguments"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "passed", "method", "optimizer_steps", "effective_adapter_update")}))


if __name__ == "__main__":
    main()
