#!/usr/bin/env python3
"""Independent stdlib verifier for frozen forward-projection smoke/formal runs.

Never imports the trainer, projection math, scorer, metrics or Torch. Reuses the
previous verifier's safe-container inspection structure; independently derives
qstar via the binary exponential-tilt normalizer and projection gradients. No
scientific gain is required. Failed/ineffective results remain valid evidence.
"""
from __future__ import annotations

import argparse
from array import array
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import permutations
import json
import hashlib
import importlib.metadata
import platform
import math
from pathlib import Path
import random
import struct
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

ACTIONS = tuple(" ".join(values) for values in permutations("ABCD"))
METHODS = ("forward_projection",)
SOURCE_SHA = {
    "exploratory/principled_ordering_20261105/finite_policy_train.py": "34e29cc48744626dec27f99afdd5bdf34660130cf45f17e6d3ff205786a446a5",
    "exploratory/principled_ordering_20261105/score_permutations.py": "affeda80416145d2b6a134172e812d15d48a86d77c67128bd2e5f84e0c341819",
    "exploratory/principled_ordering_20261105/gradient_alignment.py": "1adff082850e0b83a9a70c83565638c02b8efeebdf0fb0ade49af937df9fe41b",
    "src/recipeqa/modeling.py": "5fedb7a60877e965861aa9174179880754cb0555c4167265fdd8c3781af39771",
    "src/recipeqa/metrics.py": "5287d44a3bb4ede3a44ce0a6625aee399260afecd42c984441a8e07492a2423a",
}
STATE_NAMES = ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json")


@dataclass(frozen=True)
class InputConfig:
    min_pixels:int=65536
    max_pixels:int=262144
    recipe_token_budget:int=1536
    max_input_tokens:int=4096
    max_completion_tokens:int=32
    prompt_version:str='recipeqa-order-v1'


def hash_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def package_versions():
    result={'python':platform.python_version()}
    for name in ('torch','torchvision','transformers','trl','peft','accelerate','datasets','huggingface_hub','Pillow','numpy','tokenizers','safetensors','pyarrow'):
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return result


def model_weight_identity(directory,adapter=False):
    directory=Path(directory).resolve()
    patterns=('adapter_model*.safetensors','adapter_model*.bin') if adapter else ('model*.safetensors','pytorch_model*.bin')
    weights=sorted({p for pattern in patterns for p in directory.glob(pattern) if p.is_file()})
    require(bool(weights) and all(p.stat().st_size > 0 for p in weights),'Missing or empty model weight bytes')
    files=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in weights]
    names=('config.json','adapter_config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','chat_template.jinja','chat_template.json','preprocessor_config.json','processor_config.json')
    auxiliary=[{'name':name,'sha256':sha256_file(directory/name)} for name in names if (directory/name).is_file()]
    return {'directory':str(directory),'weight_files':files,'weights_sha256':hash_json(files),'auxiliary_files':auxiliary,'bundle_sha256':hash_json({'weights':files,'auxiliary':auxiliary})}


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


def independent_qstar(reference_logq, gold, beta):
    require(len(reference_logq) == 24 and 0 <= gold < 24 and math.isfinite(beta) and beta > 0, 'Invalid qstar input')
    require(all(math.isfinite(v) for v in reference_logq), 'Nonfinite reference logq')
    reference=[math.exp(v) for v in reference_logq]
    close(math.fsum(reference),1.,'Reference is not normalized',1e-9)
    # Binary tilt closed form, independent of the trainer's24-way normalizer.
    normalizer=1 + reference[gold]*math.expm1(1/beta)
    log_normalizer=math.log(normalizer)
    target_logq=[v + (1/beta if i == gold else 0) - log_normalizer for i,v in enumerate(reference_logq)]
    target=[math.exp(v) for v in target_logq]
    close(math.fsum(target),1.,'Independent qstar fails normalization',1e-9)
    return target,target_logq


def independent_terms(q, logq, gold, ref_logq, beta):
    target,target_logq=independent_qstar(ref_logq,gold,beta)
    coefficients=[beta*(target[i]-q[i]) for i in range(24)]
    close(math.fsum(coefficients),0.,'Projection gradient is not centered',1e-9)
    forward=math.fsum(p*(lp-lq) for p,lp,lq in zip(target,target_logq,logq))
    reverse=math.fsum(p*(lq-lp) for p,lq,lp in zip(q,logq,target_logq))
    reference_kl=math.fsum(p*(lq-lr) for p,lq,lr in zip(q,logq,ref_logq))
    return {'coefficients':coefficients,'qstar':target,'qstar_logq':target_logq,
            'forward_kl_qstar_current':forward,'reverse_kl_current_qstar':reverse,
            'forward_projection_objective_before_C':beta*forward,'exact_expected_reward':q[gold],
            'kl_q_reference':reference_kl,'regularized_expected_reward':q[gold]-beta*reference_kl,
            'closed_ce_nll':-logq[gold],'gold_index':gold,'active_beta':beta}


def check_policy_fields(row, computed):
    for key in ("q", "logq"):
        vector_close(row[key], computed[key], "Stored " + key + " differs from EOS-complete scores", 1e-9)
    close(row["logZ"], computed["logZ"], "Stored logZ differs", 1e-9)


def exact_labels(record):
    display,target=record['display_perm'],record['target_order']
    require(sorted(display) == sorted(target) == [0,1,2,3], 'Bad original target/display permutation')
    gold=' '.join(chr(65+display.index(identity)) for identity in target)
    require(record['target_labels'] == gold.split(), 'Original inverse target mapping differs')
    labels=[int(text == gold) for text in ACTIONS]
    require(sum(labels) == 1, 'Unique exact0/1 verifier failed')
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
    result.add_argument("--method", choices=METHODS, default="forward_projection")
    result.add_argument("--policy", type=Path, required=True)
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
    require(args.method == "forward_projection" and args.beta == 1.6363096678310884, "Expected projection/fixed beta mismatch")
    policy_path=data_path(args.policy)
    locked_policy=load(policy_path)
    require(locked_policy["status"] == "locked_before_projection_training" and locked_policy["method"] == args.method and locked_policy["reward"] == "unique_complete_order_exact_0_1", "Unfrozen or different child policy")
    require(locked_policy["mode"] == ("smoke" if args.limit == 4 else "formal"), "Wrong smoke/formal mode")
    policy_sha=sha256_file(policy_path)
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
    require(cli["max_steps"] == args.steps and cli["mode"] == locked_policy["mode"], "Truncated or wrong-mode run")
    require(cli["policy"] == str(policy_path) and payload["locked_policy_sha256"] == policy_sha, "Child policy signature differs")
    require({k:v for k,v in cli.items() if k != "policy"} == locked_policy["cli"], "Signed CLI differs from locked policy")
    require(cli["save_steps"] == (1 if args.steps == 1 else 16) and cli["logp_atol"] == .01,
            "Save interval or numerical tolerance differs from reviewed smoke/trial")
    require((cli["beta"] is None or cli["beta"] == 0) if args.beta == 0 else cli["beta"] == args.beta, "CLI KL coefficient differs")
    require(data_path(cli["manifest"]) == manifest and data_path(cli["adapter_path"]) == initial
            and data_path(cli["output_dir"]) == root, "CLI source/run paths differ")
    require(payload["manifest_sha256"] == cli["expected_manifest_sha256"] == args.expected_manifest_sha256, "Signed manifest differs")
    require(payload["sources_sha256"] == locked_policy["sources_sha256"] and all(locked_policy["sources_sha256"].get(k) == v for k,v in SOURCE_SHA.items()), "Run used unreviewed scientific sources")
    require(set(locked_policy["sources_sha256"]) == set(SOURCE_SHA) | {"research/projection_20261106/"+name for name in ("projection_train.py","projection_math.py","test_projection_math.py")}, "Projection source coverage differs")
    require(payload["input"] == asdict(InputConfig()), "Projection run changed frozen image/recipe/completion budgets")
    require(payload["optimizer"] == {"name": "AdamW", "lr": cli["learning_rate"], "betas": [.9, .999],
            "eps": 1e-8, "weight_decay": 0.0, "max_grad_norm": 1.0, "schedule": "constant", "fresh_state": True}
            and cli["learning_rate"] == 5e-6, "Optimizer metadata differs from reviewed fresh-state objective")
    for name, digest in locked_policy["sources_sha256"].items():
        require(sha256_file(data_path(ROOT / name)) == digest, "Current scientific source differs: " + name)
    for name,digest in locked_policy["artifacts_sha256"].items():
        require(sha256_file(data_path(name)) == digest,"Locked data/reference artifact differs: "+name)
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
    require(identities == payload["initial_identity"] == locked_policy["initial_identity"], "Actual initial base/adapter bytes differ")
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
    require(bool(cli['reference_cache']), 'Projection must reuse the signed original reference')
    qstar_file=root/'frozen_qstar.jsonl'
    qstar_rows=read_jsonl(qstar_file)
    require(len(qstar_rows) == args.limit and sha256_file(qstar_file) == payload['frozen_qstar_sha256'] == outcome['frozen_qstar_sha256'], 'Frozen target file incomplete or changed')
    target_signature=hash_json([{'q':r['qstar'],'logq':r['qstar_logq'],'reference_logq':r['reference_logq'],'rewards':r['exact_rewards'],'beta':r['beta']} for r in qstar_rows])
    require(target_signature == payload['frozen_qstar_signature'],'Frozen target object signature changed')
    reference_forwards = 0
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
        require(all(type(value) is int and value in (0,1) for value in ref["exact_rewards"] + visit["exact_rewards"]), "Reward is not integer Exact0/1")
        require(ref["exact_rewards"] == visit["exact_rewards"] == labels
                and ref["gold_candidate_index"] == gold, "Cached/actual verifier reward differs")
        audit_numerical_checks(ref["numerical_checks"], index == 0, .01, ref["token_logps"])
        audit_numerical_checks(visit["numerical_reference_checks"], index == 0, .01, visit["token_logps_including_eos"])
        current_forwards += 25
        require(visit['sampled_indices'] is None and visit['sampling_uniforms'] is None, 'Projection unexpectedly sampled')
        require(visit['sampling_rng_state_sha256'] == hash_json(rng.getstate()), 'No-sampling RNG state changed')
        terms=independent_terms(policy['q'],policy['logq'],gold,ref_policy['logq'],args.beta)
        frozen=qstar_rows[index]
        require(all(frozen[k] == record[k] for k in ['qid','recipe_id','presentation_id']) and frozen['source_split'] == 'train', 'Frozen target identity/order differs')
        require(all(type(value) is int and value in (0,1) for value in frozen['exact_rewards']), 'Frozen reward is not integer0/1')
        require(frozen['candidate_texts'] == list(ACTIONS) and frozen['exact_rewards'] == labels and frozen['beta'] == args.beta and frozen['reference_policy_sha256'] == reference_hash, 'Frozen target action/reward/reference differs')
        require(frozen['canonical_completion_token_count_including_eos'] == count and visit['frozen_qstar_sha256'] == payload['frozen_qstar_sha256'], 'Target C/hash differs')
        vector_close(frozen['reference_logq'],ref_policy['logq'],'Frozen qref differs',1e-9)
        for field in ['qstar','qstar_logq']:
            vector_close(frozen[field],terms[field],'Independent frozen '+field+' differs',1e-9)
            vector_close(visit[field],terms[field],'Independent visit '+field+' differs',1e-9)
        vector_close(visit['coefficients'],terms['coefficients'],'Independent projection coefficients differ',1e-9)
        for field in ['forward_kl_qstar_current','reverse_kl_current_qstar','forward_projection_objective_before_C','exact_expected_reward','kl_q_reference','regularized_expected_reward','closed_ce_nll','active_beta']:
            close(visit[field],terms[field],'Independent '+field+' differs',1e-9)
        close(visit['forward_projection_objective_after_C'],terms['forward_projection_objective_before_C']/count,'Forward objective divisor C differs',1e-9)
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
                root / "actual_prompt_visits.jsonl", root / "step_times.jsonl", ref_path, ref_config_path, ref_completion_path, qstar_file, policy_path]
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
    return {"status": "complete_verified_projection_run", "passed": True, "method": args.method, "seed": args.seed,
            "limit": args.limit, "optimizer_steps": args.steps, "beta": args.beta,
            "effective_adapter_update": effective, "ineffective_updates_are_scientifically_valid": True,
            "run_signature_sha256": signature, "forward_backward_counts": counts,
            "reference": {"path": str(ref_path), "sha256": reference_hash, "rows": len(references), "reused": bool(cli["reference_cache"])},
            "initial_adapter_fingerprint": initial_fingerprint, "final_adapter_fingerprint": final_fingerprint,
            "source_sha256": locked_policy["sources_sha256"], "locked_policy_sha256": policy_sha, "frozen_qstar_sha256": sha256_file(qstar_file), "manifest_sha256": args.expected_manifest_sha256,
            "required_artifacts": {str(path.relative_to(ROOT)): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in required},
            "provenance": {"verifier_sha256": sha256_file(__file__), "versions": package_versions(), "arguments": vars(args)},
            "limits": ["CPU verification recomputes mathematics and persisted artifacts, not actual VLM forwards or Adam tensor arithmetic.",
                       "Torch optimizer/scheduler/RNG archives are content-hashed and CRC-checked, never unpickled by this verifier.",
                       "Persisted LoRA tensors independently reproduce runtime adapter fingerprints; frozen vision relies on runtime guards plus unchanged initial base identity.",
                       "This child-run audit does not independently establish the separate outer-stage deadline, lock, or evaluation policy; audit those coordinator artifacts separately."]}


def main():
    args = parser().parse_args()
    try:
        result = verify(args)
    except BaseException as error:
        result = {"status": "failed_projection_verification", "passed": False,
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
