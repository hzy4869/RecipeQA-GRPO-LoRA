#!/usr/bin/env python3
"""CPU-only global positive-beta calibration on the first64 training references.

Never changes reference files, trains, reads heldout data or silently falls back
to beta=0. The requested KL radius calibrates a full-information ideal policy;
a fixed penalty does not guarantee actual parameter-training KL stays below it.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANDIDATES = tuple(" ".join(order) for order in permutations("ABCD"))
EXPECTED_ROWS = 256
PREFIX = 64
sys.path.insert(0, str(ROOT / "src"))
from recipeqa.metrics import score_order


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def receipt(path):
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def safe_input(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "Input escaped mounted project data root")
    require(not any(part.lower() == "test" or part.lower().startswith(("test.", "test_", "confirm"))
                    for part in path.parts), "Heldout/test/confirmation input is forbidden")
    return path


def normalize(logits):
    """Centered log-softmax avoids subtracting a large absolute partition."""
    require(len(logits) == 24 and all(type(value) in (int, float) and math.isfinite(value) for value in logits),
            "Need24finite log scores")
    maximum = max(logits)
    centered = [value - maximum for value in logits]
    logsum = math.log(math.fsum(math.exp(value) for value in centered))
    logq = [value - logsum for value in centered]
    return logq, [math.exp(value) for value in logq], maximum + logsum


def improve(logq_ref, rewards, inverse_beta):
    require(math.isfinite(inverse_beta) and inverse_beta >= 0, "Invalid inverse beta")
    require(len(rewards) == 24 and sum(rewards) == 1 and set(rewards) <= {0, 1}, "Missing unique exact target")
    normalized_ref, _, _ = normalize(logq_ref)
    require(max(abs(a - b) for a, b in zip(normalized_ref, logq_ref)) <= 1e-9, "Reference logq is not normalized")
    logq, q, _ = normalize([value + inverse_beta * reward for value, reward in zip(logq_ref, rewards)])
    kl_raw = math.fsum(p * (current - original) for p, current, original in zip(q, logq, logq_ref))
    require(math.isfinite(kl_raw) and kl_raw >= -1e-10, "Invalid KL after full-information update")
    return {"logq_beta": logq, "q_beta": q, "kl": max(0.0, kl_raw),
            "kl_roundoff_clamped": kl_raw < 0, "exact_expected_reward": q[rewards.index(1)]}


def calibrate(logq_bank, reward_bank, delta):
    require(len(logq_bank) == len(reward_bank) == PREFIX, "Calibration uses exactly the first64 references")
    require(type(delta) in (int, float) and math.isfinite(delta), "Nonfinite KL radius")
    for logq, rewards in zip(logq_bank, reward_bank):
        improve(logq, rewards, 0.0)
    maximum_kl = math.fsum(-logq[rewards.index(1)] for logq, rewards in zip(logq_bank, reward_bank)) / PREFIX
    require(math.isfinite(maximum_kl) and maximum_kl >= 0, "Invalid limiting KL")
    base = {"delta": delta, "D_max": maximum_kl, "prefix": PREFIX, "beta": None,
            "fixed_penalty_warning": "Calibration of ideal full-information q_beta does not guarantee actual parameter-training KL<=delta"}
    if delta <= 0 or delta >= maximum_kl:
        return {**base, "status": "non_binding_or_boundary_needs_policy_review", "exit_code": 2,
                "boundary_reason": "non_positive_radius" if delta <= 0 else "radius_at_or_above_finite_max_KL",
                "mathematical_note": "No finite positive beta realizes this binding radius; this does not mean the KL-constrained problem has no feasible policy",
                "automatic_beta_zero_fallback": False, "training_authorized_by_this_output": False}

    def evaluate(t):
        values = [improve(logq, reward, t) for logq, reward in zip(logq_bank, reward_bank)]
        return math.fsum(value["kl"] for value in values) / PREFIX, values

    low, high = 0.0, 1.0
    high_kl, values = evaluate(high)
    brackets = 0
    while high_kl < delta:
        require(brackets < 80 and high <= 2 ** 79, "Positive-beta root was not bracketed; explicit review required")
        low, high = high, high * 2
        high_kl, values = evaluate(high)
        brackets += 1
    tolerance = max(1e-12, delta * 1e-9)
    for iterations in range(1, 257):
        middle = (low + high) / 2
        actual, values = evaluate(middle)
        if abs(actual - delta) <= tolerance:
            beta = 1 / middle
            require(math.isfinite(beta) and beta > 0, "Root did not yield a finite positive beta")
            return {**base, "status": "calibrated_positive_beta", "exit_code": 0, "beta": beta,
                    "inverse_beta": middle, "actual_mean_KL": actual, "residual": actual - delta,
                    "absolute_residual_tolerance": tolerance, "bisection_iterations": iterations,
                    "bracket_expansions": brackets, "ideal_policies": values,
                    "automatic_beta_zero_fallback": False, "training_authorized_by_this_output": False}
        if actual < delta:
            low = middle
        else:
            high = middle
    raise ValueError("Positive-beta bisection did not satisfy the fixed residual tolerance")


def validate_reference(rows, config, records):
    require(len(rows) == len(records) == EXPECTED_ROWS, "Require complete256-row reference and selected training prefix")
    require(config["policy"] == "canonical24_full_sequence_plusEOS_temperature1"
            and tuple(config["candidate_texts"]) == CANDIDATES, "Wrong reference policy/candidates")
    require(config["selected_records_sha256"] == digest(records), "Reference selected training prefix differs")
    require(len({str(row["qid"]) for row in rows}) == EXPECTED_ROWS
            and len({str(row["presentation_id"]) for row in rows}) == EXPECTED_ROWS, "Duplicate reference input")
    require(config["input"] == {"min_pixels": 65536, "max_pixels": 262144, "recipe_token_budget": 1536,
                                "max_input_tokens": 4096, "max_completion_tokens": 32, "prompt_version": "recipeqa-order-v1"},
            "Input preprocessing differs from common finite-policy experiment")
    signature = digest(config)
    common_ids, common_count = None, None
    for index, (row, record) in enumerate(zip(rows, records)):
        require(row["index"] == index and row["reference_spec_sha256"] == signature, "Reference index/signature differs")
        require(row["source_split"] == record["source_split"] == "train"
                and record.get("manifest_set") not in ("val", "test", "orbit", "process", "confirmation")
                and record["technical_status"] == "ok", "Nontraining or damaged input")
        require(all(row[field] == record[field] for field in
                    ("qid", "recipe_id", "presentation_id", "display_perm", "target_order", "target_labels")),
                "Reference source mapping differs from selected training manifest")
        display = row["display_perm"]
        require(sorted(display) == [0, 1, 2, 3] and row["target_order"] == [0, 1, 2, 3]
                and row["target_labels"] == [chr(65 + display.index(i)) for i in range(4)], "Invalid gold/display mapping")
        require(len(row["candidates"]) == len(row["token_logps"]) == len(row["q"]) == len(row["logq"]) == 24,
                "Reference needs all24 complete actions")
        require(tuple(candidate["text"] for candidate in row["candidates"]) == CANDIDATES, "Candidate text/order differs")
        scores, candidate_ids = [], []
        for candidate, token_logps in zip(row["candidates"], row["token_logps"]):
            ids = candidate["token_ids"]
            require(len(ids) == len(token_logps) == candidate["token_count"] and len(ids) >= 2
                    and all(type(value) is int and value >= 0 for value in ids)
                    and type(candidate["eos_token_id"]) is int and ids[-1] == candidate["eos_token_id"]
                    and candidate["eos_token_id"] not in ids[:-1], "Incomplete canonical suffix/EOS")
            require(all(type(value) in (int, float) and math.isfinite(value) and value <= 1e-6 for value in token_logps),
                    "Invalid token likelihood")
            if common_count is None:
                common_count = len(ids)
            require(common_count == len(ids), "Canonical objective token count C differs")
            candidate_ids.append(ids)
            scores.append(math.fsum(token_logps))
        if common_ids is None:
            common_ids = candidate_ids
        require(candidate_ids == common_ids, "Canonical token IDs changed across reference inputs")
        logq, q, logz = normalize(scores)
        require(all(type(value) in (int, float) and math.isfinite(value) for value in row["logq"])
                and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in row["q"])
                and max(abs(a - b) for a, b in zip(logq, row["logq"])) <= 1e-9
                and max(abs(a - b) for a, b in zip(q, row["q"])) <= 1e-9
                and math.isfinite(row["logZ"]) and abs(logz - row["logZ"]) <= 1e-9 and logz <= 1e-4,
                "Reference q/logq/logZ differs from complete EOS likelihoods")
        rewards = [score_order(text, display, row["target_order"])["exact"] for text in CANDIDATES]
        require(row["exact_rewards"] == rewards and sum(rewards) == 1
                and type(row["gold_candidate_index"]) is int and row["gold_candidate_index"] == rewards.index(1),
                "Missing/incorrect verified exact target")
        require(row["input_audit"]["qid"] == row["qid"] and row["input_audit"]["recipe_id"] == row["recipe_id"]
                and row["input_audit"]["presentation_id"] == row["presentation_id"]
                and is_hash(row["input_audit"]["input_hash"]), "Reference input audit identity differs")
    return {"rows": EXPECTED_ROWS, "canonical_token_count": common_count, "reference_spec_sha256": signature}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "full_train_data/train_all7558.jsonl")
    parser.add_argument("--prefix", type=int, default=PREFIX)
    parser.add_argument("--delta", type=float, default=.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(ROOT.is_relative_to("/root/autodl-tmp") and ROOT.stat().st_dev != Path("/").stat().st_dev,
            "Calibration artifacts must stay on mounted project data disk")
    output = args.output.resolve()
    require(output.is_relative_to(ROOT) and not output.exists(), "Output must be a new data-disk JSON; never overwrite")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "analyzer_source_sha256": sha(__file__),
              "test_access": False, "confirmation_access": False, "gpu_access": False,
              "training_started": False, "reference_modified": False,
              "bank_selection": "First64 of complete256 fixed training prefix; no score-based selection"}
    exit_code = 2
    try:
        require(args.prefix == PREFIX, "Only the predeclared first64 training references may calibrate beta")
        cache = safe_input(args.reference_cache)
        manifest_path = safe_input(args.manifest)
        config_path = safe_input(Path(str(cache) + ".config.json"))
        completion_path = safe_input(Path(str(cache) + ".completion.json"))
        config, completion = load(config_path), load(completion_path)
        require(completion["status"] == "complete_fixed_initial_reference" and completion["rows"] == EXPECTED_ROWS
                and completion["cache_sha256"] == sha(cache)
                and completion["reference_spec_sha256"] == digest(config), "Reference sidecars do not bind complete cache")
        require(sha(manifest_path) == config["manifest_sha256"], "Training source bytes differ from reference config")
        records = jsonl(manifest_path)[:EXPECTED_ROWS]
        rows = jsonl(cache)
        validation = validate_reference(rows, config, records)
        sources = config["scoring_source_sha256"]
        required_sources = {"exploratory/principled_ordering_20261105/score_permutations.py",
                            "exploratory/principled_ordering_20261105/gradient_alignment.py",
                            "src/recipeqa/modeling.py", "src/recipeqa/metrics.py"}
        require(set(sources) == required_sources, "Unexpected/missing reference scoring source set")
        for name, expected in sources.items():
            require(is_hash(expected) and sha(safe_input(ROOT / name)) == expected, "Reference scientific source changed: " + name)
        before = {"reference_cache": receipt(cache), "reference_config": receipt(config_path),
                  "reference_completion": receipt(completion_path), "training_manifest": receipt(manifest_path)}
        bank = rows[:PREFIX]
        result = calibrate([row["logq"] for row in bank], [row["exact_rewards"] for row in bank], args.delta)
        report.update(result, reference_validation=validation, input_receipts=before,
                      reference_weight_identity=config["initial_identity"], reference_pixel_sha256=config["pixel_sha256"],
                      scoring_source_sha256=sources, bank_sha256=digest(bank),
                      bank_inputs=[{"index": row["index"], "qid": row["qid"], "recipe_id": row["recipe_id"],
                                    "presentation_id": row["presentation_id"], "input_hash": row["input_audit"]["input_hash"],
                                    "gold_candidate_index": row["gold_candidate_index"], "logq_ref": row["logq"],
                                    "q_ref": row["q"], "logZ_ref": row["logZ"], "exact_rewards": row["exact_rewards"]} for row in bank])
        for item in before.values():
            require(sha(item["path"]) == item["sha256"], "An input changed during CPU calibration")
        exit_code = result["exit_code"]
    except Exception as error:
        report.update(status="failed_reference_or_calibration_validation", error=f"{type(error).__name__}: {error}",
                      beta=None, exit_code=2, automatic_beta_zero_fallback=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report.get(key) for key in ("status", "beta", "delta", "actual_mean_KL", "D_max", "exit_code")}, allow_nan=False), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
