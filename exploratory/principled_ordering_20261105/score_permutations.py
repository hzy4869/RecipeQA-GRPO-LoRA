#!/usr/bin/env python3
"""Development-only exact likelihood diagnostic over 24 canonical order strings.

This is NEW post-hoc exploration. It does not modify the original free-generation
protocol. All candidates are constructed before consulting the gold answer.
No prefix cache, reward-based search, model download, or test override is used.
Heavy dependencies are lazy so the probability and manifest checks run on CPU.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import permutations
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recipeqa.metrics import score_order
from recipeqa.modeling import (
    InputConfig, encode_messages, ensure_data_disk, hash_json, load_images,
    load_model, load_processor, model_weight_identity, package_versions,
    prepare_record, read_jsonl, sha256_file, write_json,
)

DEV_SHA256 = "c998796fbc07a24c7a24062d6d0aba07500c75b34568fd7ba67f8c38daae03c9"
CANDIDATES = tuple(" ".join(order) for order in permutations("ABCD"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-path", required=True)
    result.add_argument("--revision")
    result.add_argument("--adapter-path")
    result.add_argument("--manifest", required=True)
    result.add_argument("--expected-manifest-sha256", default=DEV_SHA256)
    result.add_argument("--image-root", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--limit", type=int, default=2,
                        help="Fixed first N development questions; start at 2, then 32, then 200")
    result.add_argument("--microbatch", type=int, choices=(1, 2, 4), default=1)
    result.add_argument("--greedy-predictions", help="Optional strict matched existing greedy JSONL")
    result.add_argument("--batch-atol", type=float, default=0.01,
                        help="Maximum absolute token-logp difference versus unbatched reference")
    result.add_argument("--tie-atol", type=float, default=1e-8)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--min-pixels", type=int, default=65536)
    result.add_argument("--max-pixels", type=int, default=262144)
    result.add_argument("--recipe-token-budget", type=int, default=1536)
    result.add_argument("--max-input-tokens", type=int, default=4096)
    return result


def validate_records(records):
    """Reject test/train rows regardless of filename or proposed SHA override."""
    require(len(records) == 200, "The complete input must be the fixed 200-question development set")
    require(len({str(row["recipe_id"]) for row in records}) == 200,
            "Development diagnostic requires 200 distinct recipe clusters")
    require(len({(str(row["qid"]), str(row["presentation_id"])) for row in records}) == 200,
            "Duplicate development question/presentation")
    for row in records:
        require(row.get("source_split") == "val", "Only source_split=val is allowed; no test override exists")
        require(row.get("manifest_set") == "val", "Only the original single-layout val manifest is allowed")
        require(len(row.get("target_labels", [])) == 4 and set(row["target_labels"]) == set("ABCD"),
                "Gold must be a permutation of A-D")


def candidate_suffixes(processor, messages):
    """Verify canonical answers exactly match the completion-only chat suffix.

    The complete assistant template may add a whitespace newline after EOS.
    We score through EOS, not that following whitespace. Boundary tokenization
    must preserve the entire prompt token prefix; otherwise stop, never guess.
    """
    tokenizer = processor.tokenizer
    eos = tokenizer.eos_token_id
    require(isinstance(eos, int), "A single canonical tokenizer EOS ID is required")
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    rows = []
    for text in CANDIDATES:
        full = processor.apply_chat_template(
            messages + [{"role": "assistant", "content": text}],
            tokenize=False, add_generation_prompt=False,
        )
        require(full.startswith(prefix), "Assistant template changes generation prompt prefix")
        full_ids = tokenizer.encode(full, add_special_tokens=False)
        require(full_ids[:len(prefix_ids)] == prefix_ids, "Prompt/answer BPE boundary is not prefix-stable")
        suffix = full_ids[len(prefix_ids):]
        require(eos in suffix, "Assistant completion does not contain canonical EOS")
        end = suffix.index(eos)
        ids = suffix[:end + 1]
        expected = tokenizer.encode(text, add_special_tokens=False) + [eos]
        require(ids == expected, "Assistant answer suffix differs from plain canonical answer plus EOS")
        require(not tokenizer.decode(suffix[end + 1:], skip_special_tokens=False).strip(),
                "Non-whitespace template content follows EOS")
        require(tokenizer.decode(ids[:-1], skip_special_tokens=False) == text,
                "Canonical answer does not round-trip exactly")
        rows.append({"text": text, "token_ids": ids, "token_count": len(ids),
                     "eos_token_id": eos, "post_eos_whitespace_token_ids": suffix[end + 1:]})
    require(len({row["token_count"] for row in rows}) == 1,
            "Canonical candidates differ in token length; this diagnostic does not pad/length-normalize them")
    return rows


def probability_summary(scores, gold_text, tie_namespace, tie_atol=1e-8):
    """Pure CPU summary. Hash-based tie handling never reads the target."""
    require(len(scores) == 24 and all(math.isfinite(value) for value in scores), "Need 24 finite log scores")
    require(gold_text in CANDIDATES, "Unknown canonical target")
    maximum = max(scores)
    log_z = maximum + math.log(sum(math.exp(value - maximum) for value in scores))
    q = [math.exp(value - log_z) for value in scores]
    tied = [i for i, value in enumerate(scores) if maximum - value <= tie_atol]
    chosen = min(tied, key=lambda i: hash_json(["canonical-map-tie-v1", tie_namespace, CANDIDATES[i]]))
    gold = CANDIDATES.index(gold_text)
    better = sum(value > scores[gold] + tie_atol for value in scores)
    gold_ties = sum(abs(value - scores[gold]) <= tie_atol for value in scores)
    sorted_scores = sorted(scores, reverse=True)
    ordinal = [[0.0] * 4 for _ in range(4)]
    pair = {f"{a}_{b}": 0.0 for a in "ABCD" for b in "ABCD" if a < b}
    for text, weight in zip(CANDIDATES, q):
        order = text.split()
        for rank, label in enumerate(order):
            ordinal[ord(label) - ord("A")][rank] += weight
        for name in pair:
            a, b = name.split("_")
            pair[name] += weight * (order.index(a) < order.index(b))
    return {"logZ_canonical_24": log_z,
            "canonical_24_string_mass": math.exp(log_z),
            "canonical_mass_warning": "Only these 24 exact strings including EOS; not all parser-valid formats",
            "q24": q, "q24_entropy_nats": -sum(p * math.log(p) for p in q if p > 0),
            "map_text": CANDIDATES[chosen], "map_candidate_index": chosen,
            "top_tied_indices": tied, "top_two_logp_margin": sorted_scores[0] - sorted_scores[1],
            "gold_candidate_index": gold, "gold_logp": scores[gold], "q24_gold": q[gold],
            "gold_rank_best": better + 1, "gold_rank_worst": better + gold_ties,
            "gold_to_top_logp_gap": maximum - scores[gold],
            "ordinal_marginals_display_label_by_rank": ordinal,
            "pair_marginals_display_label_before": pair,
            "tie_policy": "max score within tie_atol; minimum SHA256 of namespace and displayed candidate string"}


def repeat_inputs(encoded, candidate_ids):
    """Literal independent prompt copies with images repeated in batch order."""
    import torch

    allowed = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}
    require(not (set(encoded) - allowed), f"Unsupported processor keys: {set(encoded) - allowed}")
    require(encoded["input_ids"].shape[0] == 1, "Expected one question per encoded prompt")
    batch = len(candidate_ids)
    lengths = {len(row) for row in candidate_ids}
    require(len(lengths) == 1, "Candidate batch must have equal lengths")
    ids = torch.tensor(candidate_ids, device=encoded["input_ids"].device, dtype=encoded["input_ids"].dtype)
    result = {"input_ids": torch.cat([encoded["input_ids"].repeat(batch, 1), ids], dim=1)}
    attention = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"]))
    result["attention_mask"] = torch.cat([attention.repeat(batch, 1), torch.ones_like(ids)], dim=1)
    for name in ("pixel_values", "image_grid_thw"):
        require(name in encoded, f"Missing four-image multimodal input: {name}")
        result[name] = torch.cat([encoded[name]] * batch, dim=0)
    require(encoded["image_grid_thw"].shape[0] == 4, "Expected exactly four image grids per prompt")
    if "mm_token_type_ids" in encoded:
        result["mm_token_type_ids"] = torch.cat(
            [encoded["mm_token_type_ids"].repeat(batch, 1), torch.zeros_like(ids)], dim=1)
    return result


def forward_logps(model, encoded, candidate_ids, *, literal_full_logits=False):
    """No KV cache. Full multimodal forward; exact selected answer logits only."""
    import torch

    full = repeat_inputs(encoded, candidate_ids)
    count = len(candidate_ids[0])
    prefix_count = encoded["input_ids"].shape[1]
    # The final input token's prediction is unused. Keeping count+1 positions
    # includes the prompt's last token, which predicts the first answer token.
    with torch.inference_mode():
        output = model(**full, use_cache=False, return_dict=True,
                       logits_to_keep=0 if literal_full_logits else count + 1)
        if literal_full_logits:
            logits = output.logits[:, prefix_count - 1:prefix_count + count - 1, :]
        else:
            require(output.logits.shape[1] == count + 1, "Model ignored logits_to_keep; stop rather than misalign")
            logits = output.logits[:, :-1, :]
        targets = full["input_ids"][:, prefix_count:]
        require(logits.shape[:2] == targets.shape, "Answer logit/target alignment mismatch")
        values = logits.float().log_softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        result = values.cpu().tolist()
    del output, full, logits, values
    return result


def score_candidates(model, encoded, candidates, args, *, literal_check):
    token_logps = []
    for begin in range(0, 24, args.microbatch):
        token_logps.extend(forward_logps(
            model, encoded, [row["token_ids"] for row in candidates[begin:begin + args.microbatch]],
        ))
    # Every question checks candidate 0 independently. On the first question,
    # check all first-batch positions to catch multimodal repetition mistakes.
    check_indices = list(range(args.microbatch)) if literal_check else [0]
    verification = []
    for index in check_indices:
        reference = forward_logps(model, encoded, [candidates[index]["token_ids"]],
                                  literal_full_logits=literal_check and index == 0)[0]
        delta = max(abs(a - b) for a, b in zip(reference, token_logps[index]))
        require(delta <= args.batch_atol,
                f"Independent full-forward reference failed ({delta} > {args.batch_atol}); rerun microbatch=1 in a new output")
        verification.append({"candidate_index": index, "literal_all_position_logits": literal_check and index == 0,
                             "reference_token_logps": reference, "max_token_logp_abs_difference": delta})
    scores = [sum(row) for row in token_logps]
    require(all(math.isfinite(x) for row in token_logps for x in row), "Non-finite likelihood")
    margin = sorted(scores, reverse=True)[0] - sorted(scores, reverse=True)[1]
    if args.microbatch > 1 and args.tie_atol < margin <= 2 * args.batch_atol * len(candidates[0]["token_ids"]):
        raise ValueError("Numerically fragile top-two ranking at this microbatch tolerance; rerun microbatch=1 in a new output")
    return token_logps, verification


def load_greedy(path, manifest, identity, selected):
    if not path:
        return None
    require("test" not in Path(path).name.lower(), "Test prediction files are forbidden")
    config = json.loads(Path(str(path) + ".config.json").read_text())
    require(config["manifest_sha256"] == sha256_file(manifest), "Greedy baseline used another manifest")
    require(config["weight_identity"] == identity, "Greedy baseline weights differ")
    rows = read_jsonl(path)
    result = {}
    for row in rows:
        key = (str(row["qid"]), str(row["presentation_id"]))
        require(key not in result, "Greedy file contains duplicate question/presentation")
        require(row.get("source_split") == "val" and not row.get("technical_error"),
                "Greedy file contains non-development rows or technical failures")
        require(row.get("prompt_kind") == "answer" and row["decoding"]["do_sample"] is False
                and row["decoding"]["max_new_tokens"] == 32, "Greedy baseline is not the frozen short-answer decoding")
        result[key] = row
    require(all((str(row["qid"]), str(row["presentation_id"])) in result for row in selected),
            "Greedy baseline is incomplete for selected development rows")
    return result


def get_greedy(model, processor, encoded, row, audit, bank, identity_sha):
    if bank is not None:
        old = bank[(str(row["qid"]), str(row["presentation_id"]))]
        require(old["input_audit"]["input_hash"] == audit["input_hash"], "Greedy baseline prompt/pixels/budget differs")
        require(old["model_weights_sha256"] == identity_sha, "Greedy baseline row has mismatched actual weights")
        require(old["display_perm"] == row["display_perm"] and old["target_labels"] == row["target_labels"],
                "Greedy baseline presentation or target differs")
        return {"source": "strict_matched_file", "raw_response": old["raw_response"],
                "completion_tokens": old.get("completion_tokens"),
                "score": score_order(old["raw_response"], row["display_perm"], row.get("target_order", [0, 1, 2, 3]))}
    import torch

    generation = copy.deepcopy(model.generation_config)
    generation.do_sample = False
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None
    generation.max_new_tokens = 32
    generation.use_cache = True
    generation.pad_token_id = processor.tokenizer.pad_token_id
    with torch.inference_mode():
        output = model.generate(**encoded, generation_config=generation)
    ids = output[0, encoded["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {"source": "actual_generate_greedy", "raw_response": text, "token_ids": ids.cpu().tolist(),
            "completion_tokens": int(ids.numel()), "generation_config": generation.to_dict(),
            "score": score_order(text, row["display_perm"], row.get("target_order", [0, 1, 2, 3]))}


def main():
    args = parser().parse_args()
    require(1 <= args.limit <= 200, "Limit must be in 1..200")
    require(0 < args.batch_atol <= .1 and 0 <= args.tie_atol < args.batch_atol, "Invalid numeric tolerances")
    ensure_data_disk(args.model_path, args.manifest, args.image_root, args.output)
    if args.adapter_path:
        ensure_data_disk(args.adapter_path)
    if args.greedy_predictions:
        ensure_data_disk(args.greedy_predictions)
    require("test" not in Path(args.manifest).name.lower(), "Test manifests are forbidden")
    require(sha256_file(args.manifest) == args.expected_manifest_sha256, "Manifest differs from explicitly bound development SHA")
    all_records = read_jsonl(args.manifest)
    validate_records(all_records)
    selected = all_records[:args.limit]
    identity = {"base": model_weight_identity(args.model_path),
                "adapter": model_weight_identity(args.adapter_path, adapter=True) if args.adapter_path else None}
    identity_sha = hash_json(identity)
    config = InputConfig(args.min_pixels, args.max_pixels, args.recipe_token_budget, args.max_input_tokens, 32)
    provenance = {"schema_version": 1, "post_hoc_exploration": True, "cli": vars(args), "input": asdict(config),
                  "versions": package_versions(), "manifest_sha256": sha256_file(args.manifest),
                  "selected_rows_sha256": hash_json(selected), "weight_identity": identity,
                  "model_weights_sha256": identity_sha,
                  "candidate_texts": CANDIDATES, "candidate_generation": "All itertools.permutations('ABCD'), independent of target",
                  "score": "Unnormalized complete canonical sequence log probability including EOS; no length normalization",
                  "inference": "No prefix cache; literal complete multimodal forward per candidate batch",
                  "greedy_predictions_sha256": sha256_file(args.greedy_predictions) if args.greedy_predictions else None,
                  "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in
                                     (Path(__file__), ROOT / "src/recipeqa/modeling.py", ROOT / "src/recipeqa/metrics.py")}}
    signature = hash_json(provenance)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = output.with_suffix(output.suffix + ".config.json")
    if config_path.exists():
        require(json.loads(config_path.read_text()) == json.loads(json.dumps(provenance)), "Configuration or actual weight bytes changed; choose a new output")
    else:
        require(not output.exists(), "Cannot resume output without its signed configuration")
        write_json(config_path, provenance)
    completed = {}
    allowed_keys = {(str(row["qid"]), str(row["presentation_id"])) for row in selected}
    for old in read_jsonl(output) if output.exists() else []:
        key = (str(old["qid"]), str(old["presentation_id"]))
        require(key in allowed_keys and key not in completed, "Output has an unexpected or duplicate question")
        require(old["configuration_sha256"] == signature and old["model_weights_sha256"] == identity_sha,
                "Output row has mismatched configuration or weights")
        require(not old.get("technical_error"), "Failure rows cannot silently be treated as successful completion")
        completed[key] = old
    bank = load_greedy(args.greedy_predictions, args.manifest, identity, selected)
    processor = load_processor(args.model_path, args.revision, config)
    model = load_model(args.model_path, args.revision, adapter_path=args.adapter_path, trainable=False, seed=args.seed)
    import torch

    for record in selected:
        key = (str(record["qid"]), str(record["presentation_id"]))
        if key in completed:
            continue
        prepared, audit = prepare_record(record, processor, args.image_root, config)
        candidates = candidate_suffixes(processor, prepared["prompt"])
        encoded = encode_messages(processor, prepared["prompt"], load_images(record, args.image_root)).to(model.device)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.monotonic()
        token_logps, verification = score_candidates(model, encoded, candidates, args, literal_check=not completed)
        scores = [sum(values) for values in token_logps]
        # Namespace depends only on visible-prompt identity; never on target/order metadata.
        namespace = audit["prompt_text_hash"]
        gold = " ".join(record["target_labels"])
        summary = probability_summary(scores, gold, namespace, args.tie_atol)
        require(summary["logZ_canonical_24"] <= 1e-4, "Disjoint complete-string probability mass exceeds one")
        without_eos = probability_summary([sum(values[:-1]) for values in token_logps], gold, namespace, args.tie_atol)
        greedy = get_greedy(model, processor, encoded, record, audit, bank, identity_sha)
        torch.cuda.synchronize()
        result = {"run_id": args.run_id, "qid": record["qid"], "recipe_id": record["recipe_id"],
                  "presentation_id": record["presentation_id"], "source_split": "val", "manifest_set": "val",
                  "display_perm": record["display_perm"], "target_order": record.get("target_order", [0, 1, 2, 3]),
                  "target_labels": record["target_labels"], "configuration_sha256": signature,
                  "model_weights_sha256": identity_sha, "input_audit": audit,
                  "candidates": [{**candidate, "token_logps_including_eos": values,
                                  "logp_including_eos": sum(values), "logp_without_eos_diagnostic": sum(values[:-1])}
                                 for candidate, values in zip(candidates, token_logps)],
                  "canonical_map": summary,
                  "canonical_map_score": score_order(summary["map_text"], record["display_perm"], record.get("target_order", [0, 1, 2, 3])),
                  "without_eos_diagnostic_only": without_eos, "greedy": greedy,
                  "numerical_reference_checks": verification,
                  "wall_seconds_including_greedy_and_reference": time.monotonic() - started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "completed_utc": datetime.now(timezone.utc).isoformat(), "technical_error": None}
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        completed[key] = result
        print(json.dumps({"qid": record["qid"], "done": len(completed), "planned": args.limit,
                          "greedy_exact": greedy["score"]["exact"], "map_exact": result["canonical_map_score"]["exact"],
                          "gold_rank": summary["gold_rank_best"], "q24_gold": summary["q24_gold"],
                          "seconds": result["wall_seconds_including_greedy_and_reference"]}), flush=True)
    write_json(output.with_suffix(output.suffix + ".completion.json"),
               {"status": "complete", "configuration_sha256": signature, "rows": len(completed),
                "expected_rows": args.limit, "output_sha256": sha256_file(output),
                "greedy_exact_micro": sum(row["greedy"]["score"]["exact"] for row in completed.values()) / len(completed),
                "canonical_map_exact_micro": sum(row["canonical_map_score"]["exact"] for row in completed.values()) / len(completed),
                "interpretation": "Development-only exact canonical-action inference diagnostic, not official RecipeQA SOTA"})


if __name__ == "__main__":
    main()
