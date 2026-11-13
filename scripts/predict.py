#!/usr/bin/env python3
"""Append resumable raw predictions for E0/E3, W reward bank, or E4 process prompts."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recipeqa.metrics import score_order
from recipeqa.modeling import (
    InputConfig, encode_messages, ensure_data_disk, hash_json, load_images, load_model,
    load_processor, model_weight_identity, package_versions, prepare_record, read_jsonl, sha256_file, write_json,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--adapter-path")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True, help="Unique-key JSONL; successes are skipped, failures can be retried")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-samples", type=int, default=1, help="Use 4 for the W candidate bank")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy; W bank uses 1.0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--process", action="store_true", help="E4 structured evidence prompt; use 256 new tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--recipe-token-budget", type=int, default=1536)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def prediction_key(record):
    return (record["run_id"], record["checkpoint"], record["qid"], record["presentation_id"],
            record.get("sample_index", 0), record.get("prompt_kind", "answer"))


class PredictionStore:
    """One canonical row per key, plus a separate technical-attempt journal.

    New keys append cheaply. Retrying an existing technical failure performs an
    atomic rewrite, so downstream metrics never see failed+successful duplicates.
    There must be only one writer per prediction file.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.rows = {}
        previous = read_jsonl(self.path) if self.path.exists() else []
        for row in previous:
            key = prediction_key(row)
            old = self.rows.get(key)
            if old and not old.get("technical_error"):
                if not row.get("technical_error") and old != row:
                    raise ValueError(f"Conflicting successful predictions for {key}")
                continue
            self.rows[key] = row
        if len(self.rows) != len(previous):
            self._rewrite()

    def _rewrite(self):
        temporary = self.path.with_suffix(self.path.suffix + ".partial")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in self.rows.values():
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(self.path)

    def write(self, row):
        key = prediction_key(row)
        old = self.rows.get(key)
        if old and not old.get("technical_error"):
            if row != old:
                raise ValueError(f"Refusing to replace a successful prediction for {key}")
            return
        if row.get("technical_error"):
            journal = self.path.with_suffix(self.path.suffix + ".technical_attempts.jsonl")
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.rows[key] = row
        if old is not None:
            self._rewrite()
        else:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = arguments()
    ensure_data_disk(args.model_path, args.manifest, args.image_root, args.output)
    if args.adapter_path:
        ensure_data_disk(args.adapter_path)
    if args.num_samples > 1 and args.temperature <= 0:
        raise ValueError("Multiple candidates require positive sampling temperature")
    if args.process and args.num_samples != 1:
        raise ValueError("E4 is one greedy response per fixed question")
    if args.process and args.temperature != 0:
        raise ValueError("The E4 protocol is greedy")
    if args.num_samples < 1:
        raise ValueError("num-samples must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = str(Path(args.adapter_path or args.model_path).resolve())
    weight_identity = {"base": model_weight_identity(args.model_path),
                       "adapter": model_weight_identity(args.adapter_path, adapter=True) if args.adapter_path else None}
    weights_hash = hash_json(weight_identity)
    config_path = output.with_suffix(output.suffix + ".config.json")
    if config_path.exists():
        previous_config = json.loads(config_path.read_text())
        if previous_config.get("weight_identity") != weight_identity:
            raise ValueError("Actual base/adapter contents differ or old output lacks content hashes; use a new output file")
    prompt_kind = "process" if args.process else "answer"
    completed = set()
    store = PredictionStore(output)
    for row in store.rows.values():
        if row["run_id"] != args.run_id or row["checkpoint"] != checkpoint:
            raise ValueError("Refusing to append another model/run to this prediction file")
        if row.get("model_weights_sha256") != weights_hash:
            raise ValueError("Prediction rows have another or unverified base/adapter weight identity")
        if not row.get("technical_error"):
            completed.add(prediction_key(row))

    import torch

    input_config = InputConfig(args.min_pixels, args.max_pixels, args.recipe_token_budget,
                               args.max_input_tokens, 32)
    processor = load_processor(args.model_path, args.revision, input_config)
    model = load_model(args.model_path, args.revision, adapter_path=args.adapter_path,
                       trainable=False, seed=args.seed)
    generation = copy.deepcopy(model.generation_config)
    generation.do_sample = args.temperature > 0
    generation.temperature = args.temperature if args.temperature > 0 else None
    generation.top_p = 1.0 if args.temperature > 0 else None
    generation.top_k = 0 if args.temperature > 0 else None
    generation.max_new_tokens = args.max_new_tokens
    generation.use_cache = True
    generation.pad_token_id = processor.tokenizer.pad_token_id
    records = read_jsonl(args.manifest, args.limit)
    config_payload = {"cli": vars(args), "input": asdict(input_config), "versions": package_versions(),
                      "generation": generation.to_dict(), "manifest_sha256": sha256_file(args.manifest),
                      "weight_identity": weight_identity, "model_weights_sha256": weights_hash,
                      "model_revision_resolved": getattr(model.config, "_commit_hash", None)}
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if hash_json(previous) != hash_json(config_payload):
            raise ValueError("Prediction configuration changed; use another output file")
    else:
        write_json(config_path, config_payload)

    for row in records:
            planned = [{"run_id": args.run_id, "checkpoint": checkpoint, "qid": row["qid"],
                        "model_weights_sha256": weights_hash,
                        "presentation_id": row["presentation_id"], "sample_index": i,
                        "prompt_kind": prompt_kind} for i in range(args.num_samples)]
            if all(prediction_key(value) in completed for value in planned):
                continue
            try:
                prepared, audit = prepare_record(row, processor, args.image_root, input_config, process=args.process)
                images = load_images(row, args.image_root)
                encoded = encode_messages(processor, prepared["prompt"], images).to(model.device)
                prefix_tokens = encoded["input_ids"].shape[-1]
            except Exception as error:
                for base in planned:
                    if prediction_key(base) not in completed:
                        store.write({**base, "model_id": "Qwen/Qwen3-VL-2B-Instruct", "seed": args.seed,
                                     "recipe_id": row["recipe_id"], "display_perm": row["display_perm"],
                                     "target_order": row.get("target_order", [0, 1, 2, 3]),
                                     "target_labels": row["target_labels"], "raw_response": "",
                                     "technical_stage": "preprocessing",
                                     "technical_error": f"{type(error).__name__}: {error}"})
                if not args.continue_on_error:
                    raise
                if isinstance(error, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                continue
            for sample_index in range(args.num_samples):
                base_record = {
                    **planned[sample_index], "model_id": "Qwen/Qwen3-VL-2B-Instruct", "seed": args.seed,
                    "recipe_id": row["recipe_id"], "source_split": row.get("source_split"),
                    "display_perm": row["display_perm"], "target_order": row.get("target_order", [0, 1, 2, 3]),
                    "target_labels": row["target_labels"], "input_audit": audit,
                    "source_step_ids_chronological": row.get("source_step_ids_chronological"),
                    "candidate_group_id": f"{args.run_id}:{row['qid']}:{row['presentation_id']}:{prompt_kind}",
                    "decoding": {"do_sample": args.temperature > 0, "temperature": args.temperature,
                                 "max_new_tokens": args.max_new_tokens, "top_p": 1.0, "top_k": 0},
                    "technical_error": None,
                }
                if prediction_key(base_record) in completed:
                    continue
                # Per-key sample seeds make interrupted prediction jobs resumable
                # without depending on how many previous questions were skipped.
                sample_seed = int(hash_json([args.seed, row["qid"], row["presentation_id"], sample_index])[:8], 16)
                torch.manual_seed(sample_seed)
                torch.cuda.manual_seed_all(sample_seed)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.monotonic()
                try:
                    with torch.inference_mode():
                        sequences = model.generate(**encoded, generation_config=generation)
                    torch.cuda.synchronize()
                    generated_ids = sequences[0, prefix_tokens:]
                    raw_response = processor.tokenizer.decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                    )
                    result = {
                        **base_record, "raw_response": raw_response, "sample_seed": sample_seed,
                        "input_tokens": prefix_tokens, "completion_tokens": int(generated_ids.numel()),
                        "reached_token_limit": generated_ids.numel() == args.max_new_tokens,
                        "last_token_id": int(generated_ids[-1]) if generated_ids.numel() else None,
                        "generation_seconds": time.monotonic() - started,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    }
                    if not args.process:
                        result["score"] = score_order(raw_response, row["display_perm"],
                                                      target_order=row.get("target_order", [0, 1, 2, 3]))
                    store.write(result)
                    completed.add(prediction_key(result))
                    print(json.dumps({"qid": row["qid"], "presentation_id": row["presentation_id"],
                                      "sample_index": sample_index, "raw_response": raw_response,
                                      "score": result.get("score"), "seconds": result["generation_seconds"]}), flush=True)
                except Exception as error:
                    store.write({**base_record, "raw_response": "", "sample_seed": sample_seed,
                                 "technical_stage": "generation",
                                 "technical_error": f"{type(error).__name__}: {error}"})
                    if not args.continue_on_error:
                        raise
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
