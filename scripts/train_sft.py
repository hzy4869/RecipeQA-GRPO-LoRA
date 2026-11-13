#!/usr/bin/env python3
"""Train one completion-only SFT branch, or resume its full Trainer checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recipeqa.modeling import (
    InputConfig, adapter_fingerprint, assert_training_destination, attach_language_lora,
    begin_training_attempt, ensure_data_disk,
    frozen_vision_fingerprint, load_model, load_processor, make_run_callback,
    package_versions, prepare_dataset, read_jsonl, sha256_file, training_signature_payload,
    trainable_inventory, write_json,
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--manifest", required=True, help="Pre-generated visit stream; consumed once in saved order")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-adapter", help="Common W adapter; starts a new optimizer, unlike --resume")
    parser.add_argument("--resume", help="Complete checkpoint directory (optimizer, scheduler and RNG required)")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--recipe-token-budget", type=int, default=1536)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    return parser.parse_args()


def main():
    args = arguments()
    ensure_data_disk(args.model_path, args.manifest, args.image_root, args.output_dir)
    if args.initial_adapter:
        ensure_data_disk(args.initial_adapter)
    if args.resume:
        ensure_data_disk(args.resume)
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            if not (Path(args.resume) / name).is_file():
                raise RuntimeError(f"Full resume requires {name}; loading weights alone is a restart")
    output = Path(args.output_dir)
    assert_training_destination(output, args.resume)

    import torch
    from trl import SFTConfig, SFTTrainer

    class AuditedSFTTrainer(SFTTrainer):
        def training_step(self, model, inputs, *positional, **keyword):
            metadata = inputs.pop("_recipeqa_metadata")
            step_before_update = self.state.global_step
            loss = super().training_step(model, inputs, *positional, **keyword)
            for example in metadata:
                attempt.append("actual_prompt_visits.jsonl", {
                    **example, "global_step_before_update": step_before_update,
                    "consumption_point": "training_step_forward_backward_returned",
                })
            return loss

    started = time.monotonic()
    input_config = InputConfig(args.min_pixels, args.max_pixels, args.recipe_token_budget, args.max_input_tokens)
    attempt = begin_training_attempt(output, training_signature_payload("sft", vars(args), input_config, __file__), args.resume)
    processor = load_processor(args.model_path, args.revision, input_config)
    records = read_jsonl(args.manifest, args.limit)
    if not records:
        raise ValueError("Empty training manifest")
    dataset = prepare_dataset(records, processor, args.image_root, input_config, attempt.directory / "input_audit.jsonl")
    attempt.publish_file("input_audit.jsonl")
    model = load_model(args.model_path, args.revision, adapter_path=args.initial_adapter,
                       trainable=True, seed=args.seed)
    if not args.initial_adapter:
        model = attach_language_lora(model)
    inventory = trainable_inventory(model)
    attempt.artifact("trainable_parameters.json", inventory)
    initial_adapter_hash = adapter_fingerprint(model)
    initial_visual_hash = frozen_vision_fingerprint(model)
    if args.resume:
        original = json.loads((output / "resolved_config.json").read_text())
        if original.get("initial_adapter_fingerprint") != initial_adapter_hash or original.get("initial_visual_fingerprint") != initial_visual_hash:
            raise RuntimeError("Reconstructed common W differs from the original run")
    training_config = SFTConfig(
        output_dir=str(output), per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=1, max_steps=args.max_steps,
        learning_rate=args.learning_rate, weight_decay=0.0,
        warmup_steps=.03, lr_scheduler_type="constant_with_warmup", optim="adamw_torch",
        max_grad_norm=1.0, bf16=True, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=None, packing=False, completion_only_loss=True, assistant_only_loss=False,
        shuffle_dataset=False, train_sampling_strategy="sequential", remove_unused_columns=False,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=4,
        save_only_model=False, logging_steps=1, report_to="none", eval_strategy="no",
        seed=args.seed, data_seed=args.seed, dataloader_num_workers=0, dataloader_pin_memory=False,
    )
    run_callback = make_run_callback(output, model, attempt)
    trainer = AuditedSFTTrainer(
        model=model, args=training_config, train_dataset=dataset, processing_class=processor,
        callbacks=[run_callback],
    )
    first = dataset[0]
    batch = trainer.data_collator([first])
    labels = batch["labels"][0]
    supervised = labels[labels != -100].tolist()
    decoded_labels = processor.tokenizer.decode(supervised, skip_special_tokens=True).strip()
    expected = " ".join(records[0]["target_labels"])
    if decoded_labels != expected:
        raise RuntimeError(f"Completion-only mask failed: expected {expected!r}, found {decoded_labels!r}")
    attempt.artifact("label_mask_audit.json", {
        "qid": records[0]["qid"], "expected_target": expected,
        "supervised_text": decoded_labels, "supervised_token_ids": supervised,
        "supervised_tokens": len(supervised), "masked_tokens": int((labels == -100).sum()),
        "image_grid_thw": batch.get("image_grid_thw").tolist() if "image_grid_thw" in batch else None,
        "passed": True,
    })
    original_collator = trainer.data_collator

    def audited_collator(examples):
        batch_output = original_collator(examples)
        batch_output["_recipeqa_metadata"] = [{
                "qid": example["qid"], "recipe_id": example["recipe_id"],
                "presentation_id": example["presentation_id"], "visit_id": example.get("visit_id"),
                "display_perm": example["display_perm"], "input_hash": example["input_hash"],
            } for example in examples]
        return batch_output

    trainer.data_collator = audited_collator
    planned_visits = args.max_steps * args.gradient_accumulation_steps if args.max_steps > 0 else len(records)
    attempt.artifact("resolved_config.json", {
        "method": "sft", "cli": vars(args), "input": asdict(input_config),
        "trainer": training_config.to_dict(), "versions": package_versions(),
        "model_revision_resolved": getattr(model.config, "_commit_hash", None),
        "manifest_sha256": sha256_file(args.manifest), "manifest_rows": len(records),
        "planned_prompt_visits_if_full_accumulation": planned_visits,
        "passes_over_visit_stream_if_full_accumulation": planned_visits / len(records),
        "max_steps_overrides_num_train_epochs": args.max_steps > 0,
        "actual_prompt_visits_source": "actual_prompt_visits.jsonl",
        "initial_adapter_fingerprint": initial_adapter_hash,
        "initial_visual_fingerprint": initial_visual_hash,
        "resume_is_full_state": bool(args.resume),
        "run_signature_sha256": attempt.signature_sha256,
    })
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output / "adapter_final"))
    processor.save_pretrained(str(output / "adapter_final"))
    trainer.save_state()
    final_adapter_hash = adapter_fingerprint(trainer.model)
    final_visual_hash = frozen_vision_fingerprint(trainer.model)
    if final_visual_hash != initial_visual_hash:
        raise RuntimeError("Frozen visual weights changed during training")
    if final_adapter_hash == initial_adapter_hash and trainer.state.global_step > 0:
        raise RuntimeError("No adapter weight changed; this is not an effective SFT training run")
    attempt.finish({
        "status": "completed", "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch, "metrics": result.metrics,
        "elapsed_seconds_including_preprocessing": time.monotonic() - started,
        "initial_adapter_fingerprint": initial_adapter_hash, "final_adapter_fingerprint": final_adapter_hash,
        "initial_visual_fingerprint": initial_visual_hash, "final_visual_fingerprint": final_visual_hash,
        "visual_weights_unchanged": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    })


if __name__ == "__main__":
    main()
