#!/usr/bin/env python3
"""No-reference-KL Exact/Pair GRPO with identical input visits and dual reward logs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recipeqa.metrics import score_order
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
    parser.add_argument("--manifest", required=True, help="Ordered, pre-generated unique prompt visit stream")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reward", choices=("exact", "pair"), required=True)
    parser.add_argument("--initial-adapter")
    parser.add_argument("--resume")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--save-steps", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--recipe-token-budget", type=int, default=1536)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    return parser.parse_args()


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) == 1 and completion[0].get("role") == "assistant":
        content = completion[0]["content"]
        if isinstance(content, str):
            return content
    raise TypeError(f"Unexpected native completion format: {type(completion).__name__}")


class LoggedOrderingReward:
    def __init__(self, output_dir, selected_reward, num_generations, attempt=None):
        self.output_dir = Path(output_dir)
        self.selected_reward = selected_reward
        self.num_generations = num_generations
        self.__name__ = f"recipeqa_{selected_reward}"
        self.batch_index = 0
        self.attempt = attempt

    def __call__(self, completions, display_perm, qid, recipe_id, presentation_id,
                 target_order=None, completion_ids=None, trainer_state=None, visit_id=None, **kwargs):
        count = len(completions)
        if count % self.num_generations:
            raise RuntimeError("Completion batch does not contain complete G-sized groups")
        if target_order is None:
            target_order = [[0, 1, 2, 3]] * count
        if any(len(values) != count for values in (display_perm, qid, recipe_id, presentation_id, target_order)):
            raise RuntimeError("Reward metadata does not align with completions")
        raw = [completion_text(value) for value in completions]
        scores = [score_order(text, permutation, target_order=target)
                  for text, permutation, target in zip(raw, display_perm, target_order)]
        rewards = [float(score[self.selected_reward]) for score in scores]
        state_step = getattr(trainer_state, "global_step", None)
        with (self.output_dir / "reward_groups.jsonl").open("a", encoding="utf-8") as groups, \
             (self.output_dir / "rollouts.jsonl").open("a", encoding="utf-8") as rollouts:
            for start in range(0, count, self.num_generations):
                end = start + self.num_generations
                keys = [(qid[i], presentation_id[i], tuple(display_perm[i])) for i in range(start, end)]
                if len(set(keys)) != 1:
                    raise RuntimeError(f"GRPO group mixed distinct inputs: {keys}")
                group_scores = scores[start:end]
                selected = rewards[start:end]
                mean = statistics.mean(selected)
                std = statistics.stdev(selected)
                attempt_id = self.attempt.attempt_id if self.attempt is not None else "untracked"
                group_key = f"{attempt_id}:{state_step}:{self.batch_index}:{start // self.num_generations}"
                summary = {
                    "global_step_before_update": state_step, "batch_index": self.batch_index,
                    "group_id": group_key, "qid": qid[start], "recipe_id": recipe_id[start],
                    "presentation_id": presentation_id[start], "display_perm": display_perm[start],
                    "visit_id": visit_id[start] if visit_id is not None else None,
                    "selected_reward": self.selected_reward, "G": self.num_generations,
                    "valid_count": sum(s["valid"] for s in group_scores),
                    "exact_rewards": [s["exact"] for s in group_scores],
                    "pair_rewards": [s["pair"] for s in group_scores],
                    "exact_zero_variance": len({s["exact"] for s in group_scores}) == 1,
                    "pair_zero_variance": len({s["pair"] for s in group_scores}) == 1,
                    "selected_mean": mean, "selected_sample_std": std,
                    "reference_advantages": [(r - mean) / (std + 1e-4) for r in selected],
                    "advantage_note": "Offline sample-std audit; trainer owns the actual gradient computation",
                }
                if self.attempt is not None:
                    summary.update(attempt_id=self.attempt.attempt_id,
                                   run_signature_sha256=self.attempt.signature_sha256)
                    with (self.attempt.directory / "reward_groups.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
                groups.write(json.dumps(summary, ensure_ascii=False) + "\n")
                for i in range(start, end):
                    record = {
                        "global_step_before_update": state_step, "group_id": group_key,
                        "sample_index": i - start, "qid": qid[i], "recipe_id": recipe_id[i],
                        "presentation_id": presentation_id[i], "display_perm": display_perm[i],
                        "target_order": target_order[i], "raw_response": raw[i],
                        "selected_reward": self.selected_reward, "reward": rewards[i], "score": scores[i],
                        "completion_tokens": len(completion_ids[i]) if completion_ids is not None else None,
                        "input_tokens": kwargs.get("input_tokens", [None] * count)[i],
                        "input_hash": kwargs.get("input_hash", [None] * count)[i],
                        "visit_id": visit_id[i] if visit_id is not None else None,
                    }
                    if self.attempt is not None:
                        record.update(attempt_id=self.attempt.attempt_id,
                                      run_signature_sha256=self.attempt.signature_sha256)
                        with (self.attempt.directory / "rollouts.jsonl").open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    rollouts.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_metric = kwargs.get("log_metric")
        if log_metric is not None:
            log_metric("recipeqa/valid", statistics.mean(score["valid"] for score in scores))
            log_metric("recipeqa/exact", statistics.mean(score["exact"] for score in scores))
            log_metric("recipeqa/pair", statistics.mean(score["pair"] for score in scores))
        self.batch_index += 1
        return rewards


def main():
    args = arguments()
    ensure_data_disk(args.model_path, args.manifest, args.image_root, args.output_dir)
    if args.num_generations < 2 or args.generation_batch_size % args.num_generations:
        raise ValueError("generation batch must contain complete groups with G >= 2")
    # With microbatch=1 and one process, generation-batch size equals the number
    # of buffered microsteps. Keeping it equal to grad accumulation puts every
    # saved optimizer boundary at a fresh generation boundary for clean resume.
    if args.generation_batch_size != args.gradient_accumulation_steps:
        raise ValueError("This single-GPU protocol requires generation_batch_size == gradient_accumulation_steps")
    if args.initial_adapter:
        ensure_data_disk(args.initial_adapter)
    if args.resume:
        ensure_data_disk(args.resume)
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            if not (Path(args.resume) / name).is_file():
                raise RuntimeError(f"Full resume requires {name}")
    output = Path(args.output_dir)
    assert_training_destination(output, args.resume)

    import torch
    from trl import GRPOConfig, GRPOTrainer

    started = time.monotonic()
    input_config = InputConfig(args.min_pixels, args.max_pixels, args.recipe_token_budget, args.max_input_tokens)
    attempt = begin_training_attempt(output, training_signature_payload(f"grpo_{args.reward}", vars(args), input_config, __file__), args.resume)
    processor = load_processor(args.model_path, args.revision, input_config)
    records = read_jsonl(args.manifest, args.limit)
    if len(records) % (args.generation_batch_size // args.num_generations):
        raise ValueError("The visit stream must be divisible by the independent groups per generation batch")
    if not records:
        raise ValueError("Empty training manifest")
    dataset = prepare_dataset(records, processor, args.image_root, input_config, attempt.directory / "input_audit.jsonl")
    attempt.publish_file("input_audit.jsonl")
    model = load_model(args.model_path, args.revision, adapter_path=args.initial_adapter,
                       trainable=True, seed=args.seed)
    if not args.initial_adapter:
        model = attach_language_lora(model)
    attempt.artifact("trainable_parameters.json", trainable_inventory(model))
    initial_adapter_hash = adapter_fingerprint(model)
    initial_visual_hash = frozen_vision_fingerprint(model)
    if args.resume:
        original = json.loads((output / "resolved_config.json").read_text())
        if original.get("initial_adapter_fingerprint") != initial_adapter_hash or original.get("initial_visual_fingerprint") != initial_visual_hash:
            raise RuntimeError("Reconstructed common W differs from the original run")
    training_config = GRPOConfig(
        output_dir=str(output), per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=args.generation_batch_size, num_generations=args.num_generations,
        num_train_epochs=1, max_steps=args.max_steps,
        learning_rate=args.learning_rate, weight_decay=0.0,
        warmup_steps=.03, lr_scheduler_type="constant_with_warmup", optim="adamw_torch",
        max_grad_norm=1.0, bf16=True, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=0.0, loss_type="grpo", scale_rewards="group", epsilon=.2, epsilon_high=.2,
        num_iterations=1, max_completion_length=32, temperature=1.0, top_p=1.0, top_k=0,
        use_vllm=False, use_transformers_continuous_batching=False,
        shuffle_dataset=False, train_sampling_strategy="sequential", remove_unused_columns=False,
        mask_truncated_completions=False,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=4, save_only_model=False,
        logging_steps=1, log_completions=False, report_to="none", eval_strategy="no",
        seed=args.seed, data_seed=args.seed, dataloader_num_workers=0, dataloader_pin_memory=False,
    )
    reward = LoggedOrderingReward(output, args.reward, args.num_generations, attempt)
    trainer = GRPOTrainer(
        model=model, reward_funcs=reward, args=training_config,
        train_dataset=dataset, processing_class=processor,
        callbacks=[make_run_callback(output, model, attempt)],
    )
    if trainer.accelerator.num_processes != 1:
        raise RuntimeError("This frozen implementation is single-process/single-GPU")
    planned_visits = args.max_steps * (args.generation_batch_size // args.num_generations) if args.max_steps > 0 else len(records)
    attempt.artifact("resolved_config.json", {
        "method": f"grpo_{args.reward}", "cli": vars(args), "input": asdict(input_config),
        "trainer": training_config.to_dict(), "versions": package_versions(),
        "model_revision_resolved": getattr(model.config, "_commit_hash", None),
        "manifest_sha256": sha256_file(args.manifest), "manifest_rows": len(records),
        "planned_prompt_visits": planned_visits, "planned_completions": planned_visits * args.num_generations,
        "independent_prompt_groups_per_update": args.generation_batch_size // args.num_generations,
        "passes_over_visit_stream": planned_visits / len(records), "zero_variance_resampling": False,
        "max_steps_overrides_num_train_epochs": args.max_steps > 0,
        "initial_adapter_fingerprint": initial_adapter_hash,
        "initial_visual_fingerprint": initial_visual_hash, "resume_is_full_state": bool(args.resume),
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
    # An Exact pilot may legitimately receive only constant-reward groups. Keep
    # that outcome explicit; it does not pass the effective-training gate.
    attempt.finish({
        "status": "completed", "effective_adapter_update": final_adapter_hash != initial_adapter_hash,
        "global_step": trainer.state.global_step, "epoch": trainer.state.epoch, "metrics": result.metrics,
        "elapsed_seconds_including_preprocessing": time.monotonic() - started,
        "initial_adapter_fingerprint": initial_adapter_hash, "final_adapter_fingerprint": final_adapter_hash,
        "initial_visual_fingerprint": initial_visual_hash, "final_visual_fingerprint": final_visual_hash,
        "visual_weights_unchanged": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    })


if __name__ == "__main__":
    main()
