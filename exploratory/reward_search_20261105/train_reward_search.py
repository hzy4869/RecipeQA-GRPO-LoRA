#!/usr/bin/env python3
"""Post-hoc reward search from one explicitly identified pretrained SFT adapter.

Imports frozen input/model helpers without changing them.  Training paths and
caches are checked against the rented machine's data disk.  No test evaluation,
checkpoint selection, auto-resampling, or model download occurs in this runner.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from rewards import LoggedSearchReward, REWARD_FORMULAS, REWARD_NAMES, reward_table
from recipeqa.modeling import (
    InputConfig, adapter_fingerprint, assert_training_destination,
    begin_training_attempt, ensure_data_disk, frozen_vision_fingerprint,
    hash_json, load_model, load_processor, make_run_callback, package_versions,
    prepare_dataset, read_jsonl, sha256_file, training_signature_payload,
    trainable_inventory, write_json,
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--method", choices=("grpo", "sft"), default="grpo")
    result.add_argument("--model-path")
    result.add_argument("--revision")
    result.add_argument("--manifest", help="Pre-generated ordered training visit stream; never a test manifest")
    result.add_argument("--image-root")
    result.add_argument("--output-dir")
    result.add_argument("--initial-adapter", help="Required common SFT start; weights loaded trainably, optimizer reset")
    result.add_argument("--resume", help="Optional complete checkpoint of THIS run, never another start adapter")
    result.add_argument("--reward", choices=REWARD_NAMES, default="pair")
    result.add_argument("--scale-rewards", choices=("group", "none"), default="group",
                        help="none is a separate normalization ablation, implemented with TRL scale_rewards='none'")
    result.add_argument("--max-steps", type=int, default=256)
    result.add_argument("--save-steps", type=int, default=64)
    result.add_argument("--save-total-limit", type=int, default=4)
    result.add_argument("--learning-rate", type=float,
                        help="Defaults: GRPO=5e-6; SFT=5e-5. Always recorded explicitly in resolved config")
    result.add_argument("--gradient-accumulation-steps", type=int, default=8)
    result.add_argument("--generation-batch-size", type=int, default=8)
    result.add_argument("--num-generations", type=int, choices=(4, 8), default=4)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--min-pixels", type=int, default=65536)
    result.add_argument("--max-pixels", type=int, default=262144)
    result.add_argument("--recipe-token-budget", type=int, default=1536)
    result.add_argument("--max-input-tokens", type=int, default=4096)
    result.add_argument("--reward-table", action="store_true", help="Print mathematical reward values; no GPU or data needed")
    result.add_argument("--cpu-smoke", action="store_true", help="Run CPU-only reward, layout and logging checks")
    result.add_argument("--check-api", action="store_true", help="Inspect installed TRL API without loading model or data")
    return result


def validate_arguments(args):
    for name in ("model_path", "manifest", "image_root", "output_dir", "initial_adapter"):
        if not getattr(args, name):
            raise ValueError(f"Training requires --{name.replace('_', '-')}")
    if args.max_steps <= 0 or args.save_steps <= 0 or args.max_steps % args.save_steps:
        raise ValueError("Positive max_steps must be divisible by positive save_steps")
    if args.save_total_limit < args.max_steps // args.save_steps:
        raise ValueError("Keep all planned checkpoints; increase --save-total-limit")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("Gradient accumulation must be positive")
    if args.method == "grpo":
        if args.generation_batch_size % args.num_generations:
            raise ValueError("Generation batch must contain complete groups")
        if args.generation_batch_size != args.gradient_accumulation_steps:
            raise ValueError("Single-GPU clean-boundary protocol requires generation_batch_size == gradient_accumulation_steps")
    elif args.scale_rewards != "group" or args.reward != "pair":
        raise ValueError("Reward/scaling switches apply only to GRPO; omit them for SFT")
    if args.learning_rate is None:
        args.learning_rate = 5e-6 if args.method == "grpo" else 5e-5
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    if not 0 < args.temperature <= 2:
        raise ValueError("Sampling temperature must be in (0,2]")
    ensure_data_disk(args.model_path, args.manifest, args.image_root,
                     args.output_dir, args.initial_adapter)
    if args.resume:
        ensure_data_disk(args.resume)
    # Frozen helper checks existing env vars.  Require their presence on AutoDL
    # too, so absent TMPDIR/HF_HOME cannot silently fall back to the system disk.
    if sys.platform.startswith("linux") and Path("/root/autodl-tmp").is_dir():
        required = ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TMPDIR", "TORCH_HOME")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise ValueError(f"Set data-disk environment variables before launch: {missing}")
    assert_training_destination(args.output_dir, args.resume)


def check_api():
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    grpo_required = {
        "generation_batch_size", "num_generations", "scale_rewards", "loss_type",
        "train_sampling_strategy", "shuffle_dataset", "use_transformers_continuous_batching",
        "epsilon_high", "mask_truncated_completions", "num_iterations",
    }
    sft_required = {"completion_only_loss", "assistant_only_loss", "train_sampling_strategy",
                    "shuffle_dataset", "max_length"}
    actual_grpo = {field.name for field in fields(GRPOConfig)}
    actual_sft = {field.name for field in fields(SFTConfig)}
    missing = {"grpo": sorted(grpo_required - actual_grpo),
               "sft": sorted(sft_required - actual_sft)}
    result = {
        "versions": package_versions(), "missing_fields": missing,
        "grpo_trainer_signature": str(inspect.signature(GRPOTrainer.__init__)),
        "sft_trainer_signature": str(inspect.signature(SFTTrainer.__init__)),
        "scale_rewards_annotation": str(next(field.type for field in fields(GRPOConfig)
                                             if field.name == "scale_rewards")),
        "scale_none_mapping": "none",
        "note": "Field inspection only; real GPU pilot is still required for end-to-end API validation",
    }
    print(json.dumps(result, indent=2))
    if any(missing.values()):
        raise RuntimeError("Installed TRL API lacks required fields")


def validate_stream(records, args):
    if not records:
        raise ValueError("Empty training manifest")
    if any(not record.get("visit_id") for record in records):
        raise ValueError("Every training record must have an explicit visit_id")
    if len({str(record["visit_id"]) for record in records}) != len(records):
        raise ValueError("Visit IDs must be unique even when questions repeat")
    if any("target_order" not in record for record in records):
        raise ValueError("Every training record requires explicit target_order")
    if args.method == "grpo":
        groups_per_update = args.generation_batch_size // args.num_generations
        visits = args.max_steps * groups_per_update
    else:
        visits = args.max_steps * args.gradient_accumulation_steps
    if len(records) < visits:
        raise ValueError(f"Manifest has {len(records)} visits but {visits} are planned; do not silently cycle stream")
    if args.method == "grpo" and len(records) % groups_per_update:
        raise ValueError("Visit stream must contain complete independent-group batches")
    return visits


def make_signature(args, input_config):
    signature = training_signature_payload(
        f"exploratory_{args.method}_{args.reward if args.method == 'grpo' else 'supervised'}",
        vars(args), input_config, __file__,
    )
    signature["schema_version"] = 2
    signature["post_hoc_exploration"] = True
    signature["initial_adapter_semantics"] = "shared previously trained SFT weights; new optimizer for every branch"
    signature["code_sha256"]["exploratory_rewards.py"] = sha256_file(HERE / "rewards.py")
    signature["fixed_policy"]["grpo_scale_rewards"] = args.scale_rewards
    signature["fixed_policy"]["num_generations"] = args.num_generations
    signature["fixed_policy"]["temperature"] = args.temperature
    signature["reward_formula"] = REWARD_FORMULAS[args.reward] if args.method == "grpo" else None
    return signature


def make_config(args):
    from trl import GRPOConfig, SFTConfig

    common = dict(
        output_dir=args.output_dir, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=1, max_steps=args.max_steps,
        learning_rate=args.learning_rate, weight_decay=0.0,
        warmup_steps=.03, lr_scheduler_type="constant_with_warmup", optim="adamw_torch",
        max_grad_norm=1.0, bf16=True, fp16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        shuffle_dataset=False, train_sampling_strategy="sequential", remove_unused_columns=False,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        save_only_model=False, logging_steps=1, report_to="none", eval_strategy="no",
        seed=args.seed, data_seed=args.seed, dataloader_num_workers=0, dataloader_pin_memory=False,
    )
    if args.method == "sft":
        return SFTConfig(**common, max_length=None, packing=False,
                         completion_only_loss=True, assistant_only_loss=False)
    return GRPOConfig(
        **common, generation_batch_size=args.generation_batch_size,
        num_generations=args.num_generations, beta=0.0, loss_type="grpo",
        scale_rewards=args.scale_rewards,
        epsilon=.2, epsilon_high=.2, num_iterations=1,
        max_completion_length=32, temperature=args.temperature, top_p=1.0, top_k=0,
        use_vllm=False, use_transformers_continuous_batching=False,
        mask_truncated_completions=False, log_completions=False,
    )


def make_sft_trainer(model, config, dataset, processor, records, callback, attempt):
    from trl import SFTTrainer

    class AuditedSFTTrainer(SFTTrainer):
        def training_step(self, model, inputs, *positional, **keyword):
            metadata = inputs.pop("_recipeqa_metadata")
            step = self.state.global_step
            loss = super().training_step(model, inputs, *positional, **keyword)
            for example in metadata:
                attempt.append("actual_prompt_visits.jsonl", {
                    **example, "global_step_before_update": step,
                    "consumption_point": "training_step_forward_backward_returned",
                })
            return loss

    trainer = AuditedSFTTrainer(
        model=model, args=config, train_dataset=dataset, processing_class=processor,
        callbacks=[callback],
    )
    batch = trainer.data_collator([dataset[0]])
    labels = batch["labels"][0]
    supervised = labels[labels != -100].tolist()
    decoded = processor.tokenizer.decode(supervised, skip_special_tokens=True).strip()
    expected = " ".join(records[0]["target_labels"])
    if decoded != expected:
        raise RuntimeError(f"Completion-only mask failed: expected {expected!r}, found {decoded!r}")
    attempt.artifact("label_mask_audit.json", {
        "qid": records[0]["qid"], "expected_target": expected, "supervised_text": decoded,
        "supervised_token_ids": supervised, "supervised_tokens": len(supervised),
        "masked_tokens": int((labels == -100).sum()), "passed": True,
    })
    original_collator = trainer.data_collator

    def audited_collator(examples):
        result = original_collator(examples)
        result["_recipeqa_metadata"] = [{
            name: example.get(name) for name in (
                "qid", "recipe_id", "presentation_id", "visit_id", "display_perm",
                "input_hash", "input_tokens",
            )
        } for example in examples]
        return result

    trainer.data_collator = audited_collator
    return trainer


def train(args):
    validate_arguments(args)
    import torch
    from trl import GRPOTrainer

    started = time.monotonic()
    records = read_jsonl(args.manifest)
    planned_visits = validate_stream(records, args)
    config = make_config(args)
    output = Path(args.output_dir)
    input_config = InputConfig(args.min_pixels, args.max_pixels,
                               args.recipe_token_budget, args.max_input_tokens)
    attempt = begin_training_attempt(output, make_signature(args, input_config), args.resume)
    try:
        processor = load_processor(args.model_path, args.revision, input_config)
        dataset = prepare_dataset(records, processor, args.image_root, input_config,
                                  attempt.directory / "input_audit.jsonl")
        attempt.publish_file("input_audit.jsonl")
        model = load_model(args.model_path, args.revision,
                           adapter_path=args.initial_adapter, trainable=True, seed=args.seed)
        attempt.artifact("trainable_parameters.json", trainable_inventory(model))
        initial_adapter = adapter_fingerprint(model)
        initial_visual = frozen_vision_fingerprint(model)
        if args.resume:
            original = json.loads((output / "resolved_config.json").read_text())
            if (original["initial_adapter_fingerprint"] != initial_adapter
                    or original["initial_visual_fingerprint"] != initial_visual):
                raise RuntimeError("Reconstructed common initial adapter or visual model changed")
        callback = make_run_callback(output, model, attempt)
        if args.method == "sft":
            trainer = make_sft_trainer(model, config, dataset, processor, records, callback, attempt)
        else:
            reward = LoggedSearchReward(output, args.reward, args.num_generations,
                                        args.scale_rewards, attempt)
            trainer = GRPOTrainer(
                model=model, reward_funcs=reward, args=config, train_dataset=dataset,
                processing_class=processor, callbacks=[callback],
            )
        if trainer.accelerator.num_processes != 1:
            raise RuntimeError("Exploratory runner supports one GPU/process only")
        attempt.artifact("resolved_config.json", {
            "method": f"exploratory_{args.method}", "post_hoc_exploration": True,
            "cli": vars(args), "input": asdict(input_config), "trainer": config.to_dict(),
            "versions": package_versions(),
            "model_revision_resolved": getattr(model.config, "_commit_hash", None),
            "manifest_sha256": sha256_file(args.manifest), "manifest_rows": len(records),
            "planned_prompt_visits": planned_visits,
            "planned_completions": planned_visits * args.num_generations if args.method == "grpo" else None,
            "independent_prompt_groups_per_update": args.generation_batch_size // args.num_generations
            if args.method == "grpo" else None,
            "passes_over_visit_stream": planned_visits / len(records),
            "zero_variance_resampling": False, "max_steps_overrides_num_train_epochs": True,
            "reward_formula": REWARD_FORMULAS[args.reward] if args.method == "grpo" else None,
            "actual_prompt_visits_source": "reward_groups.jsonl" if args.method == "grpo"
            else "actual_prompt_visits.jsonl",
            "initial_adapter_fingerprint": initial_adapter,
            "initial_visual_fingerprint": initial_visual,
            "initial_optimizer": "reset unless explicit full resume",
            "resume_is_full_state": bool(args.resume),
            "run_signature_sha256": attempt.signature_sha256,
        })
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_model(str(output / "adapter_final"))
        processor.save_pretrained(str(output / "adapter_final"))
        trainer.save_state()
        final_adapter = adapter_fingerprint(trainer.model)
        final_visual = frozen_vision_fingerprint(trainer.model)
        if final_visual != initial_visual:
            raise RuntimeError("Frozen visual weights changed during training")
        if args.method == "sft" and final_adapter == initial_adapter:
            raise RuntimeError("No SFT adapter weights changed")
        if trainer.state.global_step != args.max_steps:
            raise RuntimeError("Training did not reach the complete planned optimizer-step budget")
        actual = read_jsonl(output / ("reward_groups.jsonl" if args.method == "grpo" else "actual_prompt_visits.jsonl"))
        if len(actual) != planned_visits:
            raise RuntimeError(f"Actual visits {len(actual)} != planned {planned_visits}")
        expected_ids = [str(record["visit_id"]) for record in records[:planned_visits]]
        if [str(row["visit_id"]) for row in actual] != expected_ids:
            raise RuntimeError("Actual consumed visit sequence differs from fixed manifest prefix")
        if args.method == "grpo" and len(read_jsonl(output / "rollouts.jsonl")) != planned_visits * args.num_generations:
            raise RuntimeError("Actual candidate budget differs from planned complete groups")
        attempt.finish({
            "status": "completed", "effective_adapter_update": final_adapter != initial_adapter,
            "scientific_outcome": "effective_training" if final_adapter != initial_adapter
            else "no_effective_adapter_update",
            "global_step": trainer.state.global_step, "epoch": trainer.state.epoch,
            "metrics": result.metrics, "actual_prompt_visits": len(actual),
            "actual_visit_sequence_verified": True,
            "elapsed_seconds_including_preprocessing": time.monotonic() - started,
            "initial_adapter_fingerprint": initial_adapter, "final_adapter_fingerprint": final_adapter,
            "initial_visual_fingerprint": initial_visual, "final_visual_fingerprint": final_visual,
            "visual_weights_unchanged": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
    except BaseException as error:
        attempt.artifact("attempt_failure.json", {
            "status": "failed", "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.monotonic() - started,
            "attempt_id": attempt.attempt_id, "run_signature_sha256": attempt.signature_sha256,
        })
        metadata_path = attempt.directory / "attempt.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(status="failed", failed_utc=datetime.now(timezone.utc).isoformat(),
                        error=f"{type(error).__name__}: {error}")
        write_json(metadata_path, metadata)
        raise


def main():
    args = parser().parse_args()
    modes = (args.reward_table, args.cpu_smoke, args.check_api)
    if sum(modes) > 1:
        raise ValueError("Choose one non-training inspection mode at a time")
    if args.reward_table:
        print(json.dumps({"formulas": REWARD_FORMULAS, "rows": reward_table()}, indent=2))
    elif args.check_api:
        check_api()
    elif args.cpu_smoke:
        import unittest
        import test_rewards
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_rewards))
        if not result.wasSuccessful():
            raise SystemExit(1)
    else:
        train(args)


if __name__ == "__main__":
    main()
