#!/usr/bin/env python3
"""CPU-only installed-Trainer remainder probe: 9 records, accumulation 8.

Uses a fresh one-parameter toy model, never the project VLM or any project data.
The result also records whether SFTTrainer shares Trainer's inner training loop.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recipeqa.modeling import ensure_data_disk, hash_json, package_versions, sha256_file, write_json


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def implementation_record(owner, method):
    function = getattr(owner, method, None)
    if function is None:
        return {"present": False}
    actual_owner = next((ancestor for ancestor in owner.__mro__ if method in ancestor.__dict__), None)
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = None
    return {"present": True, "defining_class": actual_owner.__module__ + "." + actual_owner.__name__ if actual_owner else None,
            "function_module": getattr(function, "__module__", None),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest() if source is not None else None,
            "source": source}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    ensure_data_disk(output)
    require(ROOT.is_relative_to("/root/autodl-tmp") and ROOT.stat().st_dev != Path("/").stat().st_dev,
            "Run this installed-environment probe only from the separate project data disk")
    for name in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TMPDIR", "TORCH_HOME"):
        require(bool(os.environ.get(name)), "Missing data-disk environment: " + name)
        ensure_data_disk(os.environ[name])
    output.parent.mkdir(parents=True, exist_ok=True)
    toy_dir = output.parent / (output.stem + "_toy_trainer")
    require(not output.exists() and not toy_dir.exists(), "Choose fresh output; never overwrite a prior probe")
    # Set before importing torch/Transformers; this child cannot claim a GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    started = time.monotonic()
    report = {"status": "started_cpu_remainder_probe", "source_sha256": sha256_file(__file__),
              "started_utc": datetime.now(timezone.utc).isoformat(), "gpu_execution": False,
              "real_model_loaded": False, "project_data_read": False,
              "records": 9, "microbatch": 1, "accumulation": 8, "epochs": 1}
    try:
        import torch
        from torch.utils.data import Dataset, SequentialSampler
        from transformers import Trainer, TrainerCallback, TrainingArguments
        from trl import SFTTrainer

        torch.set_num_threads(1)
        require(not torch.cuda.is_initialized(), "CUDA was unexpectedly initialized before CPU probe")
        methods = ("train", "_inner_training_loop", "set_initial_training_values", "_set_initial_training_values", "_get_train_sampler")
        report["installed_implementation"] = {
            "Trainer": {name: implementation_record(Trainer, name) for name in methods},
            "SFTTrainer": {name: implementation_record(SFTTrainer, name) for name in methods},
            "sft_shares_inner_training_loop": getattr(SFTTrainer, "_inner_training_loop", None) is getattr(Trainer, "_inner_training_loop", None),
            "sft_shares_train_entrypoint": getattr(SFTTrainer, "train", None) is getattr(Trainer, "train", None),
        }
        require(report["installed_implementation"]["sft_shares_inner_training_loop"],
                "SFTTrainer overrides the inner loop; direct Trainer boundary is not transferable without review")

        class TinyDataset(Dataset):
            def __len__(self):
                return 9

            def __getitem__(self, index):
                return {"input_ids": torch.tensor([index + 1], dtype=torch.float32),
                        "labels": torch.tensor([2 * (index + 1)], dtype=torch.float32),
                        "sample_id": index}

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.1))

            def forward(self, input_ids, labels=None, **kwargs):
                prediction = self.weight * input_ids.float()
                return {"loss": ((prediction - labels.float())**2).mean(), "logits": prediction}

        visits, optimizer_events, step_end_events = [], [], []

        class TinySequentialTrainer(Trainer):
            def _get_train_sampler(self, *positional, **keyword):
                return SequentialSampler(self.train_dataset)

            def training_step(self, model, inputs, *positional, **keyword):
                identifiers = inputs.pop("sample_id").reshape(-1).tolist()
                before = self.state.global_step
                loss = super().training_step(model, inputs, *positional, **keyword)
                visits.extend({"sample_id": int(identifier), "global_step_before_update": before,
                               "consumption_point": "training_step_forward_backward_returned"}
                              for identifier in identifiers)
                return loss

        class Events(TrainerCallback):
            def on_optimizer_step(self, arguments, state, control, **kwargs):
                optimizer_events.append({"global_step_before_increment": state.global_step,
                                         "weight_after_optimizer": float(kwargs["model"].weight.detach().item())})

            def on_step_end(self, arguments, state, control, **kwargs):
                step_end_events.append({"global_step": state.global_step,
                                        "weight": float(kwargs["model"].weight.detach().item())})

        training_args = TrainingArguments(
            output_dir=str(toy_dir), use_cpu=True, per_device_train_batch_size=1,
            gradient_accumulation_steps=8, num_train_epochs=1, max_steps=-1,
            learning_rate=.001, weight_decay=0, lr_scheduler_type="constant", warmup_steps=0,
            optim="adamw_torch", max_grad_norm=1, bf16=False, fp16=False,
            dataloader_num_workers=0, dataloader_pin_memory=False, dataloader_drop_last=False,
            save_strategy="no", eval_strategy="no", logging_strategy="no", report_to=[],
            disable_tqdm=True, remove_unused_columns=False, seed=42, data_seed=42,
        )
        trainer = TinySequentialTrainer(model=TinyModel(), args=training_args,
                                        train_dataset=TinyDataset(), callbacks=[Events()])
        result = trainer.train()
        require([visit["sample_id"] for visit in visits] == list(range(9)), "Installed Trainer did not consume exactly nine sequential records")
        require([visit["global_step_before_update"] for visit in visits] == [0] * 8 + [1],
                "Installed Trainer accumulation boundary is not 8+1")
        require(trainer.state.global_step == result.global_step == 2, "Installed Trainer did not perform two optimizer updates")
        require([event["global_step"] for event in step_end_events] == [1, 2], "Optimizer-step end events differ")
        require(len(optimizer_events) == 2, "Expected two actual optimizer callback events")
        require(not torch.cuda.is_initialized() and trainer.model.weight.device.type == "cpu",
                "Probe unexpectedly used CUDA")
        counts = Counter(visit["global_step_before_update"] for visit in visits)
        report.update(status="passed_cpu_remainder_probe", passed=True, versions=package_versions(),
                      visits=visits, optimizer_events=optimizer_events, step_end_events=step_end_events,
                      global_step=trainer.state.global_step, epoch=trainer.state.epoch,
                      actual_accumulation_examples=[counts[0], counts[1]], last_accumulation_examples=counts[1],
                      cuda_initialized=False, model_device=str(trainer.model.weight.device),
                      training_arguments=training_args.to_dict(),
                      full_data_expected_updates=math.ceil(7558 / 8), full_data_expected_last_examples=7558 % 8,
                      interpretation="Installed shared Trainer inner loop consumes a final partial accumulation group; full7558 exposure still requires its own audit",
                      metrics=result.metrics)
    except BaseException as error:
        report.update(status="failed_cpu_remainder_probe", passed=False,
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output, report)
    print(json.dumps({"status": report["status"], "global_step": report["global_step"],
                      "actual_accumulation_examples": report["actual_accumulation_examples"],
                      "sft_shares_inner_training_loop": report["installed_implementation"]["sft_shares_inner_training_loop"],
                      "output": str(output)}))


if __name__ == "__main__":
    main()
