"""Shared, auditable Qwen3-VL input and LoRA utilities.

Only pixels, visible labels and the complete ordered recipe enter the prompt.
Source identifiers and filenames are provenance, never model-visible content.
Heavy dependencies are imported lazily so CPU-side audit tools remain usable.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Any
import uuid


LABELS = ("A", "B", "C", "D")
PROMPT_VERSION = "recipeqa-order-v1"
TASK_TEXT = (
    "The recipe steps below are in their intended order.\n"
    "Four photographs from this recipe are shown with labels A, B, C, and D.\n"
    "Return the image labels in the order that their depicted steps occur in the recipe.\n"
    "Use each label exactly once. Output only four labels separated by spaces."
)
PROCESS_TEXT = (
    "For each image, identify its corresponding numbered recipe step and give one short "
    "observation that is visible in the image. Use 'unclear' if a step cannot be identified. "
    "Then order the images according to the recipe. Use exactly this format:\n"
    "A: step <number or unclear> | <one short visible observation>\n"
    "B: step <number or unclear> | <one short visible observation>\n"
    "C: step <number or unclear> | <one short visible observation>\n"
    "D: step <number or unclear> | <one short visible observation>\n"
    "Order: <four image labels>"
)


@dataclass(frozen=True)
class InputConfig:
    min_pixels: int = 65536
    max_pixels: int = 262144
    recipe_token_budget: int = 1536
    max_input_tokens: int = 4096
    max_completion_tokens: int = 32
    prompt_version: str = PROMPT_VERSION


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def model_weight_identity(directory: str | Path, *, adapter: bool = False) -> dict:
    """Hash actual local weight bytes on every launch, never just path or mtime."""
    root = Path(directory).resolve()
    patterns = (("adapter_model*.safetensors", "adapter_model*.bin") if adapter else
                ("model*.safetensors", "pytorch_model*.bin"))
    weights = sorted({path for pattern in patterns for path in root.glob(pattern) if path.is_file()})
    if not weights:
        raise ValueError(f"No {'adapter' if adapter else 'base model'} weight files found in {root}")
    files = [{"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in weights]
    if any(row["bytes"] <= 0 for row in files):
        raise ValueError(f"Empty weight file in {root}")
    auxiliary_names = ("config.json", "adapter_config.json", "generation_config.json", "tokenizer.json",
                       "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
                       "chat_template.jinja", "chat_template.json", "preprocessor_config.json", "processor_config.json")
    auxiliary = [{"name": name, "sha256": sha256_file(root / name)} for name in auxiliary_names
                 if (root / name).is_file()]
    return {"directory": str(root), "weight_files": files, "weights_sha256": hash_json(files),
            "auxiliary_files": auxiliary, "bundle_sha256": hash_json({"weights": files, "auxiliary": auxiliary})}


def assert_training_destination(output_dir: str | Path, resume: str | None) -> None:
    output = Path(output_dir).resolve()
    if not resume:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Training output is not empty. Use an explicit full --resume or a new run directory; never append a fresh run to old logs.")
        return
    checkpoint = Path(resume).resolve()
    if checkpoint.parent != output or not checkpoint.name.startswith("checkpoint-"):
        raise ValueError("Full resume must use a checkpoint inside the same run output directory")
    for path in (output / "run_signature.json", output / "resolved_config.json"):
        if not path.is_file():
            raise ValueError(f"Cannot verify immutable identity for resume: missing {path}")
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                 "recipeqa_checkpoint_audit.json"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"Full resume is missing {checkpoint / name}")


def training_signature_payload(method: str, cli: dict, input_config: InputConfig, script_path: str | Path) -> dict:
    """Bind seed, budget, optimization source, all data bytes, Base and W bytes."""
    scientific_cli = {key: value for key, value in cli.items() if key not in ("resume", "output_dir")}
    source_root = Path(__file__).parent
    sources = [Path(script_path), Path(__file__), source_root / "metrics.py"]
    return {
        "schema_version": 1, "method": method, "scientific_cli": scientific_cli,
        "input": asdict(input_config), "versions": package_versions(),
        "manifest_sha256": sha256_file(cli["manifest"]),
        "base_identity": model_weight_identity(cli["model_path"]),
        "initial_W_adapter_identity": model_weight_identity(cli["initial_adapter"], adapter=True)
        if cli.get("initial_adapter") else None,
        "code_sha256": {path.name: sha256_file(path) for path in sources},
        "fixed_policy": {"num_train_epochs": 1, "microbatch": 1, "lora_r": 16, "lora_alpha": 32,
                         "lora_dropout": 0.0, "weight_decay": 0.0, "warmup_ratio": .03,
                         "scheduler": "constant_with_warmup", "max_grad_norm": 1.0,
                         "bf16": True, "attention": "sdpa", "max_completion_tokens": 32,
                         "shuffle_dataset": False, "train_sampling_strategy": "sequential",
                         "grpo_beta": 0.0, "grpo_loss_type": "grpo", "grpo_scale_rewards": "group",
                         "grpo_epsilon": .2, "grpo_num_iterations": 1},
    }


@dataclass
class TrainingAttempt:
    output: Path
    directory: Path
    attempt_id: str
    signature_sha256: str
    resume_global_step: int

    def append(self, filename: str, payload: dict) -> None:
        row = {**payload, "attempt_id": self.attempt_id, "run_signature_sha256": self.signature_sha256}
        encoded = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        for directory in (self.output, self.directory):
            with (directory / filename).open("a", encoding="utf-8") as handle:
                handle.write(encoded)

    def artifact(self, filename: str, payload: dict, *, latest: bool = False) -> None:
        write_json(self.directory / filename, payload)
        if latest or not (self.output / filename).exists():
            write_json(self.output / filename, payload)

    def publish_file(self, filename: str) -> None:
        if not (self.output / filename).exists():
            shutil.copy2(self.directory / filename, self.output / filename)

    def finish(self, outcome: dict) -> None:
        outcome = {**outcome, "attempt_id": self.attempt_id,
                   "run_signature_sha256": self.signature_sha256,
                   "resume_global_step": self.resume_global_step}
        self.artifact("train_outcome.json", outcome, latest=True)
        metadata = json.loads((self.directory / "attempt.json").read_text())
        metadata.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                        final_global_step=outcome["global_step"])
        write_json(self.directory / "attempt.json", metadata)


def begin_training_attempt(output_dir: str | Path, signature_payload: dict,
                           resume: str | None) -> TrainingAttempt:
    output = Path(output_dir).resolve()
    assert_training_destination(output, resume)
    signature = {"payload": signature_payload, "sha256": hash_json(signature_payload)}
    signature_path = output / "run_signature.json"
    resume_step = 0
    if resume:
        previous = json.loads(signature_path.read_text())
        if previous.get("sha256") != hash_json(previous.get("payload")):
            raise ValueError("Stored run signature is internally inconsistent")
        if previous != signature:
            old = previous.get("payload", {})
            changed = [key for key in set(old) | set(signature_payload) if old.get(key) != signature_payload.get(key)]
            raise ValueError(f"Resume changes immutable run conditions: {sorted(changed)}")
        checkpoint = Path(resume).resolve()
        audit = json.loads((checkpoint / "recipeqa_checkpoint_audit.json").read_text())
        if audit.get("run_signature_sha256") != signature["sha256"]:
            raise ValueError("Checkpoint belongs to another or unsigned run")
        state_hashes = audit.get("state_sha256", {})
        required_state = {"trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"}
        if set(state_hashes) != required_state:
            raise ValueError("Checkpoint lacks complete recovery-state content hashes")
        for name, expected_hash in state_hashes.items():
            if sha256_file(checkpoint / name) != expected_hash:
                raise ValueError(f"Checkpoint recovery state content changed: {name}")
        if audit.get("adapter_identity") != model_weight_identity(checkpoint, adapter=True):
            raise ValueError("Checkpoint adapter content changed or lacks a verified weight identity")
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        resume_step = state.get("global_step")
        if type(resume_step) is not int or resume_step < 0:
            raise ValueError("Checkpoint has an invalid global step")
    output.mkdir(parents=True, exist_ok=True)
    if not resume:
        write_json(signature_path, signature)
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = output / "attempts" / attempt_id
    directory.mkdir(parents=True)
    attempt = TrainingAttempt(output, directory, attempt_id, signature["sha256"], resume_step)
    write_json(directory / "attempt.json", {
        "attempt_id": attempt_id, "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_signature_sha256": signature["sha256"], "resume_from": str(Path(resume).resolve()) if resume else None,
        "resume_global_step": resume_step,
    })
    if resume:
        # Keep raw cost/history in its original attempt directories. Canonical
        # logs describe only the active optimizer trajectory, never 20+2=22.
        archive = directory / "superseded_before_resume"
        archive.mkdir()
        rules = {"gradient_audit.jsonl": ("global_step_before_update", False),
                 "reward_groups.jsonl": ("global_step_before_update", False),
                 "rollouts.jsonl": ("global_step_before_update", False),
                 "actual_prompt_visits.jsonl": ("global_step_before_update", False),
                 "step_times.jsonl": ("global_step", True), "trainer_metrics.jsonl": ("global_step", True)}
        for filename, (field, inclusive) in rules.items():
            path = output / filename
            if not path.exists():
                continue
            rows = read_jsonl(path)
            if any(type(row.get(field)) is not int for row in rows):
                raise ValueError(f"Cannot safely identify pre-resume trajectory in {filename}")
            shutil.copy2(path, archive / filename)
            kept = [row for row in rows if row[field] <= resume_step] if inclusive else [row for row in rows if row[field] < resume_step]
            temporary = path.with_suffix(path.suffix + ".partial")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in kept:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            temporary.replace(path)
        for name in ("train_outcome.json", "trainer_state.json", "adapter_final"):
            path = output / name
            if path.exists():
                shutil.move(str(path), str(archive / name))
        for path in output.glob("checkpoint-*"):
            if path.is_dir() and path.name.split("-")[-1].isdigit() and int(path.name.split("-")[-1]) > resume_step:
                shutil.move(str(path), str(archive / path.name))
    return attempt


def ensure_data_disk(*paths: str | Path) -> None:
    """Enforce the user's disk policy on the rented AutoDL machine."""
    data_disk = Path("/root/autodl-tmp")
    if platform.system() == "Linux" and data_disk.is_dir():
        for path in paths:
            resolved = Path(path).expanduser().resolve()
            if not resolved.is_relative_to(data_disk):
                raise ValueError(f"Path must be on the data disk /root/autodl-tmp: {resolved}")
        for name in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TMPDIR", "TORCH_HOME"):
            value = os.environ.get(name)
            if value and not Path(value).expanduser().resolve().is_relative_to(data_disk):
                raise ValueError(f"{name} must point to /root/autodl-tmp, got {value}")


def load_processor(model_path: str, revision: str | None, config: InputConfig):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path, revision=revision, local_files_only=True, trust_remote_code=False,
        min_pixels=config.min_pixels, max_pixels=config.max_pixels,
    )
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    size = getattr(processor.image_processor, "size", {})
    # Both the legacy attributes and modern size dictionary must agree. Reject
    # ignored settings rather than quietly running another visual token budget.
    actual_min = size.get("shortest_edge", getattr(processor.image_processor, "min_pixels", None))
    actual_max = size.get("longest_edge", getattr(processor.image_processor, "max_pixels", None))
    if actual_min != config.min_pixels or actual_max != config.max_pixels:
        raise RuntimeError(f"Processor pixel settings were not applied: size={size}")
    return processor


def load_model(model_path: str, revision: str | None = None, *, adapter_path: str | None = None,
               trainable: bool = False, seed: int = 42):
    import torch
    from transformers import Qwen3VLForConditionalGeneration, set_seed

    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this model execution path")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The frozen BF16 protocol requires a BF16-capable GPU")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, revision=revision, local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=trainable)
    model.config.use_cache = not trainable
    if not trainable:
        model.to("cuda")
        model.eval()
    return model


def attach_language_lora(model, *, rank: int = 16, alpha: int = 32):
    import torch
    from peft import LoraConfig, get_peft_model

    pattern = re.compile(
        r"(?:^|\.)language_model\.layers\.\d+\."
        r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
    )
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and pattern.search(name)]
    expected_layers = model.config.get_text_config().num_hidden_layers
    if len(targets) != 7 * expected_layers:
        raise RuntimeError(f"Language LoRA whitelist expected {7 * expected_layers} modules, found {len(targets)}")
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=targets, task_type="CAUSAL_LM",
    ))
    return model


def trainable_inventory(model) -> dict:
    names = []
    total = trainable = 0
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            if "lora_" not in name or ".language_model.layers." not in name:
                raise RuntimeError(f"Unapproved trainable parameter: {name}")
            names.append({"name": name, "shape": list(parameter.shape), "numel": parameter.numel()})
            trainable += parameter.numel()
    if not names:
        raise RuntimeError("No trainable LoRA parameters")
    return {"total_parameters": total, "trainable_parameters": trainable, "parameters": names}


def adapter_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_vision_fingerprint(model) -> str:
    digest = hashlib.sha256()
    found = False
    for name, parameter in model.named_parameters():
        if ".visual." in name or name.startswith("visual."):
            if parameter.requires_grad:
                raise RuntimeError(f"Vision parameter is trainable: {name}")
            found = True
            digest.update(name.encode())
            digest.update(parameter.detach().view(-1).view(__import__("torch").uint8).cpu().numpy().tobytes())
    if not found:
        raise RuntimeError("No visual weights were found for the freeze audit")
    return digest.hexdigest()


def load_images(record: dict, image_root: str | Path):
    root = Path(image_root).resolve()
    paths = record.get("image_paths_display")
    context = f"qid={record.get('qid')} presentation={record.get('presentation_id')}"
    if not isinstance(paths, list) or len(paths) != 4:
        raise ValueError(f"{context}: expected four resolved image paths, got {paths!r}")
    for label, path in zip(LABELS, paths):
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"{context} Image {label}: missing/unresolved image path; "
                             f"technical_status={record.get('technical_status', 'unspecified')}")
    from PIL import Image

    images = []
    for label, path in zip(LABELS, paths):
        full = (root / path).resolve()
        if not full.is_relative_to(root):
            raise ValueError(f"{context} Image {label}: image escaped image-root: {path}")
        try:
            with Image.open(full) as opened:
                images.append(opened.convert("RGB"))
        except (OSError, ValueError) as error:
            raise ValueError(f"{context} Image {label}: cannot decode {full}: {error}") from error
    return images


def _render_recipe(steps: list[dict], tokenizer, body_limit: int | None) -> str:
    rendered = []
    for ordinal, step in enumerate(steps, 1):
        title = str(step.get("title", "")).strip()
        body = str(step.get("body", "")).strip()
        if body_limit is not None:
            tokens = tokenizer.encode(body, add_special_tokens=False)
            body = tokenizer.decode(tokens[:body_limit], skip_special_tokens=True).strip()
        rendered.append(f"Step {ordinal}: {title}" + (f"\n{body}" if body else ""))
    return "\n\n".join(rendered)


def build_messages(recipe_text: str, *, process: bool = False) -> list[dict]:
    task = TASK_TEXT
    if process:
        task = task.replace("Use each label exactly once. Output only four labels separated by spaces.",
                            "Use each label exactly once in the final order.")
    content = [{"type": "text", "text": task + "\n\nOrdered recipe:\n" + recipe_text}]
    for label in LABELS:
        content.extend([{"type": "text", "text": f"\nImage {label}:"}, {"type": "image"}])
    content.append({"type": "text", "text": PROCESS_TEXT if process else "\nAnswer:"})
    return [{"role": "user", "content": content}]


def encode_messages(processor, messages: list[dict], images: list):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[text], images=[images], return_tensors="pt", padding=False,
                        truncation=False, add_special_tokens=False)
    encoded.pop("token_type_ids", None)
    return encoded


def prepare_record(record: dict, processor, image_root: str | Path, config: InputConfig,
                   *, process: bool = False) -> tuple[dict, dict]:
    """Uniform body clipping; no gold step selection, title dropping or image truncation."""
    tokenizer = processor.tokenizer
    steps = record["steps"]
    if not steps:
        raise ValueError(f"Empty recipe for {record['qid']}")
    images = load_images(record, image_root)
    raw_recipe = _render_recipe(steps, tokenizer, None)
    raw_tokens = len(tokenizer.encode(raw_recipe, add_special_tokens=False))
    title_recipe = _render_recipe(steps, tokenizer, 0)
    if len(tokenizer.encode(title_recipe, add_special_tokens=False)) > config.recipe_token_budget:
        raise ValueError(f"All step titles exceed recipe budget: {record['qid']}; revise protocol explicitly")
    largest_body = max(len(tokenizer.encode(str(s.get("body", "")), add_special_tokens=False)) for s in steps)
    low, high = 0, largest_body
    best = None
    title_messages = build_messages(title_recipe, process=process)
    title_encoded = encode_messages(processor, title_messages, images)
    title_template = processor.apply_chat_template(title_messages, tokenize=False, add_generation_prompt=True)
    visual_expansion = (title_encoded["input_ids"].shape[-1]
                        - len(tokenizer.encode(title_template, add_special_tokens=False)))
    # Binary search one identical per-step body cap, checking the actual expanded
    # multimodal input. This cannot silently remove later steps or image tokens.
    while low <= high:
        cap = (low + high) // 2
        recipe = _render_recipe(steps, tokenizer, cap)
        recipe_tokens = len(tokenizer.encode(recipe, add_special_tokens=False))
        if recipe_tokens > config.recipe_token_budget:
            high = cap - 1
            continue
        messages = build_messages(recipe, process=process)
        template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_tokens = len(tokenizer.encode(template, add_special_tokens=False)) + visual_expansion
        if input_tokens <= config.max_input_tokens:
            best = (cap, recipe, recipe_tokens, messages)
            low = cap + 1
        else:
            high = cap - 1
    if best is None:
        raise ValueError(f"Recipe titles plus four images exceed input budget: {record['qid']}")
    cap, recipe, recipe_tokens, messages = best
    encoded = encode_messages(processor, messages, images)
    if encoded["input_ids"].shape[-1] > config.max_input_tokens:
        raise RuntimeError("Expanded token accounting changed between identical image inputs")
    target = " ".join(record["target_labels"])
    if len(tokenizer.encode(target, add_special_tokens=False)) > config.max_completion_tokens:
        raise ValueError("Target exceeds output budget")
    prepared = {**record, "prompt": messages,
                "completion": [{"role": "assistant", "content": target}],
                "images": [str((Path(image_root) / p).resolve()) for p in record["image_paths_display"]]}
    grid = encoded.get("image_grid_thw")
    audit = {"qid": record["qid"], "recipe_id": record["recipe_id"],
             "presentation_id": record["presentation_id"], "recipe_tokens_raw": raw_tokens,
             "recipe_tokens_used": recipe_tokens, "uniform_body_cap": cap,
             "input_tokens": int(encoded["input_ids"].shape[-1]),
             "steps": len(steps), "titles_retained": len(steps),
             "body_clipped_step_ids": [s.get("source_step_id", i + 1) for i, s in enumerate(steps)
                                       if len(tokenizer.encode(str(s.get("body", "")), add_special_tokens=False)) > cap],
             "image_grid_thw": grid.tolist() if grid is not None else None,
             "image_token_count": int((encoded["input_ids"] == processor.image_token_id).sum())
             if hasattr(processor, "image_token_id") else None,
             "prompt_text_hash": hash_json(messages),
             "input_hash": hash_json({"messages": messages, "display_perm": record["display_perm"],
                                      "image_paths_display": record["image_paths_display"],
                                      "preprocessing": asdict(config)})}
    prepared["input_tokens"] = audit["input_tokens"]
    prepared["input_hash"] = audit["input_hash"]
    return prepared, audit


def hf_dataset(prepared: list[dict]):
    from datasets import Dataset, Image, List

    dataset = Dataset.from_list(prepared).cast_column("images", List(Image()))

    def to_rgb(batch):
        batch["images"] = [[image.convert("RGB") for image in images] for images in batch["images"]]
        return batch

    return dataset.with_transform(to_rgb)


def prepare_dataset(records: list[dict], processor, image_root: str | Path, config: InputConfig,
                    audit_path: str | Path):
    prepared = []
    path = Path(audit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            example, audit = prepare_record(record, processor, image_root, config)
            prepared.append(example)
            handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    return hf_dataset(prepared)


def package_versions() -> dict:
    names = ("torch", "torchvision", "transformers", "trl", "peft", "accelerate", "datasets",
             "huggingface_hub", "Pillow", "numpy", "tokenizers", "safetensors", "pyarrow")
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def make_run_callback(output_dir: str | Path, model, attempt: TrainingAttempt | None = None):
    import torch
    from transformers import TrainerCallback

    class RunAuditCallback(TrainerCallback):
        def __init__(self):
            self.start = time.monotonic()
            self.step_start = None
            self.gradient_audited = False

        def append(self, filename, payload):
            if attempt is not None:
                attempt.append(filename, payload)
                return
            with (Path(output_dir) / filename).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_start = time.monotonic()

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            entries, frozen_grads = [], []
            for name, parameter in kwargs["model"].named_parameters():
                if parameter.grad is None:
                    continue
                if not parameter.requires_grad:
                    frozen_grads.append(name)
                else:
                    entries.append((name, parameter.grad.detach().float().norm()))
            # One device-to-host synchronization rather than one per LoRA
            # tensor; each finite norm also excludes NaN/Inf gradient entries.
            norms = torch.stack([value for _, value in entries]).cpu().tolist() if entries else []
            nonfinite = [name for (name, _), value in zip(entries, norms) if not math.isfinite(value)]
            if nonfinite or frozen_grads:
                raise RuntimeError(f"Invalid gradient audit: nonfinite={nonfinite}, frozen={frozen_grads}")
            self.append("gradient_audit.jsonl", {"global_step_before_update": state.global_step,
                        "nonzero_gradient_tensors": sum(n > 0 for n in norms),
                        "trainable_gradient_tensors": len(norms), "l2_norm": sum(n * n for n in norms) ** .5})

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.append("step_times.jsonl", {"global_step": state.global_step, "epoch": state.epoch,
                        "optimizer_step_seconds": time.monotonic() - self.step_start,
                        "elapsed_seconds": time.monotonic() - self.start,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved()})

        def on_log(self, args, state, control, logs=None, **kwargs):
            self.append("trainer_metrics.jsonl", {"global_step": state.global_step, **(logs or {})})

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
            missing = [name for name in required if not (checkpoint / name).is_file()]
            write_json(checkpoint / "recipeqa_checkpoint_audit.json", {
                "global_step": state.global_step, "epoch": state.epoch, "required_state": list(required),
                "missing": missing, "complete_single_process_resume_state": not missing,
                "attempt_id": attempt.attempt_id if attempt else None,
                "run_signature_sha256": attempt.signature_sha256 if attempt else None,
                "state_sha256": {name: sha256_file(checkpoint / name) for name in required if name not in missing},
                "adapter_identity": model_weight_identity(checkpoint, adapter=True),
            })
            if missing:
                raise RuntimeError(f"Checkpoint missing full resume state: {missing}")

    return RunAuditCallback()
