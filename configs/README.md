# Candidate training protocol

`candidate_protocol.json` records the pre-pilot settings; it is **not a compatibility or performance result**. Every execution writes its actual CLI, package versions, Trainer configuration, model revision, data hash, input audit, and parameter inventory under its output directory. Root orchestration must freeze the protocol only after real GPU pilot and recovery checks.

## Data and model paths

On the AutoDL server, the model snapshot, input manifests, images, environment caches, temporary files, logs and checkpoints must all reside under `/root/autodl-tmp`. The scripts refuse another path when that data disk exists. Model loading is local-only; downloading is a separate controlled step.

Use the exact downloaded model snapshot path and the actual image-root from the data audit. Do not pass model ID strings in place of a local snapshot directory. All base/processor loading uses one revision and one image/text policy.

## Inputs and fixed ordering

The three branches consume `train_visits.jsonl` exactly once: 2,048 ordered visits already encode two passes over 1,024 recipes and the per-visit image display permutation. Do not set two Trainer epochs on this file.

`SFTConfig.shuffle_dataset=False` alone does not guarantee sequential training: Transformers 5.22.1 still defaults to a random sampler. The scripts also set `train_sampling_strategy='sequential'`. GRPO uses its official `RepeatSampler` with `shuffle_dataset=False` and repeats each input for its G completions. The generated group log checks that qid, presentation and semantic mapping agree within each group.

Four RGB images are processed natively. Every ordered recipe title is preserved. Body text is clipped with one equal per-step token cap, without selecting gold-matching steps. The actual processor-expanded input length is checked before training; neither SFT nor GRPO may silently truncate image tokens. The preprocessing audit stores titles retained, clipped step IDs, raw/used recipe tokens, full input tokens, image grids and prompt hash.

## Common W versus resume

`--initial-adapter PATH` loads the common W adapter as trainable and creates a new optimizer/scheduler. Omitting it creates the same seeded language-only LoRA initialization on Base. Every branch saves the initial adapter fingerprint; these fingerprints must match across branches.

`--resume CHECKPOINT` instead asks Trainer to restore a complete interrupted branch. The path must contain optimizer, scheduler, RNG and Trainer state. A weights-only adapter is insufficient. The branch still uses the same base and W arguments as its original launch. All optimizer checkpoints align with GRPO generation-block boundaries under the enforced single-process configuration.

Trainer's state restoration must be tested with an actual exit/restart. The existence of the files is necessary but does not prove the resumed sample sequence or optimizer state is correct. Audit global step, actual visit IDs, scheduler, and RNG against an uninterrupted small control run before production.

## Training smoke commands

Fill the placeholders with data-disk paths. These examples intentionally train only a small pilot, not the formal branches.

```bash
python scripts/train_sft.py \
  --model-path <local-snapshot> \
  --manifest <data-disk>/manifests/debug.jsonl \
  --image-root <actual-image-root> \
  --output-dir <data-disk>/runs/pilot_sft \
  --limit 8 --max-steps 2 --save-steps 1

python scripts/train_grpo.py \
  --model-path <local-snapshot> \
  --manifest <data-disk>/manifests/debug.jsonl \
  --image-root <actual-image-root> \
  --output-dir <data-disk>/runs/pilot_pair \
  --reward pair --limit 8 --max-steps 2 --save-steps 1
```

Repeat the RL pilot with `--reward exact` and another output directory. A constant-reward Exact pilot may produce no adapter change: the script records that explicitly and the effective-training gate remains unpassed. It does not manufacture a positive reward or silently resample such groups.

For formal runs, remove the pilot limit/step overrides, use `train_visits.jsonl`, and attach the same W if the format gate required one. The candidate whole-stream budgets are 256 SFT updates and 1,024 GRPO updates, not equal compute. Common checkpoint selection is external, using the same fixed development manifest and greedy decoder.

## Prediction commands

```bash
python scripts/predict.py \
  --model-path <local-snapshot> \
  --manifest <data-disk>/manifests/val.jsonl \
  --image-root <actual-image-root> \
  --output <data-disk>/runs/base_val/predictions.jsonl \
  --run-id base_val --limit 20
```

Add `--adapter-path <checkpoint-or-final-adapter>` for a trained model. For the W candidate bank, use its bank manifest plus `--num-samples 4 --temperature 1.0`; the same candidates can then be scored with Exact and Pair. For E4, use the process manifest and `--process --max-new-tokens 256`; the ordinary four-label parser is deliberately not used on process prose.

Prediction records are keyed by run/checkpoint/qid/presentation/sample/prompt-kind. New keys append; a technical failure retried successfully is replaced by an atomic upsert, leaving one canonical row per key. Technical attempts remain in a separate journal. A successful key is skipped when restarting; conflicting successful duplicates are rejected. Sampling seeds are derived per key, so skipping prior records does not change the remaining candidates. Preprocessing and generation failures do not count as invalid model responses and stop execution unless explicitly asked to continue.

## Per-run checks

- `label_mask_audit.json`: SFT supervision decodes to only the correct four labels; padding/prompt/image positions are masked.
- `trainable_parameters.json`: only the explicitly enumerated language-layer LoRA weights are trainable.
- `gradient_audit.jsonl`: finite gradients, nonzero trainable gradients when signal exists, and no frozen-weight gradients.
- `step_times.jsonl`: synchronized full optimizer-step time and peak allocated/reserved memory, including generation and accumulation.
- `reward_groups.jsonl`: exact/pair rewards on identical candidates, group coherence, zero-variance indicators, and offline sample-std advantages.
- `rollouts.jsonl`: raw candidate text, identities, target mapping, dual score, actual output/input token counts.
- `checkpoint-*/recipeqa_checkpoint_audit.json`: complete single-process recovery files written after Trainer saves.
- `train_outcome.json`: actual completion, adapter change, frozen visual hash, costs; not a substitute for dev/test performance.

Only the remote GPU measurements can validate memory fit, effective learning and total runtime. Current source checks and CPU reward tests are not counted as a passed training pilot.
