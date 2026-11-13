# Exploratory reward runner, 2026-11-05

These are **new post-hoc experiments**. They do not replace or modify the frozen
seed42/seed43 results. Development-set orchestration and test authorization live
in the separately maintained exploratory protocol.

## Entry points

```sh
python exploratory/reward_search_20261105/train_reward_search.py --help
python exploratory/reward_search_20261105/train_reward_search.py --cpu-smoke
python exploratory/reward_search_20261105/train_reward_search.py --reward-table
python exploratory/reward_search_20261105/train_reward_search.py --check-api
```

The first three need only Python's standard library and frozen local metric
helpers. API inspection imports installed TRL but does not load model weights.
The real GPU pilot remains necessary. All training arguments must name data-disk
paths. On AutoDL, HF_HOME, HF_HUB_CACHE, HF_DATASETS_CACHE, TMPDIR and TORCH_HOME
must be set to data-disk locations before importing training dependencies.

## Rewards

Let `e` be strict full-order correctness, `q` the six-pair correctness fraction,
and `d` the inversion count of a valid four-label permutation.

| Name | Reward |
| --- | --- |
| exact | e |
| pair | q |
| exp_inv | exp(-d) for valid answers; zero for invalid answers |
| success_gate_pair | Exact for the entire group if any candidate is exactly correct; otherwise Pair |

No formatting bonus, length reward, extra parser tolerance, reward model, or
constant-group resampling is used. Ground-truth source identities are only used
inside the existing verifier, never injected into the model prompt.

`success_gate_pair` is group dependent and needs complete groups. The logger
rejects mixed qids, recipes, presentations, targets, visits and input hashes.
It records the gate rule, reward vector returned to TRL, every alternate reward
vector, strict component scores and offline reference advantages. All actual
visits and complete candidate counts are checked at the end of a run.

Under group normalization, positive reward rescaling almost cancels (the `1e-4`
epsilon prevents exact invariance). A monotonic transform also cannot change the
ideal normalized advantages when a group has only two distinct score levels.
The exponential reward does not automatically cure duplicate candidates or
constant groups. Diversity and score-level diagnostics must be reported.

## Common start and comparisons

`--initial-adapter` is mandatory and loaded with `is_trainable=True`. Both
`--method grpo` and `--method sft` start a fresh optimizer from its same weights.
`--resume` instead requires complete state from this same run and an identical
run signature. It must not be used to load the previous SFT run's optimizer.

Defaults preserve the earlier protocol: native four images, 1536 recipe tokens,
4096 expanded input tokens, 32 output tokens, language-only LoRA, frozen vision,
BF16 SDPA, sequential stream, one GPU, GRPO loss, beta zero, one policy iteration.
G4/G8 and sampling temperature are explicit CLI parameters. `--scale-rewards
none` is a separate algorithm ablation, not a hidden part of a reward change.

For GRPO, generation batch size must equal gradient accumulation and be divisible
by G. With batch size 8, G4 consumes two independent prompt visits per update;
G8 consumes one. SFT consumes gradient-accumulation-size visits per update.
Thus equal optimizer steps are not equal prompt exposure or equal compute.

The manifest must contain enough visits for the requested budget. The runner
will not silently cycle it. All scheduled checkpoints are retained, contain
optimizer/scheduler/RNG state, and use the original checkpoint auditor. There is
no automatic checkpoint selection or test access in this runner.

The run signature binds the actual initial adapter and base-weight bytes,
manifest bytes, installed versions, new reward source, runner, frozen helpers,
reward formula, seed, temperature, G, scaling, input and optimization settings.
Runtime failure produces an explicit failed attempt record.
