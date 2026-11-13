# Paired source-step matching / ordering diagnostic

Status: **runnable draft; no model call, SSH, GPU execution or new freeze was made while preparing these files.** Root must independently review and freeze a separate30min / approximatelyCNY0.8 stage before running it. Current frozen projection sources are untouched.

The current lean plan supersedes the earlier single-model/cached-A cost example in `DESIGN_ZH.md`. It fixes **SFT44 raw_sft/checkpoint-64 and Exact-KL44/checkpoint-64**, first32 existing development recipes, and four new conditions per model:

| Condition | Calls per model | Input / response |
|---|---:|---|
| A |32| Reconstructed original recipe/four images; answer only, greedy32 |
| B |32| Identical recipe bytes/four-image tensors; five-line step assignment + order, greedy64 |
| C |32| Correct authoritative step-number table only; answer only, greedy32 |
| Cpi |32| Independently prompted cyclic table `(1,2,3,0)`; scored against that table, greedy32 |

Total256 raw observations:128 image inputs and128 text-table inputs. New contemporaneous A is primary; historical A is a repeat/identity control. First2 fixed recipes × fourconditions × two models produce16 canonical pilot observations, **included in256 and never rerun**. Gate estimates the other240, both remaining full model reloads,1.5× safety and120seconds I/O allowance. Gate failure stops the complete batch; no condition, question or model is silently dropped.

`decomposition.py` converts backend source IDs to actual visible1-based ordinals. Out-of-range, missing and unclear values count wrong in fixed pair denominators; equal values make a wrong pair and cannot create a complete order. Final `Order` validity is scored independently from the four-number format. The output preserves invalids, ties, ambiguity and token-limit reaches. Summary requires all256 unique observations without technical errors. It reports fixed32/128/192 denominators and the predeclared29unclipped/3clipped recipe strata.

`run_decomposition.py` checks frozen source/input/image SHA, same original cap and prompt hash, currentA/B literal recipe UTF8 bytes and actual CPU `pixel_values`/`image_grid_thw` equality. Actual inference rechecks pixels and grid. Historical full pixel bytes cannot be retroactively established; this limitation is saved explicitly. All32 templates must fit existing4096input/1536recipe/65536–262144pixel settings and a legitimate B completion includingEOS must fit64tokens. It never searches a fresh clipping policy, retries generation or changes a model/reward.

`raw_predictions.jsonl` records the canonical response and `prewrite_seconds`; `timing_events.jsonl` separately records each matching key and complete measured post-raw/state-write interval used by the gate. The gate can therefore be reconstructed from persisted timings rather than an unrecorded in-memory mutation. Runtime/token counts and source-match/order/consistency cross-tabs are saved per recipe.

The runner requires exactly the frozen **25CPU tests**, a structured actual `tests_run=25/skipped=0/failures=0/errors=0/passed=true` receipt and a clean terminal unittest result before processor/model work. Zero-collected or incomplete success is rejected even with exitcode0. The outer `launch_stage.py` uses GNU `timeout --signal=TERM --kill-after=10s` around the entire runner, including CPU tests and pixel preflight. Its interval is the remaining **original first-preflight/absolute clock**, never a new1800seconds. A real in-process SIGALRM independently handles cooperative paths. Up to10seconds after the deadline is cleanup only; timeout owns the process group so a blocked nativeCUDA call cannot continue indefinitely. Fresh output directory and exclusive legacy/new pipeline locks prevent resume or concurrentGPU workers.

CPU checks, safe locally:

```sh
python3 research/grounding_decomposition_20261106/test_decomposition.py
```

The suite uses stdlib only and the existing known-development metadata. It tests noncontiguous/zero source IDs, label round trips, immutable cap reconstruction, output-budget refusal, ties/range/duplicate/malformed/missing fields, absolute-vs-relative matching, self-consistency, cyclic oracle answers, remaining240 timing, the original deadline, draft refusal and complete denominators. It does **not** pretend fake-tokenizer fixtures establish actualQwen processing or GPU readiness; launch-time real CPU processor checks establish that separately. Missing actual metadata is a failure, never a skip.

Draft building after deploying the reviewed files to the existing data disk, with actual images already present:

```sh
/root/autodl-tmp/rl_recipeqa/.venv/bin/python /root/autodl-tmp/rl_recipeqa/research/grounding_decomposition_20261106/build_draft_policy.py --data-root /root/autodl-tmp/rl_recipeqa --output /root/autodl-tmp/rl_recipeqa/research/grounding_decomposition_20261106/DRAFT_PROTOCOL.json
```

This reads and hashes existing data only; it creates an **unexecutable** policy with null clock and null budget authorization. It may accept an independently audited exact128image SHA map via `--image-sha256-json` when local image bytes are absent; the runner still hashes real remote image bytes before inference. Historical model identities are copied from original config and then actually rehashed remotely, not claimed as locally re-read tensors.

Root later reviews, announces and freezes: status `locked_before_paired_grounding_decomposition`, budget authorization, first-preflightUTC and absolute≤first+1800seconds. Retain a GNUtimeout version/launch receipt in deployment evidence. Start the reviewed `launch_stage.py --policy <absolute frozen policy path>` from the same data-disk environment with existing `env.sh`; no password belongs in any file/command/config/log. Do not invoke `run` directly in formal use because that bypasses the native watchdog. No retries after launch, clock reset after technical repair, or partial result advertised as an overall comparison.

Interpretation limits:32known-development recipes, one optimization flow and pending human semantic uniqueness audit. C/Cpi remove images/long context, use oracle information and change task/output demands: highC minus lowB is **not** a causal estimate of a visual module's contribution. B statements are observable outputs, not recovered hidden reasoning. A B-prompt gain is a prompt/computation result, not RL or training gain. No test/confirmation access, SOTA claim, optimizer, new reward or weight update is part of this diagnostic.
