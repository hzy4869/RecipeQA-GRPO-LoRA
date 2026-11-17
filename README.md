# RecipeQA-GRPO-LoRA

Comparing SFT, Exact-GRPO, and Pair-GRPO for visual recipe ordering with Qwen3-VL-2B-Instruct and language-only LoRA.

## Task

Given an ordered recipe and four shuffled images labeled A-D, predict the chronological image order. This is a derived free-ordering task and is not directly comparable with the original RecipeQA multiple-choice benchmark.

## Code structure

- `src/recipeqa/data.py`: data validation, image permutations, and deterministic training manifests.
- `src/recipeqa/modeling.py`: multimodal inputs, model loading, language-only LoRA, and runtime audits.
- `src/recipeqa/metrics.py`: strict answer parsing, exact-match and six-pair rewards, and recipe-clustered statistics.
- `src/recipeqa/process_metrics.py`: process-output evaluation.
- `scripts/`: training, prediction, validation, analysis, and plotting entry points.
- `ops/`: preparation, warmup, formal training, checkpoint selection, and evaluation pipelines.
- `configs/`: experiment settings and protocol locks.
- `tests/`: validation and regression tests.
- `research/` and `exploratory/`: supplementary experiment code and design notes.

## Training design

The primary branches start from the same warmup adapter W. SFT supervises only the answer completion. GRPO samples four answers per prompt and uses either an exact-match reward or the fraction of correct pairwise ordering relations. The primary GRPO configuration has beta=0 (no reference KL penalty). Vision weights remain frozen.

See `EXECUTION_GUIDE.md`, `RESEARCH_PLAN.md`, and `configs/README.md` for execution details. `PROJECT_STATUS.md` is a dated historical snapshot, not live runtime status. Server-specific paths must be adapted before running elsewhere.

## Experiments and results

Verified against saved server artifacts on **2026-11-18 (Asia/Shanghai)**. This documentation update did not launch training. Primary training and automatic evaluations are complete; the early snapshot in PROJECT_STATUS.md is historical.

### Setup and metrics

Qwen3-VL-2B-Instruct with language-only LoRA (r=16, alpha=32); base and vision weights frozen. The primary pool contains 1,024 questions, with two fixed passes encoded as 2,048 prompt visits. Warmup uses a separate 256-question pool. Checkpoints are selected on 200 development recipes by Exact, then Pair, then earlier step. Official train/val/test recipe splits remain disjoint.

```text
Base -> SFT warmup -> common W
                         |-- SFT continuation
                         |-- Exact-GRPO
                         `-- Pair-GRPO
```

Each branch resets its optimizer. Primary GRPO does NOT start from the fully trained SFT endpoint. SFT supervises the answer completion; GRPO samples four answers per prompt with Exact or Pair rewards and beta=0 (no reference KL penalty).

- **Exact**: entire permutation correct.
- **Pair**: fraction of the six pairwise ordering relations correct, not the fraction of images placed at their correct rank.
- **Valid**: output parses as a legal A-D permutation.
- Tables use **recipe-macro** averaging: average within each recipe, then across recipes. Invalid responses score zero and remain in denominators; technical failures are reported separately.
- E0 has 905 questions / 597 recipes. Every E0/E3 entry below has complete coverage and zero technical failures.
- Shared maximum prompt visits do not imply equal compute: SFT uses 2,048 supervised targets; each GRPO branch samples 8,192 completions. Development-selected endpoints can have different actual exposures.

### Completed primary experiments

| Experiment | Purpose and execution | Verified outcome |
| --- | --- | --- |
| Base and warmup | Establish format/order behavior and select common W | On development200, Base Valid=27%, Exact=12%; selected W checkpoint-16 Valid=99.5%, Exact=42.5%. These are development figures, not E0. |
| Training pilots and recovery | Check completion masks, trainable parameters, frozen vision, visits and checkpoint state | SFT/Exact/Pair same-W pilots each completed 20 updates. SFT and Pair underwent real restart checks; BF16 trajectories did not meet bitwise/current-tolerance equivalence. |
| Primary branches | Compare completion-only SFT with Exact/Pair GRPO from W | seed42 SFT completed 256 updates; both GRPO branches completed 1,024. LoRA changed and vision stayed frozen. |
| E0 | Evaluate exact ordering, pair relations and validity | All five models evaluated on 905 questions / 597 recipes. |
| E1 | Score identical candidates with both rewards and audit training dynamics | Fixed W bank100 x 4 completed; formal runtime/reward/gradient logs available. |
| E2 | Diagnose relations, ranks and error severity | Derived from the same E0 predictions, not another independent test set. |
| E3 | Measure behavior under all display permutations | Three seed42 endpoints, each 60 recipes x 24 layouts. |
| E4 | Request explicit image-to-step mapping, observations and final Order | Automatic scoring completed for 40 responses / 160 images per endpoint; human semantic review remains pending. |
| Optional seed43 | Repeat both GRPO branches from the same W and visit stream | Both completed 1,024 updates plus development selection and E0/E3. No second-seed SFT. |

### E0: primary held-out test

| Model | Checkpoint | Exact | Pair | Valid |
| --- | --- | --- | --- | --- |
| Base | Qwen3-VL-2B-Instruct | 12.91% | 25.92% | 30.71% |
| Exact-GRPO seed42 | checkpoint-768 | 54.10% | 87.84% | 99.58% |
| Pair-GRPO seed42 | checkpoint-1024 | 56.61% | 88.64% | 99.83% |
| SFT seed42 | checkpoint-256 | 69.57% | 91.56% | 100.00% |
| Common W | checkpoint-16 | 41.14% | 83.37% | 100.00% |

### Second training seed: E0 sensitivity check

Same W and visit stream; only the two GRPO branches were repeated. SFT has one training seed.

| Model | Checkpoint | Exact | Pair | Valid |
| --- | --- | --- | --- | --- |
| Exact-GRPO seed43 | checkpoint-1024 | 53.96% | 87.45% | 99.66% |
| Pair-GRPO seed43 | checkpoint-768 | 52.73% | 87.68% | 99.41% |

At seed42, Pair-GRPO exceeds Exact-GRPO by 2.51 percentage points in Exact, but trails SFT by 12.97 points; Exact-GRPO trails SFT by 15.47 points. Both GRPO branches improve over W. At seed43, Pair trails Exact by 1.22 points: the ranking reverses.

For Pair minus Exact in E0 Exact, the paired recipe-bootstrap 95% intervals are [+0.16, +4.69] points at seed42 and [-3.71, +1.31] at seed43 (2,000 resamples). These intervals measure test-recipe sampling uncertainty, not training-seed variability.

The optional repeat policy was fixed after seed42 test results were known. The original second seed was 2026; using 43 was an implementation deviation disclosed in metadata/seed43_protocol_deviation.json, not a performance-based seed selection. Two seeds are a sensitivity check, not proof of stable superiority.

### E1: identical-candidate reward comparison

On the same 100 development prompts with four W samples each:

| Group category | Groups / 100 |
| --- | --- |
| Both rewards vary | 49 |
| Only Pair varies | 41 |
| Both constant | 10 |

Exact distinguishes 49% of groups; Pair distinguishes 90%. More distinguishable feedback does not establish better generalization or a reward-density-only causal effect: Pair also changes preferences among incorrect permutations.

### E2: local error diagnostics (seed42)

Selected-image target ranks, not the beginning/middle/end of the full recipe:

| Model | First-rank correct | Last-rank correct | Rank2 before rank3 correct |
| --- | --- | --- | --- |
| eval_exact_seed42 | 79.67% | 79.22% | 80.95% |
| eval_pair_seed42 | 80.81% | 80.21% | 82.79% |
| eval_sft_seed42 | 85.91% | 85.49% | 87.99% |

### E3: all 24 display permutations

Fixed 60 recipes x 24 layouts per endpoint. Labels and targets are remapped together. All24 requires every layout correct. Modal consistency can also mean consistently wrong. There are 60 independent recipe units, not 1,440 independent samples.

| Model | Mean Exact | All24 Exact | Semantic modal consistency |
| --- | --- | --- | --- |
| Exact-GRPO seed42 | 60.14% | 25.00% | 80.07% |
| Pair-GRPO seed42 | 64.38% | 31.67% | 82.22% |
| SFT seed42 | 73.13% | 45.00% | 88.13% |
| Exact-GRPO seed43 | 63.19% | 26.67% | 79.44% |
| Pair-GRPO seed43 | 58.19% | 20.00% | 79.58% |

### E4: explicit-process output, automatic scoring only

Different process prompt and a 256-token output budget; these models were trained for short answers. Each endpoint has 40 responses / 160 images. Five W development calibration examples are separate. The ordinary short-answer parser is not applied to process prose.

| Model | Full format valid | Order parsable | Process Order Exact | Source-step match mean |
| --- | --- | --- | --- | --- |
| eval_exact_seed42 | 10.00% | 25.00% | 7.50% | 41.88% |
| eval_pair_seed42 | 5.00% | 22.50% | 15.00% | 44.38% |
| eval_sft_seed42 | 0.00% | 2.50% | 0.00% | 26.88% |

Process-format compliance is poor. These results cannot be substituted for E0 short-answer accuracy or interpreted as hidden reasoning quality. Two independent human reviews and consensus are still **pending**; factuality, discriminative visual evidence, support and human joint success are not filled with invented labels or zeros.

### Input sanity control and training cost

For W on the same 20 development recipes: real images Exact=7/20, blank images=1/20, both Valid=20/20. This small out-of-distribution blank-image control alone does not prove correct visual grounding.

The following is full training-invocation wall time, including preprocessing/finalization, excluding the full development search, evaluation, downloads, other failed attempts and idle rental time. It is not total project cost or equal-compute evidence.

| Branch | Completed updates | Invocation minutes |
| --- | --- | --- |
| sft_seed42 | 256 | 21.25 |
| exact_seed42 | 1024 | 108.71 |
| pair_seed42 | 1024 | 108.37 |
| exact_seed43 | 1024 | 106.90 |
| pair_seed43 | 1024 | 113.32 |

### Supplementary exploration already executed

These studies followed known primary outcomes and reused development data. Completion is not new held-out confirmation; conditional 24-answer policies are different from primary free-token GRPO. Artifact paths below are server-relative, not bundled result links.

| Study | Purpose | Verified state / result scope | Artifact directory |
| --- | --- | --- | --- |
| Reward/sampling screen | Compare group size, rewards and standardization | Partial historical snapshot; no complete eight-arm conclusion. An old running state is not evidence of a live process. | exploratory/reward_search_20261105/ |
| Larger-pool SFT | Check increased supervised-data coverage | Full-data SFT completed; selected step768: development Exact=71.50%, Pair=90.83%, Valid=100%. Not test performance or the primary four-pass control. | exploratory/principled_ordering_20261105/full_sft/ |
| Finite-answer policy trials | Compare raw SFT, Exact, Exact+KL and RLOO+KL over 24 legal answers | finite_trials_v2 completed; flow44 multi-arm exploration and flow45 conditional check of selected Exact+KL. | exploratory/principled_ordering_20261105/finite_trials_v2/ |
| Gradient-noise diagnostic | Examine finite-policy gradient statistics | Read-only 32-row diagnostic complete; no optimizer or weight updates. | exploratory/principled_ordering_20261105/gradient_noise_v1/ |
| Development permutation orbits | Compare semantic stability of four exploratory endpoints | Complete; 16 recipes x 24 layouts per endpoint, separate from primary E3. | exploratory/grounding_equivariance_20261106/orbit_v1/ |
| Projection-update control | Examine projected updates in finite-policy training | stage_v2 completed two fixed64-step endpoints; development only. | exploratory/projection_20261106/stage_v2/ |
| Grounding/order decomposition | Compare answer-only, process and oracle mapping-table conditions | Paired32 development decomposition completed. Oracle tables change available information, not deployment image accuracy. | exploratory/grounding_decomposition_20261106/stage_v2/ |
| Coverage extension | Compare raw SFT and Exact+KL on two visit flows | Four fixed256-step endpoints complete; step64 diagnostic only; development/orbit evaluation. | exploratory/coverage_20261106/stage_v1/ |
| Endpoint candidate bank | Score finite-answer policies without training | Complete: 9 formal tasks, 2,024 formal rows; zero updates or generation calls. Candidate likelihood forwards are not free sampling. | exploratory/endpoint_bank_20261106/stage_v1/ |
| KL reference refresh | Compare raw SFT, fixed beta, half beta and refreshed reference | stage_v4 completed eight endpoints across two flows with independent verification. Earlier versions include failed/repaired attempts. | exploratory/reference_refresh_20261107/stage_v4/ |

flow44/flow45 share an existing SFT initialization and previously seen training pool; they are not two independent full-training replications. Their development-selected or diagnostic numbers must not replace primary held-out results.

### Conclusions and limitations

1. At the tested common-W initialization, budget and selection protocol, seed42 SFT outperforms both GRPO variants. Both GRPO variants improve over W. This does not test full-SFT-then-GRPO or establish universal SFT superiority.
2. Pair increases same-candidate reward discrimination but does not produce stable cross-seed whole-order gains. The mechanism is not isolated by this comparison.
3. GRPO has two training seeds and SFT only one; no cross-dataset generalization is established. Recipe bootstrap does not measure training-seed variance.
4. Split isolation does not rule out pretraining contamination. Human ambiguity/visual-evidence audits remain incomplete. Process output does not establish faithful hidden reasoning.
5. Post-hoc development exploration is disclosed separately. Reusing development data does not establish fresh test improvement, original RecipeQA benchmark SOTA, or a new algorithm.

### Result provenance

Paths are relative to the server project root. Full result artifacts are excluded from Git; cloning this repository does not provide them. This README summarizes the saved evidence.

| Evidence | Server artifact |
| --- | --- |
| E0 seed42 / seed43 | results/e0_main/summary.json / results/seed43/e0/summary.json |
| E1 reward bank / training audits | results/e1_w_bank/summary.json / results/e1_formal_training_evidence.json |
| E2 | Relation/rank/error fields in E0 summaries |
| E3 seed42 / seed43 | results/e3_main/summary.json / results/seed43/e3/summary.json |
| E4 automatic/human status | results/e4_main/summary.json |
| Input control | results/w_blank_control/summary.json |
| Selection / repetition disclosure | metadata/formal_training_pipeline.json, metadata/seed43_pipeline.json, metadata/seed43_protocol_deviation.json |
| Runtime / pilots / recovery | runs/<branch>/train_outcome.json, results/same_w_pilot_summary.json, research/runtime/resume_precision_review.md |

SHA256 of the four verified E0/E3 summaries:

| File | SHA256 |
| --- | --- |
| results/e0_main/summary.json | cb286580a399ed0c74b1e16a10da45d72ad84b1d890f4e92cb0dce190a0823f4 |
| results/seed43/e0/summary.json | 5fc0657d5053753da6deacb430d12cdbd3871f6e3c2d9602ed146cd5e5ca5138 |
| results/e3_main/summary.json | 802406bbfb3cb88c9121438335d0bbc8421eabcfdaf0fbd77aa27fc26ebed62a |
| results/seed43/e3/summary.json | 983801e92fcbfc20cf579956cc4d210490fde6a6fcaee4825eaa9386fd2cf0fc |

## Artifacts and access

Datasets, model weights, checkpoints, caches, prediction records, and generated runtime outputs are excluded. Obtain the underlying data and model separately under their respective terms. This source snapshot does not contain the full experiment artifacts or original Git history. Existing protocol commit identifiers refer to the earlier source lineage.
