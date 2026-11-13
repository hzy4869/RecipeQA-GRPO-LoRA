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

## Artifacts and access

Datasets, model weights, checkpoints, caches, prediction records, and generated runtime outputs are excluded. Obtain the underlying data and model separately under their respective terms. This source snapshot does not contain the full experiment artifacts or original Git history. Existing protocol commit identifiers refer to the earlier source lineage.
