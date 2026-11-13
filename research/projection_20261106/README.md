# DRAFT: forward-KL projection of the existing finite-policy RL target

No SSH, GPU execution, training, added stage budget or queue was created here. The running display-orbit batch and all original sources remain unchanged.

Read `DESIGN_ZH.md` for the research question, objective, fairness conditions and attribution limits. `MATH_REVIEW.md` and the independent reference/JSON belong to the independent mathematics reviewer.

New files:

- `projection_math.py`: immutable qstar and detached β(qstar−qθ) ascent coefficients; exact0/1 only.
- `projection_train.py`: NEW prototype adapted from the frozen finite trainer, importing its verified24-action scorer and differentiable complete answer+EOS logp. It refuses any policy without a separate locked status.
- `test_projection_math.py`:12 local CPU tests pass, including finite differences in24 logits and nonlinear shared parameters, EOS and C, β, target freezing, objective identities, original hashes, JSON signatures and draft refusal before model access.
- `PROJECTION_smoke44.draft.json`: separately locked4-question/1-update/save1 acceptance mode, reusing the existing original reference4_smoke.
- `PROJECTION_formal44.draft.json`, `PROJECTION_formal45.draft.json`: original256-question/64-update/save16 formal flows and their original reference caches.

Templates already include existing manifest/reference/config/completion SHA, actual initial base/adapter identities and current source hashes. Their status refuses execution and `wall_seconds` is deliberately null. Root must freeze a new full-stage protocol and budget/coordinator first. A third flow is a design possibility, not an implemented or budgeted arm here.

Before GPU allocation, the prototype validates all CLI fields, source/artifact SHA, actual image bytes, initial disk weights, original cache spec/coverage/gold/EOS/logp, immutable qstar and JSON-native signature. It does not regenerate reference targets from the current model or sample candidate successes. CPU preflight success is not GPU backward/optimizer acceptance.

The new goal is β KL(qstar||qθ)/C; the old Exact+KL is equivalent to β KL(qθ||qstar)/C up to a constant. This is a known analytic RL target projected by supervised soft-target fitting, not raw sequence SFT and not a new RL algorithm. Initial directions are collinear with sample-dependent amplitude; low-reference-probability tasks still have weak target changes. Same visits do not mean same compute. Report all arms and failed runs, keep the fresh confirmation200 sealed, and claim no SOTA.

Latest outer-stage draft: `projection_pipeline.py`, `run_cpu_preflight.py`, `test_projection_pipeline.py`, `STAGE_PROTOCOL.draft.json`, and `stage_children_draft/*.json`. The current proposal is ONLY two new FWD arms on old flows, with frozen historical SFT/EK64 controls; the earlier third-flow/five-arm idea is superseded. Proposed new budget4h/6.4CNY is not execution authorization.29 total CPU tests are now discovered (12math+7pipeline+3Torch+7verifier); local26 pass/3explicit Torch skips, remote must29pass/0skip/noCUDA. Every child policy is fixed before any update; no post-smoke formal parameter edits. Original full-orbit batch is complete and remains separate.
