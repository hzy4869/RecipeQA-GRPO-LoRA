# Development-only reward screen

Post-hoc exploratory screen on the reused fixed development200; not heldout confirmation or an official benchmark SOTA

Status: **verified_partial_development_snapshot**. Completed selections: 0/8.
Common SFT start: Exact 67.00%, Pair 90.33%, valid 100.00% on 200 development recipes.

| Arm | Selected step | Dev Exact | Change vs common start | Active groups | Mean unique valid orders |
| --- | ---: | ---: | ---: | ---: | ---: |
| sft_cont | pending | pending | pending | — | — |
| pair_g4 | pending | pending | pending | — | — |
| pair_g8 | pending | pending | pending | — | — |
| exact_g8 | pending | pending | pending | — | — |
| exp_inv_g8 | pending | pending | pending | — | — |
| success_gate_pair_g8 | pending | pending | pending | — | — |
| pair_g8_none | pending | pending | pending | — | — |
| exp_inv_g8_none | pending | pending | pending | — | — |

All four planned checkpoint points are preserved in development_points.csv, including pending points in a partial snapshot.
A selected step0 means continued training did not beat the shared start under the locked Exact→Pair→earlier rule.

Finished command wall: 0.119 hours. This is not the rental bill.

## What can be concluded

These are descriptive, development-selected results. They can identify candidates for a separately locked follow-up and show whether exploration, reward variation and greedy accuracy move together. They do not establish test improvement, a causal mechanism, stable cross-seed superiority, or official SOTA.

## Limits

- Same shared SFT checkpoint and one screening seed44; not a full seed replication
- All arms share maximum256promptvisits; G8 uses twice G4 candidate generation, and SFT/RL compute differs
- G4 versus G8 also changes sampling temperature in this locked design
- group versus none changes gradient magnitude as part of the algorithm; fixed LR does not isolate objective shape
- Checkpoint and arm selection reuse development200, creating selection optimism
- Original test outcomes were known before exploration; no new test results are read here
- Human image/ambiguity and explanation audits remain unresolved
