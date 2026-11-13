# RecipeQA four-endpoint full-display-orbit diagnostic (not executed)

- `PROTOCOL_ZH.md`: full research question, fixed design, limits, budget and evidence checklist.
- `ORBIT_PROTOCOL.draft.json`: unreviewed template, model identities from already verified step64 predictor configs, seeds44/45, all existing source SHA receipts.
- `ORBIT_PROTOCOL.json`: same draft currently; root may freeze this separately after review by adding a new absolute deadline and locked status. Predictor will refuse draft status.
- `orbit_diagnostic.py`: CPU manifest constructor and independent scorer; remote read-only coordinator invokes existing frozen predictor/coverage validator. No model implementation, SSH, training or optimizer code.
- `test_orbit_diagnostic.py`: nine CPU tests pass locally. `python3 -m unittest discover -s research/grounding_equivariance_20261106 -p 'test_orbit_diagnostic.py' -v`.

Fixed models: both flows' raw_sft and exact_kl step64, all same256visits/64updates. Do not call these previously dev-selected checkpoints: flow45 SFT was selectedstep48 in that separate analysis.

Before execution root must independently review, perform real-manifest CPU preflight, announce new2h/maximum3.2CNY batch, lock all reviewed SHA values and a new absolute UTC deadline. If the deadline starts at preflight, the absolute deadline must remain based on that preflight start, not reset at coordinator launch. Configure all active HF, Torch, Triton, CUDA, TorchInductor, XDG, Pip and temporary caches on `/root/autodl-tmp/rl_recipeqa`; existing env.sh does most of this but active cache values require explicit inspection.

Fresh destination only: `exploratory/grounding_equivariance_20261106/orbit_v1`. One new2h phase plus new absolute deadline, earlier wins. No restart, retries or resumption; incomplete orbits never become complete summaries. All four models pilot first; gate sees CPU-preprocessing-inclusive timestamped output intervals. Formal384 predictions/model,1536 total; pilot4/model,16 total. Outputs include raw/configs, logs/arrival timing/returncodes, coverage, independent scores, per-recipe metrics, SHA receipts, weight fingerprints, final summary.

No GPU or SSH execution has been performed by the drafting agent. Data/images/base/adapter weights are absent locally by design, so full inference and actual timing remain unmeasured.

终止审查补充：最后 coverage 后、最终 fingerprint 前后及标记 completed 前均检查剩余预算；SIGTERM/SIGHUP 转为异常，进入进程组清理与锁释放。SIGKILL 无法由 Python 捕获，不能声称这种强杀情形也保证无孤儿 GPU 进程；应由后台观察器再核验。

硬截止补充：SIGALRM 真实 wall timer 覆盖 CPU preflight、哈希及汇总，取新7200秒wall与冻结absolute较早值。截止后仅执行 owned child 的 TERM/KILL 清理（最多8秒）与日志线程收尾（最多2秒），不启动新研究任务；finally取消闹钟并恢复旧handler。
