# Producer / independent verifier interface draft

Local-only, not frozen. Parent `PLAN.md` controls. Independent verifier owns `verify_refresh_run.py` / `test_verify_refresh_run.py`; producer owns `refresh_contract.py`, `refresh_train.py`, `refresh_pipeline.py`, `build_draft_policy.py`, `launch_stage.py`, `run_cpu_preflight.py`, `test_refresh_contract.py`, `test_refresh_pipeline.py`, `TRAINER_LINEAGE.json`. All source closure paths will be published once final; do not monkeypatch/import the old trainer to execute the new experiment.

## Child policy

```
status: locked_before_reference_refresh_training
mode: smoke | path_smoke | formal
arm: raw_sft | fixed_beta | fixed_half_beta | refresh_beta
flow: flow44 | flow45
cli:
  method: raw_sft (arm raw_sft) | exact_kl (other three)
  beta: 0 | BETA | BETA/2 | BETA
  seed: 44 | 45, matching flow
  manifest: prepared_v1/{smoke|formal}_{flow}.jsonl
  unique_count: 4 (smoke/path_smoke) | 256 (formal)
  passes: 1 (smoke) | 2 (path_smoke/formal)
  limit: 4 (smoke) | 8 (path_smoke) | 512 (formal)
  max_steps: 1 | 2 | 128
  save_steps: 1 | 1 | 64
  refresh_boundary_step: null (all ordinary smoke/nonrefresh arms) | 1 (refresh path_smoke) | 64 (refresh formal)
  reference_cache: generated q0 cache with 4 or256 unique rows, never512 fake scores
  final_bank_count: unique_count, on every run including technical smoke
  [same fixed input/model/initial adapter/LR5e-6/logp_atol.01/wall fields]
initial_identity: exact frozen SFT768/base identity
sources_sha256: full reviewed new closure including independent verifier/tests and old helper dependencies
artifacts_sha256: exact old source/cache/spec/audit/manifests/prior-consumption and locked documents
reward: unique_complete_order_exact_0_1
selection: fixed128_primary_step64_boundary_no_dev_selection
first_preflight_started_utc == batch_clock_started_utc
absolute_deadline_utc: first+72000 seconds
q0_reference_lineage: [path]
generated_reference_receipt: [path]
pool_spec: original7558/prefix256/254recipes/prior actual6144 and signed initial step768 proof
```

Eight ordinary smokes are `smoke_{flow}_{arm}` with4unique/4visits/1update. Two path smokes are `path_smoke_{flow}_refresh_beta` with4unique/8visits/2updates and actual refresh after1. All ten must independently pass before the all-eight formal resource gate. Formal names `{flow}_{arm}`; all eight complete. Each flow's ordered256 records are repeated identically in pass2; smoke uses the same flow's first4. A question identity remains `(qid,presentation_id)`; a visit is `(pass_index,index_within_pass)`. Original record bytes, visit_id, recipe, permutation and input remain unchanged.

## q0 materialization and source binding

Source is the completed real1024 initial-reference cache. Each new q0 cache has its config/completion/lineage sidecars, preserving all original numeric/candidate/EOS/audit fields and original literal-check schedule. Only per-row `index` and `reference_spec_sha256` may change; spec only `manifest_sha256`, `selected_records_sha256`, `pixel_sha256` may change. Original source fields/hashes/row source_index retained in lineage; target row count4/256. The old four scoring helpers and exact versions remain the scoring closure; the full new trainer/coordinator/verifier closure is separately signed. Never relabel old model inference as new inference.

`generated_reference_receipt.json` seals all derived q0 artifacts and its own SHA is then registered by parent immutable checks, before any smoke. All child policies are fixed beforehand with deterministic cache paths; actual cache SHA is sealed by parent and each run_signature. No policy rewriting after score generation.

## Run signature / actual visits

`run_signature.json={sha256:hash_json(payload),payload:{...}}`, with immutable child/parent policy SHA, full sources and versions, exact q0 cache/spec/lineage SHA, unique manifest/selected records hash, two-pass visit sequence hash, first/deadline, arm/method/beta, unique_count/passes/actual_updates/planned refresh, C5/accum4/LR/clip and initial actual identities/fingerprints. Its `reference_policy_sha256` always means q0; later reference events are linked to this stable run signature, never change it in place.

Each `actual_prompt_visits.jsonl` row retains old token_logps, q/logq/logZ, exact_rewards, coefficients and all input/autograd counters, plus:

- `visit_index` global0..limit-1; `pass_index=visit_index//unique_count`; `index_within_pass=visit_index%unique_count`;
- `reference_row_index=index_within_pass`;
- `q0_reference_policy_sha256` immutable;
- `active_reference_policy_sha256` and `reference_generation` 0 or1;
- `reference_logq` is the active actual row used by this visit;
- `run_signature_sha256`, `consumption_point=all_selected_candidate_backward_calls_returned`.

Nonrefresh arms generation0 for both passes. Refresh ordinary smoke remains generation0; refresh path/formal switches exactly after the boundary and only for pass2. Native online J is relative to that active ref and native β; never interpret different round objectives as a single static endpoint objective.

Each step row retains exact pre/postclip, Adam step, actual parameter-update norm, cumulative forward/backward/reference counters, plus arm and active generation; optimizer state step content must increase continuously across64→65→128 (or1→2 technical path). Checkpoints64/128 (or smoke1/path1/2) store original full optimizer/scheduler/RNG/audit identities, and audit declares q0 and active reference hashes separately.

## Real refresh event

Only refresh path/formal creates `refresh_at_step{1|64}/` after that real checkpoint exists. It contains:

- `continuity_before.pt` and `continuity_after.pt`: real optimizer state_dict, scheduler state_dict, parameter gradients, full named-module training flags, Python global and sampling RNG, NumPy RNG, torch CPU and all CUDA RNG, global step; CPU-safe load supported. Each component also has a deterministic logical-content fingerprint (tensor dtype/shape/raw bytes, not torch ZIP filename bytes).
- `reference_policy.jsonl` / `.config.json` / `.completion.json`: all4/256 **new actual current-model** raw24×5 token scores, EOS grammar, input audit, q/logq/logZ, candidate exact verification metadata, current checkpoint identity and boundary step. Config explicitly names the *scored current identity*, not merely q0. Version/scoring-source/input/pixel identities bind every row. Includes all rows, source order and first-row literal checks; no gold-dependent reweighting.
- `refresh_event.json`: stable run/child SHA; checkpoint audit/path/SHA/identity; boundary/count; before/after state file SHA and component fingerprints; adapter/visual/mode hashes before/after; fresh cache/spec/completion SHA; actual forward count and zero optimizer/backward/generation; start/end UTC and wall time.

The deterministic state equality check covers actual optimizer moments and step tensors, gradients, scheduler, module modes and RNG, not only file hashes/booleans. No manual RNG reset or optimizer reconstruction is a valid substitute. The frozen verifier independently loads these state payloads and reconstructs fingerprints. Before/after serialization files can differ as ZIP bytes while logical state is identical; both physical file SHAs still bind evidence. All eight full optimizer/gradient snapshot files are approximately210MB each (four formal and four technical-path files); a four-question path still contains the whole optimizer and gradient state. The storage gate includes all eight, approximately1.68GB before overhead.

The trainer continues the same in-memory optimizer/scheduler/model after the reference scoring; only active detached reference pointer/hash changes. Source/q0/current ref artifacts never overwritten. Fresh reference token rows are signed by the event; every second-pass visit must consume its corresponding row.

## Final fixed train bank

After final checkpoint128 (technical endpoints1/2) is saved, the same still-loaded final model scores unique256 (or technical4) once into `final_train_bank/scores.jsonl`, with config/completion/identity/no-update fingerprints. This is a fixed endpoint measurement, separate from online rows. Each record includes raw24×5 token_logps/candidates/EOS, input audit, stable run SHA, final checkpoint physical identity, initial q0 row hash/source index. All final qgold/KL/J_beta/J_half/gaps/realization are independently reconstructed against q0 later. No devcanonical task, no new greedy decoding or gold tie selection in this bank.

`train_outcome.json` includes full actual visits512, unique256, updates128, all checkpoint64/128, refresh event iff planned, final-bank256 receipt/hash, total current/gradient/ref/final-bank forward and backward counters separately. Shared old q0 creation charged historically once, derived caches zero new forwards; refresh and final-bank added actual forwards. No resume/automatic retry/test/confirmation.

## Stage and analysis interface

Outer stage binds all eight formal, eight smoke and two path-smoke child policies, unique common first/absolute clock, full source and artifact closure, expected CPU count and all study docs. GNUtimeout parent and every scientific child share PGID; no detached descendants. Persisted CPU/q0/ten-technical-smoke/all-eight resource authorizations precede formal. Native watchguard and all CPU work are included from first preflight, not formal start.

Primary result arrays have200 recipe rows×8 fixed arm columns, aligned by recipe. Each contrast is two-flow arithmetic mean of within-flow paired differences: refresh−fixed_half_beta and refresh−raw_sft. Bootstrap draws5000/seed20261107 use one shared sampled200-index array per draw across all eight columns and both contrasts; no400 independent units. Final-train unit is254recipe clusters/256questions, orbit16recipe clusters/384layouts. Initial strata are frozen using q0 and originalβ (193/14/49) for every arm and both endpoint J targets.

The fixed stage schedule has74 commands: original73 CPU/q0/technical/formal/evaluation commands and final `stage_global_verify`. After73 successful commands the producer writes `science_completion.json` (science_complete_pending_global_verification,commands_completed73,parentSHA,all8formal/all10technical,4672responses/2048bankrows). Command74 runs the independent `--stage-only` complete-stage integrity guard within the original20h/native watchdog. Its only live state is running_stage_global_verify with the first73 success records and last returncodeNone, actual verifierPID=activePID and matching nativePGID; a one-byte parent barrier persists this identity before same-PID exec. Success is hash-bound in final completion.json with74commands and terminal independently_verified status; no pending global audit can masquerade as complete. The verifier separates the mutable live state journal from permanent evidence receipts. Scientific effect reporting is performed after this guard, and technical independent smoke/run verification never selects methods by score.

## 已实现的精确补充（freeze前审阅）

- `logical_content`序列化：primitive为`{kind:type.__name__,value}`；dict为`{kind:dict,items:[[logical(key),logical(value)],...]}`，按key的sort_keys/ensure_ascii=False JSON字符串排序；tuple/list保留类型；tensor为`{kind:torch_tensor,dtype:str(dtype),shape:list,sha256:SHA256(detach.cpu.contiguous.reshape(-1).view(uint8).numpy.tobytes)}`；NumPy数组类似numpy_array，NumPy标量保留dtype及逻辑item。全组件`hash_json`采用原modeling的`json.dumps(sort_keys=True,ensure_ascii=False)`SHA，logical总hash对组件SHA字典再hash。
- `.pt` root恰七键：schema_version、global_step、optimizer、scheduler、gradients、model_modes、rng。`continuity_summary`存组件SHA、总SHA、392个实际optimizer step、scheduler_last_epoch/global_step。独立CPU加载重算，不要求torch.save ZIP文件字节相等。
- 每个step日志有实际`optimizer_step_values`392整数及`scheduler_last_epoch`，全部必须step+1；旧coverage原本已有constant LambdaLR，本轮只增加观测。
- 评分完成sidecar为`{scores|reference_policy}.jsonl.completion.json`，其中`elapsed_seconds==wall_seconds`覆盖评分前状态/后指纹；refresh_event也保留两个相同alias。`final_train_bank`无另一个completion.json。
- run_signature的`mode`和`refresh_mode`都为childmode，旧自然语言执行说明字段改名`execution_mode`防同名覆盖。
- authorization必须核CPU实际parent policy/source/originalclock/native，以及四q0/十technical分别完整role、flow、arm与对应child SHA，不能重复一个passed receipt冒充多flow。
