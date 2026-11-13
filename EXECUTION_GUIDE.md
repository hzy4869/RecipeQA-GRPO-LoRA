# RecipeQA 项目执行与复现指南

供三位组员查运行、核结果和接手审核。此文不改变已冻结协议，也不代表尚未运行的实验已完成。研究问题、统计定义和验收标准以 [RESEARCH_PLAN.md](RESEARCH_PLAN.md) 为准；动态进度见 [PROJECT_STATUS.md](PROJECT_STATUS.md)，并核对远端进程与产物。

## 1 当前实际起点与三人分工

2026年11月5日（香港）的执行快照：主SFT已完成256更新、2048次访问并通过工件验收；四个checkpoint、256次非零梯度与视觉权重前后SHA一致均已核验，含前处理及最终保存的wall time为1274.93秒。四个开发点Exact依次为56%、65.5%、67%、67%，按Pair平局处理选出step256；因最后两点Exact没有改善，四passes检查未触发。统一4线程编排父PID11447已进入Exact正式训练；两个正式RL分支尚未完成，不能把开发成绩称为主test成绩。此前72线程前处理调用在正式零更新时停止并保留记录。之后以当前 `metadata/formal_training_pipeline.json`、真实PID及验收产物为准。

- **A：运行与结果核验。** 记录真实完成、覆盖、恢复和实例计费；由代理顺序运行单卡GPU任务，避免另起任务争用显存。
- **B/C：独立人工审核。** 各自审核输入100题，后续各自审核E4的120个响应；先保留独立意见，再讨论分歧。当前这些人工任务为 **pending**，不能由代理代填。
- **三人共同：** 理解实测发现、核对图表与论文表述、录制演示并最终提交。训练未胜过SFT时如实分析，不换题、不筛掉失败测试题。

已固定：Qwen3-VL-2B-Instruct；同W的SFT-cont、Exact-GRPO、Pair-GRPO；正式首seed为42。W是 `runs/warmup_v1/checkpoint-16`，adapter文件SHA256为 `5b8768ecc20cfc031ccf21ed8509e66c91170d384c9f747142dec53dcb1ab31a`。选择依据为训练前固定的200题开发集，不是正式test。

## 2 连接、环境与文件位置

在本机进入远端；复用socket只在当前SSH master仍有效时可用，不包含或保存密码：

```bash
ssh -S /Users/rainchow/Documents/ChatGPT/RL_for_MLLM/.state/ssh/remote.sock \
  -o BatchMode=yes -p 31755 root@connect.westd.seetacloud.com
```

进入远端后，每个新shell先运行：

```bash
export TASK_ROOT=/root/autodl-tmp/rl_recipeqa
source "$TASK_ROOT/env.sh"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
cd "$TASK_ROOT"
export TASK_PY="$TASK_ROOT/.venv/bin/python"
export TASK_MODEL="$TASK_ROOT/models/Qwen3-VL-2B-Instruct"
export TASK_W="$TASK_ROOT/runs/warmup_v1/checkpoint-16"
export TASK_REV=89644892e4d85e24eaac8bacfd4f463576704203
```

`/root/autodl-tmp`是50GB数据盘；`/tmp`属于系统盘，不能放项目数据、模型、缓存、环境或输出。`env.sh`已把临时文件、HF、pip、Torch、Triton、CUDA及字节码缓存指向数据盘。不要在运行期间升级依赖或重建环境。实际锁定版本见 `configs/protocol_lock.json`；本次核验为Python3.12.3、Torch2.8.0+cu128、Transformers5.22.1、TRL1.17.1、PEFT0.21.2、Accelerate1.15.0、Datasets5.1.0。

**四个线程export必须在每次 `source env.sh` 后执行，正式三分支、开发评测及恢复保持相同设置。** 8个固定输入的CPU探针按4→72→4线程比较，全部encoded张量逐字节一致；本次72线程预处理耗时为4线程的13.9–25.3倍，说明此前发生CPU线程过度竞争。原科学源码、数据、W与协议锁均不变，只在首个正式更新前统一修订运行环境，补记于 `metadata/runtime_threads.json`；Torch interop线程保持默认。探针一致性不等于完整GPU训练逐位一致。旧调用记录索引为 `metadata/cpu72_preprocessing_aborted_20261104T164230Z/interruption_observation.json`，原成本不删除。

| 数据盘内相对路径 | 用途 |
|---|---|
| `configs/protocol_lock.json` | 已冻结源码、数据、模型/W内容哈希、版本与政策；不可覆盖 |
| `data/raw/`、`data/images/` | 官方JSON/ZIP、已解压照片；8871张所需图均已实际RGB解码 |
| `data/manifests/` | 固定题池、展示与标签；`train_visits.jsonl`已含两passes共2048访问，只消费一遍 |
| `data/review_views/` | 仅供人工审核／E4汇总的模型可见正文派生视图；不替换冻结预测manifest |
| `data/controls/` | blank20、解析期望与两布局手算控制；解析控制不是模型成绩 |
| `runs/sft_seed42/` | 主SFT：256更新，checkpoint64/128/192/256 |
| `runs/exact_seed42/`、`runs/pair_seed42/` | 主RL：各1024更新，checkpoint256/512/768/1024 |
| `runs/dev/` | 各候选checkpoint的200题开发预测与覆盖检查 |
| `metadata/` | 编排JSON、PID、stdout日志、W选择、预算政策 |
| `results/`、`review/` | 自动分析/图表、人工审核材料 |

`runs/sft_4pass_seed42/`仅在预先规定的开发触发条件满足时运行：从同W独立训练四passes作为预算检查，保留主两passes结果；不能称其与主RL等曝光或等计算。触发与选点见主SFT的 `sufficiency_trigger.json` 和协议锁。

## 3 看运行，不重复启动

```bash
cat metadata/formal_training_pipeline.json
TASK_PID=$(cat metadata/formal_training_pipeline.pid)
ps -p "$TASK_PID" -o pid,ppid,stat,etime,args
tail -n 20 metadata/sft_seed42.log
tail -n 20 metadata/formal_training_pipeline_cpu4.log
```

新编排主stdout为 `metadata/formal_training_pipeline_cpu4.log`；各脚本stage的日志仍为 `metadata/<stage>.log`，按JSON当前 `phase` 查找。主stdout为空不证明GPU未执行或任务已失败，应联查stage日志和run工件。编排进程等待GPU子进程时处于S状态正常；PID缺失或Z/X才表示原进程终止。JSON写着running但PID已经终止时，不能声称任务仍在运行，也不能仅因一次查询超时重启。

每个run应保留 `resolved_config.json`、`run_signature.json`、`input_audit.jsonl`、`gradient_audit.jsonl`、`step_times.jsonl`、`trainer_metrics.jsonl`、完整checkpoint和 `train_outcome.json`。SFT另有真实消费的 `actual_prompt_visits.jsonl`；RL另有 `reward_groups.jsonl`、`rollouts.jsonl`。`attempts/`保留所有调用历史及被恢复取代的工件，不能与根目录当前有效轨迹混合统计。

正式编排按顺序训练、验收并用开发集选点；最终选择写各run的 `best_checkpoint.json`。**不要默认最后checkpoint最优。** 选点顺序为开发recipe-macro Exact、Pair、较早checkpoint。GPU空闲不等于允许改锁定源码；不要重跑下载、解压、warm-up或整条正式编排。

## 4 中断后的严格恢复

先确认原编排与训练子进程都已终止，保存失败日志，找到有 `recipeqa_checkpoint_audit.json` 的完整checkpoint。恢复必须保持同源码、环境版本、manifest、W、seed、reward及全部原参数，只增加 `--resume`。完整checkpoint应有adapter、`trainer_state.json`、`optimizer.pt`、`scheduler.pt`、`rng_state.pth`及相应内容哈希；`adapter_final`不是完整恢复入口。

下面是**将来确有checkpoint64时**的主SFT恢复配方，不是现在启动的指令：

```bash
"$TASK_PY" scripts/train_sft.py \
  --model-path "$TASK_MODEL" --revision "$TASK_REV" \
  --image-root "$TASK_ROOT/data/images" \
  --manifest "$TASK_ROOT/data/manifests/train_visits.jsonl" \
  --output-dir "$TASK_ROOT/runs/sft_seed42" --initial-adapter "$TASK_W" \
  --max-steps 256 --save-steps 64 --seed 42 \
  --resume "$TASK_ROOT/runs/sft_seed42/checkpoint-64" \
  >> "$TASK_ROOT/metadata/sft_seed42_resume.log" 2>&1
```

RL用原 `train_grpo.py` 命令，保留原 `--reward exact` 或 `pair`、1024步和256保存间隔，再加其完整checkpoint。原启动参数可从编排JSON的 `commands[].arguments` 和run配置核对。不要把恢复改成 `--initial-adapter <中断模型>`，那会重建optimizer。不要改 `--max-steps`绕过签名保护。新run目录不可预先写stdout日志；stdout放 `metadata/`。

恢复工具会保留历史并将根日志裁到checkpoint有效轨迹，不把回放算成额外正式访问。已有恢复实测核对了状态、样本、RNG和LR，但BF16跨进程数值轨迹并非逐位一致；见 `research/runtime/resume_precision_review.md`，报告须保留此限制。

正式编排脚本没有自动跳过已完成stage的恢复入口；**不能重启整条编排来接着跑**。由代理按原阶段清单补回失败stage及剩余阶段，每段完成后再验收：

```bash
"$TASK_PY" scripts/validate_run.py training \
  --run-dir "$TASK_ROOT/runs/sft_seed42" --min-steps 256 \
  --output "$TASK_ROOT/runs/sft_seed42/artifact_validation.json"
```

RL对应 `--min-steps 1024`。训练工件验收不等于模型效果或完整恢复数值等价证明。

## 5 E0/E3/E4预测：训练和选点完成后逐项执行

统一评测入口已实现为 `ops/evaluation_pipeline.py`。它在三分支及开发选点全部验收前返回 `not_ready`，不会等待或启动GPU。先作只读检查；只有返回ready且没有存活的训练、交接或评测进程时，才首次启动一次。已存在评测记录须先查失败原因，再显式加 `--resume`，不能再开第二个作业。

```bash
"$TASK_PY" ops/evaluation_pipeline.py --root "$TASK_ROOT" --check-only
# ready后首次运行；stdout和逐stage日志均在数据盘。
"$TASK_PY" ops/evaluation_pipeline.py --root "$TASK_ROOT" \
  >> "$TASK_ROOT/metadata/evaluation_pipeline.stdout.log" 2>&1
```

也有 `ops/wait_and_evaluate.py --root "$TASK_ROOT" --parent-pid <真实P6父PID> --check-only` 可检查自动交接条件；去掉 `--check-only` 后只跟踪这一个已观测的父进程，等P6成功提交选点、原父进程退出、评测preflight通过且GPU空闲才启动评测。它不恢复训练，也不自动重试评测；已有 `metadata/evaluation_handoff.json` 或评测记录时不要重复启动。进度分别看 `metadata/evaluation_handoff.json`、`metadata/evaluation_pipeline.json`，评测stage日志为 `metadata/eval_<stage>.log`。本指南不声称交接已经启动。

评测编排会先在每个主checkpoint的固定10个monitor输入上重复greedy两次，逐字响应一致才运行E3完整orbit。它也生成 `results/training_dynamics/{training_dynamics.json,figures_manifest.json}`，核验开发曲线和训练信号图完整；可选缺图会单独声明。自动部分成功状态是 `automated_complete_human_pending`，仍需真人完成审核。

下述评测目录是统一配方，**本指南不声称它们已经生成**。当前GPU正式训练结束前不并行提交这些预测。先以SFT为例；Exact/Pair替换 `TASK_METHOD`，从各自选择文件读取实际checkpoint：

```bash
TASK_METHOD=sft_seed42
TASK_ADAPTER=$("$TASK_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["checkpoint"])' \
  "$TASK_ROOT/runs/$TASK_METHOD/best_checkpoint.json")
TASK_EVAL="$TASK_ROOT/runs/evaluation/$TASK_METHOD"

"$TASK_PY" scripts/predict.py \
  --model-path "$TASK_MODEL" --revision "$TASK_REV" \
  --adapter-path "$TASK_ADAPTER" --image-root "$TASK_ROOT/data/images" \
  --manifest "$TASK_ROOT/data/manifests/test.jsonl" \
  --output "$TASK_EVAL/e0.jsonl" --run-id "eval_$TASK_METHOD" --seed 42

"$TASK_PY" scripts/validate_run.py predictions \
  --manifest "$TASK_ROOT/data/manifests/test.jsonl" --predictions "$TASK_EVAL/e0.jsonl" \
  --run-id "eval_$TASK_METHOD" --checkpoint "$TASK_ADAPTER" --seed 42 \
  --expected-questions 905 --expected-recipes 597 --output "$TASK_EVAL/e0.coverage.json"
```

同一方法各实验保持相同 `--run-id eval_<method>`、checkpoint、seed，使用不同输出文件，便于E4匹配其E0答案。预测配置/权重内容变化会拒绝复用输出；技术错误须修复后原配置重试。Invalid文本保留并计0，不能当技术错误删掉。

| 实验 | `predict.py`替换项 | 必需覆盖：每个声明的方法单独验收 |
|---|---|---|
| E0 | `test.jsonl` → `e0.jsonl`，默认greedy/32 tokens | 905题、597食谱、905成功推理记录 |
| E3 | `orbit.jsonl` → `e3.jsonl`，默认greedy/32 | 60题、60食谱、1440记录；coverage加 `--require-orbit` |
| E4 | `process.jsonl` → `e4.jsonl`，加 `--process --max-new-tokens 256` | 40题、40食谱、40记录；coverage加 `--prompt-kind process` |

E3/E4均按上面的同一coverage命令替换manifest、预测、题数及输出。E0包含Base、W、三主分支；Base用run-id `eval_base`、输出 `runs/evaluation/base/e0.jsonl`，不加adapter，coverage的checkpoint填本地Base目录；W用run-id `eval_w`、输出 `runs/evaluation/w/e0.jsonl`，adapter及coverage的checkpoint均为固定 `TASK_W`。二者也须全905/597覆盖。E3/E4最低范围只含三个主分支。若四passes SFT预算检查实际触发并完成，另报其E0和额外成本。

这些路径与评测wrapper统一：`runs/evaluation/<method>/{e0,e3,e4}.jsonl`，run-id为 `eval_<method>`；其中主方法名为 `sft_seed42`、`exact_seed42`、`pair_seed42`。不要另建旧的 `runs/eval/` 或使用不带 `eval_` 的评测run-id。

**required coverage清单**：E0每模型905、E1同W bank100题×4=400回答、E3每主模型60×24=1440、E4每主模型40。不可仅合计行数；须匹配qid、presentation、sample、layout、target、模型内容身份、run和seed。每组要求对应coverage JSON的 `passed=true`，缺失整个方法也不能算完成。

## 6 从raw重算主表、E1–E4和图表

以下均为CPU分析，可在数据盘环境重跑。只有覆盖验收齐全才能进入正式表。例子采用上一节拟定评测路径；输入文件尚未生成时先完成相应预测，不生成占位结果。

```bash
# E0主表 + E2：五模型预测及逐模型覆盖均齐备后运行。
"$TASK_PY" scripts/summarize_predictions.py \
  "$TASK_ROOT/runs/evaluation/base/e0.jsonl" \
  "$TASK_ROOT/runs/evaluation/w/e0.jsonl" \
  "$TASK_ROOT/runs/evaluation/sft_seed42/e0.jsonl" \
  "$TASK_ROOT/runs/evaluation/exact_seed42/e0.jsonl" \
  "$TASK_ROOT/runs/evaluation/pair_seed42/e0.jsonl" \
  --paired --output-dir "$TASK_ROOT/results/e0_main"

# E1：已保存的同W候选必须完全相同，不能拿两个on-policy策略的候选冒充同bank。
"$TASK_PY" scripts/summarize_predictions.py \
  "$TASK_ROOT/runs/w_candidate_bank/predictions.jsonl" \
  --candidate-bank --candidate-group-size 4 --output-dir "$TASK_ROOT/results/e1_w_bank"

# E3：严格完整24展示，不使用allow-incomplete-orbits替代正式结果。
"$TASK_PY" scripts/summarize_predictions.py \
  "$TASK_ROOT/runs/evaluation/sft_seed42/e3.jsonl" \
  "$TASK_ROOT/runs/evaluation/exact_seed42/e3.jsonl" \
  "$TASK_ROOT/runs/evaluation/pair_seed42/e3.jsonl" \
  --orbit --paired --output-dir "$TASK_ROOT/results/e3_main"

# E4：summary用模型可见正文view；predict/coverage继续用原process.jsonl。
"$TASK_PY" scripts/summarize_process.py \
  --manifest "$TASK_ROOT/data/review_views/process40_model_visible.jsonl" \
  --predictions "$TASK_ROOT/runs/evaluation/sft_seed42/e4.jsonl" \
    "$TASK_ROOT/runs/evaluation/exact_seed42/e4.jsonl" "$TASK_ROOT/runs/evaluation/pair_seed42/e4.jsonl" \
  --answer-only "$TASK_ROOT/runs/evaluation/sft_seed42/e0.jsonl" \
    "$TASK_ROOT/runs/evaluation/exact_seed42/e0.jsonl" "$TASK_ROOT/runs/evaluation/pair_seed42/e0.jsonl" \
  --embed-images --image-root "$TASK_ROOT/data/images" --output-dir "$TASK_ROOT/results/e4_main"

"$TASK_PY" scripts/make_figures.py \
  --summary "$TASK_ROOT/results/e0_main/summary.json" "$TASK_ROOT/results/e3_main/summary.json" \
    "$TASK_ROOT/results/e1_w_bank/summary.json" --output-dir "$TASK_ROOT/results/figures"
```

E1 bank先用 `validate_run.py predictions` 单独验收：manifest为 `bank.jsonl`，run为 `w_candidate_bank`，checkpoint为固定W，seed42，`--samples-per-input 4 --expected-questions 100 --expected-recipes 100`。正式训练动态来自各run根目录 `reward_groups.jsonl`、`rollouts.jsonl`、`trainer_metrics.jsonl`、`step_times.jsonl`；可用现有 `summarize_pilot.py --run-dir <三个正式run> --output results/formal_runtime_summary.json` 重算日志、信号及耗时概览。这个工具自身声明只提供实施／耗时证据，不能代替主测试或自动声称RL更好。逐step动态曲线须对应原始日志和开发结果；未画出的曲线不称完成。

`summarize_predictions.py`输出 `summary.json`、`per_record.csv`、`per_recipe.csv`；E3另有 `orbit_per_question.csv`、`rank_position_matrix.csv`，E1另有 `dual_reward_groups.csv`。主指标读recipe-macro，micro另列；配对条目须为 `status=computed`，不能只看脚本退出0。bootstrap覆盖食谱采样不确定性，不包含重训seed不确定性。

图表脚本输出SVG/PDF及 `figures_manifest.json`；其 `unavailable` 必须逐项查看。文件存在不表示每张必需图都画成功。E4只衡量显式输出的对应、可见依据和一致性，不把长解释称为内部推理忠实性的证明。

## 7 人工审核：当前pending，保留独立原表

**模型可见正文：** 审核者应查看模型实际获得的文字，不用被裁掉的源全文补全信息。固定audit100中7题正文受裁剪，process40中1题受裁剪；全部标题与步骤保留。派生视图为 `data/review_views/audit100_model_visible.jsonl`、`process40_model_visible.jsonl`；只改正文并附来源哈希、每题cap和渲染哈希，题目、图片、布局、gold、步骤ID、标题均与原清单相同。`review_views_report.json`记录逐题roundtrip及同cohort等价验证。训练、E4预测和coverage继续读取原 `data/manifests/process.jsonl`；只有E4的 `summarize_process.py --manifest` 指向派生process视图。

校准用固定monitor前5题也有1题裁剪，因此校准盲审／summary同样使用 `data/review_views/calibration5_model_visible.jsonl`；校准预测和coverage仍用wrapper生成的原5题控制manifest。5题只用于两名审核者对齐规则，不计入主E4的40题／120响应。源全文不一致的历史证据保留在 `metadata/calibration5_text_view_validation.json`，派生view的正确性另记，不能将历史失败报告覆盖成通过。需要重建审核材料时，以下命令只运行CPU并强制禁用GPU，不改冻结原manifest：

```bash
"$TASK_PY" scripts/build_review_views.py --root "$TASK_ROOT" \
  --include-calibration5 --build-input-packet
```

**输入100题：** 新入口为 `review/input_audit_v3_model_visible/PUBLIC_input_audit.zip`，包含两人的“模型可见正文”离线HTML、空白CSV和中性图片；v2全文包保留为历史，不作为实际输入文字支持的审核入口。分别给B/C其对应页面，独立填可读、可区分、模型可见文字状态、替代顺序、天然编号／水印及备注；不提供模型结果或私有映射。模糊题保留主test/orbit，不能按模型答错删题。人工标签仍为pending。

**E4：** 上节脚本生成 `results/e4_main/blinded/review_items.jsonl`（可嵌图片）及 `labels_reviewer_1.template.csv`、`labels_reviewer_2.template.csv`，另有独立consensus模板。每人审核120个响应、480张图的F/D/S：事实可见准确／有步骤区分性／支持所声称关系，填写yes/no/unclear。代理可以整理可读材料，不能代替人工判断。只共享 `blinded/` 中的审核材料；`blind_mapping.private.json` 和 `scored_process.private.jsonl` 只给研究者。

保存已填写CSV为新文件、不改原始模板。重新执行同一 `summarize_process.py` 命令，追加 `--labels <reviewer1已填.csv> <reviewer2已填.csv>`；讨论后的表可另加 `--consensus-labels <consensus已填.csv>`，不覆盖独立意见。空白为pending，unclear是真实审核后的无法判断，两者不可互换。最终人工审查未完成时，联合指标、分歧和可见依据结论仍未完成。

## 8 最终交付与完成检查

建议最终整理至数据盘 `deliverables/`，这是**待创建的交付目录规划**：

| 交付目录 | 必需内容 |
|---|---|
| `report/` | 报告PDF及源文件：研究问题、条件、真实E0–E4、反证、成本、限制和引用 |
| `figures/` | 来源可追踪的SVG/PDF、对应summary与figure manifest |
| `evidence/` | E0–E4 raw预测、coverage、主表计算、训练动态、真实人工原表/共识、选点及恢复证据 |
| `reproduction/` | 冻结协议／依赖／源码commit／哈希、数据来源及manifest、运行和重算说明 |
| `demo/` | 同题跨方法、跨布局的真实可视化案例；最终视频、脚本和团队实际录制材料 |

不把模型大文件、原始ZIP、缓存、`.venv`、SSH socket或认证信息混进报告材料。选中LoRA是否另附按课程提交规则处理；基础权重与数据来源保留可复现指引。实例账单按实际租用时间记，下载／空闲／失败／恢复和W准备成本都不能隐藏。

完成前逐项核：三分支实际训练和有效性证据；固定开发选点及SFT充分性触发处理；声明的Base/W/E0和三个主终点E3/E4覆盖；E1同候选及动态；两类真实双人审核；配对统计和矢量图；报告、复现说明、真实demo及视频。只有产物与证据全部齐备才可标整个goal完成；“已启动”“有checkpoint”“调研完成”均不足以验收。
