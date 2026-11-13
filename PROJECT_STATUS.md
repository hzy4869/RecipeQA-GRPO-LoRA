# RecipeQA项目状态

更新日期：2026年11月5日，Asia/Hong_Kong。实时状态以远端metadata与实际PID为准，下列步数为带时间的快照。

## 当前阶段

**P6正式训练中，主SFT训练和开发选点已完成，Exact-GRPO正在训练。** 同一起点W的SFT、Exact、Pair各20步pilot均通过工件验收，87项远端测试通过。正式SFT完成256更新/2048真实访问，256次梯度有限且非零，视觉SHA前后相同，四个检查点完整，artifact_validation passed。UTC 2026-11-04 17:18:45（香港11月5日01:18）Exact已完成54/1024更新。尚未产生正式test模型输出，不预判方法优劣。

正式父进程PID **11447**；原SFT子进程11449已完成，当前phase为exact_seed42、子进程12625。权威状态`metadata/formal_training_pipeline.json`；父进程stdout `metadata/formal_training_pipeline_cpu4.log`，各阶段日志`metadata/<stage>.log`。父进程等待子进程时显示S为正常情况。后台作业不依赖本机SSH持续在线。

人工输入审核：**用户暂无法安排，pending**。训练与自动分析继续；不代填人工标签，不将未审核内容称为人工验证。E4双人语义盲审仍需真实团队审核或在交付中明确缺项。

## 权威范围与资源

- [RESEARCH_PLAN.md](RESEARCH_PLAN.md) v1.2优先于早期调研候选方案。
- Qwen/Qwen3-VL-2B-Instruct；RecipeQA Visual Ordering的文字引导四图自由排序衍生任务，不直接与原四选一榜单比较。
- 同W的SFT-cont、Exact-GRPO、Pair-GRPO；Base/W为起点。主2pass共享1024题池及2048次prompt访问，不称equal FLOPs。
- E0主测试、E1同候选反馈/训练动态、E2局部错误、E3全24展示及目标序位×显示位置、E4固定40题显式过程；不增模型或数据集。
- SSH：root@connect.westd.seetacloud.com:31755。认证秘密不保存。项目根目录`/root/autodl-tmp/rl_recipeqa`。
- **所有模型、数据、环境、缓存、临时工件和输出必须在数据盘/root/autodl-tmp。普通/tmp属于系统盘，禁止用于项目工件。** env.sh已重定向相关缓存。
- 实测GPU报告RTX 4080 SUPER、32760MiB、BF16可用；12 CPU配额、约62GB内存。数据盘50GB，最近约35GB空余；系统overlay约524MB使用。
- 租期约两周，人民币总预算1000以内，当前约1.6元/小时。按14天估算537.6元，实际额外费用/计费起点/余额尚未知。费用按实例实际计费时长，包含下载、CPU和空闲，不以训练秒数替代。当前显存够用，无需加卡。
- 实际提交日期未提供；既有20天窗口不因接入延迟重新顺延。

## 当前执行约束

科学锁源commit `5a07ce6b87a8ca57bcf0aab116fe1d572383e102`；`configs/protocol_lock.json` SHA256：

```text
9089f48c5f3b571dfa5fddd31b843a0ae2c22296d9ca4a4d03ee05ab7a75f18c
```

已有冻结src/scripts/ops/configs及RESEARCH_PLAN不得热改。PROJECT_STATUS、新交付文档和新编排文件可维护，但不得改变科学输入/评分/选点。新增文件需单独记录版本和运行hash。

在首次正式更新前，实测默认Torch72线程对12CPU配额严重超额，8条固定探针的编码张量逐字节相同，CPU4编码显著更快。原formal尝试在**0更新**时终止并完整归档，没有丢弃已训练权重。各正式/评测进程统一在source env.sh后设置：

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
```

独立执行环境修订记录`metadata/runtime_threads.json`；原协议锁保持不变。旧尝试`runs/sft_seed42_cpu72_preprocessing_aborted_20261104T164230Z`及`metadata/cpu72_preprocessing_aborted_20261104T164230Z`保留，开销计入总账。探针仅证明CPU编码提速，不将其14–25倍比例外推为GPU训练提速。详见[线程探针](research/runtime/cpu_thread_probe.md)。

## 已核验的证据

| 项目 | 证据与结果 |
|---|---|
| 研究与协议 | 22篇指定实验阅读、数据/科学/训练/实施独立评审、数学576种映射和三种解析控制已完成；research目录 |
| 依赖 | Python3.12.3、Torch2.8.0+cu128、Transformers5.22.1、TRL1.17.1、PEFT0.21.2、Accelerate1.15.0、Datasets5.1.0已锁 |
| 模型 | HF revision89644892e4d85e24eaac8bacfd4f463576704203；下载完成marker、LFS权重SHA256及辅助文件身份核对通过 |
| 图片 | ZIP完整校验并提取137,996文件；所需8,871唯一文件名全部RGB可解码；442组同名副本仅在内容SHA一致时消歧，未换题/删题 |
| 固定样本 | 1024训练池、256 warmup、32 debug；val200、test905/597 recipes、bank100、audit100、orbit60×24、process40 |
| Base开发 | 200题技术完整；valid54/200，Exact24/200，触发预定warmup规则；不是正式test |
| W | warmup完整64更新/512访问，按最早valid≥98%选checkpoint-16：valid199/200、Exact85/200；选中W仅128访问，完整warmup开销仍记账 |
| W身份 | adapter文件SHA256 `5b8768ecc20cfc031ccf21ed8509e66c91170d384c9f747142dec53dcb1ab31a`；选择metadata/w_selection.json |
| 有效拟合 | pilot_sft_debug_v2：20更新/160访问，debug Exact5/32→31/32、loss .809→.0055，视觉SHA不变、LoRA变化；仅训练拟合证据 |
| 同W pilot | SFT20更新/160访问；每RL20更新/40组/160回答，results/same_w_pilot_summary.json三项completed_evidence_summary且issues[] |
| 共同起点 | 三pilot运行时initial_adapter_fingerprint均 `d1fb4a12a73f7b90b2388d6828f1f331c2dee477446aeee472fb35d36b92a028`，正式SFT亦一致 |
| 恢复 | SFT和Pair实际18→20完整恢复；RNG、LR/scheduler、输入访问一致，但BF16数值轨迹不达bitwise/当前容差等价。CPU392个LoRA及AdamW重载逐值相同，具体数值差源未定位；不得声称位级等价或指定某kernel为原因 |
| E1同候选 | W固定100题×4：49组两reward均可分，41组仅Pair可分，10组均常数；valid394/400。41额外组中39全合法，40存在合法错误排列间差异；不把此结果当泛化改善或density单因果证明 |
| 输入sanity | W固定20题实图Exact7/20、白图1/20，均valid20/20；只作排错，不证明正确grounding。解析随机/照抄及两布局控制已生成 |

主要说明：[pilot与预算](research/runtime/pilot_budget_and_bank.md)、[E1格式分层](research/runtime/bank_format_stratification.md)、[恢复精度](research/runtime/resume_precision_review.md)。E1真实SVG/PDF位于远端`results/e1_w_bank/figures`。

同W pilot旧72线程非保存wall中位SFT5.56s、Exact7.38s、Pair7.12s；据此约4.5–4.7小时仅为三个主训练循环组件，不含加载、正式预处理、开发/终点评测、补充SFT或故障余量。正式4线程SFT最近每步约4.4s；完整预算继续用实测更新，不把两个口径混用。

正式SFT实测汇总`results/sft_seed42_training_evidence.json`：completed_evidence_summary、issues[]；完整单次调用1274.93秒，trainer loop1115.20秒，非checkpoint相邻wall中位4.314秒，峰值allocated5.246GB。此数不含随后四开发点与之前失败尝试，不是平台总计费时长。

SFT开发64/128/192/256步的Exact为56%/65.5%/67%/67%，Pair为.8775/.900833/.9000/.903333，valid为99.5%/99.5%/100%/100%。按Exact→Pair平局规则选256；256对192的Exact净增0/200，预先规定的四pass控制未触发。该事实不等于证明SFT充分收敛，不作为正式test性能。选择及规则工件为`runs/sft_seed42/{best_checkpoint,sufficiency_trigger}.json`。

## 正式队列与评测

1. P6：SFT256/save64→四点val200；若末点最优且256对192净增至少4/200，从同W独立跑4096访问/512更新四pass补充→Exact1024/save256→四点val→Pair1024/save256→四点val。Exact→Pair→较早点选点，不使用test。SFT4pass主流前2048行byte-identical，prefix审计已保存。
2. 每分支实际训练工件门需passed。不要盲目重开整条formal队列；失败先查PID、末完整checkpoint、签名和真实错误，再显式同参数完整恢复，保存失败证据。
3. 新增`ops/evaluation_pipeline.py`已通过源码审阅、7项CPU安全控制和远端真实40/5题审核view核对；只在P6三主终点选定后顺序跑E0、复用E2、E3、E4，复用现有E1。`ops/wait_and_evaluate.py`观察已存在父11447的进程身份，必须见其成功选点并退出、GPU无计算进程且预检通过才启动一次评测；失败/失踪/源码变化停并记证据，不自动重启训练或恢复评测。当前待root部署并arm，不能把预检当已启动评测。
4. E0五模型各905；E3三主各60×24；E4三主各40、process输出256tokens，5个W开发校准例单独统计。严格覆盖、固定身份、固定recipe bootstrap，不将24展示或两位审核者当独立样本。
5. 主终点评测尚未运行。若触发4pass，对其补充E0和成本，并据结果收窄结论，不能隐藏更强SFT。

## 人工审核与交付缺项

- v2包显示原食谱全文，仅保留历史。v3模型可见正文包已完成：`review/input_audit_v3_model_visible/PUBLIC_input_audit.zip`；audit100中7题、process40中1题正文受1536token预算裁剪。`data/review_views/{audit100,process40}_model_visible.jsonl`逐条保护题目/图片/布局/gold/标题，正文渲染roundtrip通过，原冻结输入不变，不借此筛题。
- 5题开发校准也有1题裁剪，已生成`data/review_views/calibration5_model_visible.jsonl`，逐题cap/可见prompt哈希与实际W开发推理一致。共同报告`data/review_views/review_views_report.json` SHA256 `5ada579f8f7b4ed16b3fee294aa125591d45dc6bc128eb50fdb28eaab027fc70`；各冻结文件前后hash通过。预测/coverage仍用原manifest，仅人类summary用派生view。
- `scripts/plot_training_dynamics.py`补真实开发四点/实际曝光和on-policy动态图；真实SFT四点及PDF视觉检查通过，三分支合成控制仅用作CPU测试且已删除，不是研究结果。当前真实图`results/training_dynamics_sft_validation`只代表SFT。统计解释边界见`research/runtime/analysis_interpretation_checklist.md`。
- 用户暂无法安排两人审核。输入可辨性/歧义和E4事实/区分性/支持关系标签保持pending；自动字段完整性和来源步骤匹配不冒充语义验证。
- 后续仍需真实E0–E4结果、统计图、报告、复现说明、真实demo/视频素材。录制、署名和最终提交由团队负责。整个goal保持active；训练启动或自动评测完成都不等于项目完成。

## 恢复入口与下一步

1. 先查`metadata/formal_training_pipeline.json`及实际父/子PID，当前阶段日志、trainer_metrics/gradient_audit/step_times和完整checkpoint标记。不要因观察超时或父进程睡眠重启。
2. 审核并部署新增评测/成功交接编排；formal失败则停并记录，成功后才接评测，不自动改配置或反复重试。
3. 完成人工可见正文view、执行指南、统计解释清单，保留每项来源和未完成状态。
4. 用真实主结果完善报告与演示；如暂缺人类标签，明确未完成，不填零、不用代理视觉判断充当人类。

本机SSH复用socket `.state/ssh/remote.sock`。先BatchMode检查master；失效可按用户已授权认证恢复，不保存密码。已完成下载无需重启；历史失败准备链/旧下载日志仅作溯源，不作为当前等待项。

## Goal续行审计

外部SSH依赖已解除。最近已完成三同W pilot及恢复核查、协议冻结、实测CPU瓶颈修复、主SFT全部256更新/2048访问和四检查点验收，属于持续实质进展。人工审核不可用尚不阻止自动实验，不能据此提前blocked或complete。
