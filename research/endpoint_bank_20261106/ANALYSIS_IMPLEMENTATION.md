# 独立数学、逐任务验收与完整末点分析

这里是 PLAN 的 CPU 实现说明，不是新 GPU 结果。新增六份源码仅用标准库：`bank_math.py`、`verify_endpoint_scores.py`、`analyze_endpoint_bank.py` 及其三个测试文件。数学与验收不导入生产端评分器。旧科学源码不修改。

## 入口与完整性

协调器对 cached / pilot / formal 三种角色逐个调用：

```bash
python3 research/endpoint_bank_20261106/verify_endpoint_scores.py \
  --policy /root/autodl-tmp/rl_recipeqa/research/endpoint_bank_20261106/PROTOCOL_V1.json \
  --phase pilot --task-id initial_sft768__dev200 \
  --run-dir /root/autodl-tmp/rl_recipeqa/exploratory/endpoint_bank_20261106/stage_v1/pilot/initial_sft768__dev200 \
  --output /root/autodl-tmp/rl_recipeqa/exploratory/endpoint_bank_20261106/stage_v1/pilot/initial_sft768__dev200/independent_verification.json
```

逐任务验收含：独立固定源/来源SHA、模型与bank/phase签名、实际照片与磁盘权重流式SHA、C5含EOS的24候选、literal数值检查、原始顺序/分母、零更新/反向/生成、冻结eval状态和非空392项language LoRA指纹。大权重文件分块哈希，不整块保存在内存。缓存模式逐值核原始1024参考的前256行、六项源回执、同一实际run的signature/spec/audit、照片身份和复制lineage；缓存不伪造新模型指纹或新forward。

GPU评分签名必须绑定事先持久授权。独立核 CPU receipt 同policy/source/首时钟且零跳过，缓存审计先完成，formal另要求全部九pilot审计与完整时间/空间门；授权发出不晚于score开始。相同PGID与原三小时截止保持不变。

软件版本由封存source_config的versions固定；新评分逐项绑定实际runtime版本，缓存明确标记来源为封存cache。不得悄悄用不同软件版本的初始参考和末点分数作同题比较。

全部结束后，在稳定的远端证据导出上运行：

```bash
python3 research/endpoint_bank_20261106/analyze_endpoint_bank.py \
  --policy research/endpoint_bank_20261106/PROTOCOL_V1.json \
  --root .state/remote_principled_20261105 --require-all
```

分析器首先核38条命令、九pilot、九formal、缓存256与零更新完整门，未齐时不读取任何新候选分数。之后独立复算全部2280单元，并绑定真实远端逐任务物理审计。缺少本地图像/权重时明确只依赖已绑定的远端物理回执，不能声称本地重验tensor/CRC。远端审计必须包含核心metadata以及由签名派生的全部实际照片、权重和辅助文件SHA/字节；不能用空回执和一个“physical=True”布尔值替代。

默认生成全新目录 `complete_analysis_v1/`：2280行 `cells.csv`、保留空层的30行 `strata.csv`、912行配对末点差 `paired_endpoint_differences.csv` 和来源/统计/限制 `summary.json`。已有输出或 `.partial` 均不覆盖。完整阶段计数为2024正式新输入+18额外pilot、51050新候选前向、18模型加载；缓存256单元不重新计模型调用，原1024参考25600次历史forward单列。

## 数学、单位和解释

24个完整排列的5-token logp先求和（含EOS），再温度1归一化。概率与qgold为条件24动作概率；KL、熵、NLL和log canonical mass使用自然对数。原始完整序列NLL、原始CE/C与条件NLL分列。C=5仅是共同loss缩放，不进入 `exp(1/β)` 赔率。

统一 `β=1.6363096678310884`；计算 `J=qgold−βKL(q||qref)`、解析 `q*∝qref exp(r/β)`、`J*=β logZ_reward` 和 `J*−J=βKL(q||q*)`。q*归一化用log1p/expm1稳定计算。SFT的共同J明确为反事实诊断，不能称其监督训练损失。

同时计算 `Jref=qref_gold`、可用增益 `J*−Jref`、已实现增益 `J−Jref`。各bank/端点/分层报告 `sum(已实现)/sum(可用)`，恰零分母为NA、负值不裁剪，不平均逐题小分母比值。连续报告，不制定事后large/small-gap阈值或自动选择下个方法。

MAP在重建logscore精确相等时取全局canonical索引最小者；与endpoint、runid和gold无关。另报精确tie集合、gold最佳/最差排名与1e−9 nats近tie数量；近tie只报告，不改变选择。三个层由共同初始策略及gold依赖q*固定：初始MAP正确、初始错而q* MAP正确、初始错且q* MAP仍错；五端点共用分层。

q*使用真值，不能部署，不是heldout准确率上界，也不证明共享VLM参数空间中的全局可实现最优。q* MAP仍错，不禁止另一个J较低的策略取得正确MAP；测试明确包含该反例。小绝对gap也可能伴随极小可用增益，因此不能单独当作充分优化证明。

仅dev200对齐已有五模型开放greedy原文/配置/权重/输入审计。逐行decoder、seed/runid及坐标核对；包括invalid在内保留全部200，另报MAP/open正确性2×2。train256没有既有匹配greedy，明确为None，不新增生成。条件候选解码与开放greedy的差异不自动算RL收益。

## 设计评审与成本判断

该设计可区分“同题末点是否兑现同一目标”“固定KL解析目标是否偏向保留初始MAP”“条件MAP与开放greedy是否错配”，比变化输入/变化θ的在线均值提供更直接、可反驳的证据。它依然是目标与模型之间的诊断，无法仅凭这些结果证明视觉/推理能力的因果机制。

256训练题包含254食谱且已被初始SFT消费；dev200已被反复使用，分开报告，不合并成泛化分数。未新增数据、确认集或checkpoint选择。2.43秒/题粗估2024正式评分约82分钟，三小时有合理余量；但九任务首两题不保证覆盖全bank最长输入，1.5倍时间门不是统计保证。真实全部九pilot门优先，若超时保留不完整证据，不减任务、不换题、不改科学口径。

最终预声明资源公式逐task为 `row0+(N−1)*max(row1,封存1024参考elapsed/1024)+max(0,outerwall−row0−row1)`，全九task之和再乘1.5并加600秒；缓存/CPU/pilot已耗时间从共同三小时余额扣除。独立验证器重算来源、每个分量及空间门，完整分析器再把outerwall与最终state真实命令逐项对照。第一行额外literal检查只计一次是预测模型，不是已识别其纯因果耗时。

当前本地测试仅合成反例和已完成旧flow44的8个访问；不读取新endpoint-bank GPU结果。最终真实物理门、完整38命令和2280单元仍须实际验收。
