# 草案：扩大训练覆盖以后，Exact-KL是否仍不能稳定超过匹配SFT？

状态：CPU本地规划，未SSH/GPU/训练/冻结。root已另行公告本轮上限从首次preflight起16小时、约人民币25.6元；实际预计8–10小时，完成即停。科学政策仍需独立审阅与冻结，不能沿用旧批时钟或把草案当冻结授权。

## 研究问题与可反驳预期

之前256访问/64更新上，完全枚举消除了24候选内策略梯度的抽样方差，但没有稳定RL提分。本轮扩大到1024不同训练题/256更新，检验“只是后训练覆盖/优化预算不足”是否足以解释这一负结果。固定Exact-KL与原序列SFT进行匹配对照；若两flow都不提升，就否证该**特定规模、学习率和固定beta**下“扩大覆盖即可恢复RL优势”的解释。没有更一般否证RL，也不以新候选空间或额外初始SFT收益冒充RL收益。

所有新臂从同一个SFT768开始。训练pool是冻结`train_all7558.jsonl`的原文件前1024条，不按题目难度/初始概率/后续分数选择；这些题**已被初始SFT768见过**。本轮是已有训练题上的更大后训练覆盖，不是加入初始SFT未见的新数据。flow44保留原pool文件顺序，flow45用固定`random.Random(45).shuffle`对同一1024身份进行一次置换；每个flow中SFT与RL输入/顺序完全相同。两flow共享初始训练模型，不能称完整训练的两个独立seed。

独立CPU核查该pool为1024题/963食谱。flow45在1024上洗牌后的前256与旧256池洗牌仅重合82题；新step64是本新run诊断，不能称旧flow45的同前缀复现或以跨批变化作纯规模因果估计。

| 新臂 | 访问/更新 | 目标 | beta | 主终点 |
|---|---:|---|---:|---|
| flow44/raw_sft |1024/256| 原完整答案+EOS序列CE |0|固定256 |
| flow44/exact_kl |1024/256| 完整24候选上Exact0/1期望奖励减固定KL |1.6363096678310884|固定256 |
| flow45/raw_sft |1024/256| 同上SFT |0|固定256 |
| flow45/exact_kl |1024/256| 同上RL |1.6363096678310884|固定256 |

共同条件：Qwen3-VL-2B-Instruct原revision、同SFT768 adapter、4累积、LR5e-6、全新AdamW、clip1、canonical答案长度C=5含EOS、24候选温度1、原输入裁剪/像素预算。Exact完整排序正确才奖励1，其余0；无手工奖金、重采样、强制gold或额外对齐标签。

各臂每64保存一次，128/192仅工件、不评测，共16个正式checkpoint；先根据实际optimizer字节作数据盘容量门。**step256是唯一主终点，step64 dev200仅预先声明的覆盖曲线诊断，不按dev选点。** 所有四臂各评测固定dev200与16×24完整展示orbit在256，另各step64 dev200；不打开旧test/新确认集。每臂784条greedy预测，整批3136条；16食谱不是384独立样本。依LONG_TERM既有门：mean[RL256−SFT256]达到约3pp且两个flow分别严格为正才进入下一批事前冻结的至少3flow复验；不能直接开启新200确认集，不临时降低为2pp。

## 参考缓存：一次真实评分，两种输入顺序，不能改展示置换

1. 用同初始SFT768对flow44实际1024输入构建全新qref，保存每个候选完整token分数、EOS、候选token IDs、输入audit/像素hash、初始权重与helper源码SHA。不从旧256缓存拼接，也不按概率挑题。
2. flow45严格按`(qid,presentation_id)`查找同一记录；要求完整记录除声明的visit/index字段外逐字等同、显示置换/金标/图片路径/输入audit全部相同。只重排缓存行与`index`，为该flow重新计算manifest/selected-record/refspec哈希；保留每行来源index及原始payload SHA证明。不能更改候选、token分数或图片置换。
3. 两flow的缓存均独立重算softmax、EOS完整性、Exact映射；检查身份集合一一对应，候选数及C=5。训练时原helper再次编码并检验input_hash/candidates与缓存一致。两方法都使用同flow同一参考文件。
4. 参考创建的actual forwards单列为整批一次成本；复用文件不能让四臂都重复计费或声称成本为零。SFT仍有24候选诊断评分，因此也不宣称这是最轻SFT实现。

## 最小实现与不可变边界

不编辑任何旧冻结源码。新增`coverage_contract.py`（预算/池/顺序/引用重排/计时纯CPU定义），`coverage_train.py`（显式派生副本，扩展范围与政策/源码签名），`coverage_pipeline.py`（固定命令队列、四smoke、gate、评测），draft builder，CPUtests/receipt入口，独立`verify_coverage_run.py`及外部watchdog。旧24评分/反传helper直接调用，损失/实际训练更新循环逐字保留并记录lineage；不动态monkeypatch旧模块的全局函数。

已发现旧trainer的硬限制是最多256题和单进程10800秒；新派生副本仅在新命名空间把允许题数拓展为4或1024、明确单进程剩余预算，同时绑定原trainer/helper SHA和新源。不能修改旧文件或依赖动态算法替换。旧独立verifier同样有范围/源码限制，新增审计层需明确接受本轮范围、引用重排证据、新policy签名，并沿用独立数学与实际checkpoint二进制/CRC审计；不能简单把旧256报告改数字。

引用采用两层绑定：所有childpolicy的来源、算法、pool/顺序、路径与确定性生成合同先冻结，GPU产生qref后不重写childpolicy；`generated_reference_receipt.json`在首次smoke更新前密封所有实际cache/sidecar/lineage SHA，parent随后逐命令校验包括receipt自身SHA，每个实际run_signature也绑定所用cache和receipt SHA。此生成规则没有依分数分支或人工选样。

## 计时门、完整性与失败政策

首次CPUpreflight起算共享16小时/绝对UTC截止，外部GNUtimeout监督整个coordinator及全部子孙进程；子进程不另建session或进程组，deadline后最多10秒清理。CPU测试须实际检验孙进程被清理。所有路径、缓存、临时、环境与输出在`/root/autodl-tmp/rl_recipeqa`；原pipeline排它锁与新coverage锁同时获取；健康旧队列不重复启动。

先完整CPU零跳过检查与实际reference-only前检，再每个flow×method各固定前4题1更新共四个smoke（新namespace、独立输出、同起点）。四个smoke必须独立验收后，才能对**全部四臂**训练/所需checkpoint/4×784评测/独立核验估算剩余时间；使用实测最长step、模型加载/保存成本、已有同任务预测吞吐、安全系数与固定I/O裕量。门不通过则不启动任何正式臂；不能删除慢臂、缩短训练、减少orbit或重置时钟让其通过。低分不构成技术停止理由。

粗估（非实测承诺）：旧rawSFT64约13分钟、Exact-KL64约42分钟；线性放大四臂训练共约7.3–7.8小时，新的1024参考约0.3–0.8小时，评测/核验/保存约0.5–1小时。总8.1–9.6小时乘1.5再加10分钟裕量约12.3–14.6小时，因此root公告16小时资源上限，不能把已知不足的12小时门用缩减臂数绕过。最终仍以新四smoke实测gate决定；若不能完整容纳，先报告证据/具体新增需求，不能悄悄缩水或超过上限。

技术失败保留全部源、政策、stdout、最后PID/state与部分工件，版本化修复，沿用本轮原时钟；不因分数低而改奖励/seed/样本或重启。只有全部四新臂训练/两端点诊断/主终点orbit/独立审计齐全才报完整结论。输出并列给Exact、Pair、有效率、recipe级orbit变化、实际候选forward/backward、梯度/clip/update轨迹及墙时；区分更大数据覆盖、更长优化和RL目标，不能把1024阶段相对旧256的净变化当纯数据规模因果效应。
