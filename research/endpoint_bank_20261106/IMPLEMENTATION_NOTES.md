# 固定末点评分实现与执行边界

当前只有本机草案、CPU 测试和已有数据/元数据的读取；没有 SSH、GPU、模型调用、冻结或远端启动。此文不是运行验收。

`build_draft_policy.py` 从已逐 SHA 验证的最终 coverage 快照生成草案：共同 SFT768、四个固定 checkpoint256 的真实文件身份，原训练文件前256题（254食谱）、既有 dev200，以及初始 train256 的原参考缓存来源。保留所有原记录字段，包括训练 `manifest_set=train_all7558`。旧五个 dev greedy 原文/config 只绑定为独立分析来源；没有 train greedy，也不生成新回答。

一个父协议冻结九个任务及顺序；不需要九个各自起算的新时钟。所有 scorer 签名逐个绑定同一父协议 SHA、源代码、模型/图片/输入身份和共同首次 preflight/绝对截止。草案时钟为空且不能执行；首次远端 preflight 之前由 root 统一赋予原始 10800 秒时钟和对应绝对 UTC，独立二审后才可启动。

执行顺序固定为：

1. 原生 GNUtimeout 拥有科学进程组，清理余量最多10秒；Python、缓存、临时与所有数据必须在现有数据盘，四个线程环境变量预先设为4。CPU 门检验精确测试数、零错误/失败/跳过，并实际验证另一个有1.3秒硬上限的 CPU-only TERM 忽略子孙探针。探针没有科学/GPU工作。
2. 核验五个实际模型的完整物理身份，再复制初始 train256 的真实缓存前缀，保存原 payload、逐行 lineage 和完整来源签名链，由独立 verifier 复验；复制阶段没有新模型加载或前向。
3. 九个任务各独立加载模型，评分固定前2题，共18次额外输入、450次候选 forward。每个 pilot 有独立原始工件和 CPU 数学/身份验收。
4. 全九个 pilot 成功才计算时间/空间门。每任务评分估计固定为 `row0 + (N−1)*max(row1, historical_floor)`，加外层真实命令墙时减两题评分时间的加载/指纹/IO余量，再整体乘1.5并加600秒验证余量。第0题的 literal 全位置校验只计一次；历史 floor 从绑定 SHA 的真实 `reference_only_outcome.json` 计算为 `2485.4406706159934/1024≈2.427188秒`。这是两题的资源预测，不是时长上界，也不把不同题目 row0−row1 的差值当作纯 literal 因果成本。门持久化完整九份 pilot 时间/字节输入，供独立重算。空间采用实际 pilot 字节数外推、2倍余量及额外1.25GiB。任何门不通过均禁止正式评分，不删任务或缩小题数。
5. 正式九个任务共2024次新输入、50600次候选 forward。每个 scorer 在加载模型前读取持久授权：CPU/cache 先通过，正式任务另需9份独立 pilot 收据与整批门；匹配 policy、任务、phase、时间和 PGID，防止只靠环境变量早发 GPU。完成后逐任务独立验收，全部完成才记录批次完成。

新评分仅调用已冻结旧源码的候选 suffix、完整 forward 和输入编码函数，不调用其 dev-only main、greedy 或 generation 路径。全24候选在读题前由固定排列生成，所有完整字符串评分含 EOS、C=5；每题另做 candidate0 的独立数值校验，每个 pilot/formal 的第0题使用 literal 全位置 logits。模型始终 eval、所有参数 requires_grad=False，并在 inference_mode 下评分；没有 optimizer、backward 或 generation。

`required_versions` 直接来自原缓存已封存的环境版本，不取本机版本；scorer 在加载 processor/model 前要求实际环境逐项相同。cached 包装中的版本明确是历史参考环境来源，不伪称本阶段重新加载模型的观察。

每任务保存 `scores.jsonl/signature.json/completion.json/independent_verification.json`。原始行保留24×5 token logp、完整候选 token IDs、literal 检查、实际输入审计、图片 SHA、模型与配置身份、计数和时间。完成收据比较实际磁盘模型身份，以及全392个语言 LoRA 参数和视觉参数的运行时前后指纹。推理 LoRA 指纹显式按参数名称筛选，不使用仅遍历 requires_grad 的旧 helper，避免冻结状态造成空指纹。

父状态只报告覆盖、成本和零更新计数，不计算 MAP/J 或选择模型。新的 CPU 数学 verifier/analyzer 由另一位作者独立实现；正式结果完全结束并逐 SHA 下载后，才进行目标、解析 q*、兑现率与旧 dev greedy 的分析。任何 q* 都依赖真值、不可部署；没有新确认集、测试集或 SOTA 声明。
