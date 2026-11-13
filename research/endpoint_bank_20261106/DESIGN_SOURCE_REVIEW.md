# 固定末点bank：独立设计与源码总审

**本地设计/源码审阅PASS，`ready_for_remote_preflight=true`。无剩余源码阻断。** 这只表示可在不改科学设置的前提下冻结并执行真实远端CPU/原生watchdog及九项pilot，不表示远端门已经通过；formal必须等真实全部九pilot及资源门通过。

审阅公开草案：`DRAFT_PROTOCOL_V1.json`，SHA `9ead4301e8e6f65c8fe218bb6c0ee9b6ae457712e5abc61e72b77e34c747c5af`。PLAN SHA为 `44b20da20d05472fc018b7d13c2dcc1b81aa6569f8d91dc6afaefdaea0766593`。13个新源+3个固定旧源共16项，另62项来源artifact；完整SHA列表见同名JSON，实际逐项核验。旧coverage20科学源及其ANALYSIS_PLAN未改。

## 设计足以回答的问题

五个固定末点对同一train256/dev200评分，能把先前在线变化θ/变化题的描述推进到固定θ/固定输入的比较。它可区分目标是否兑现、固定KL解析目标是否保留初始错误，以及conditional MAP与既有开放greedy是否不一致；不能仅凭这些结果认定推理或视觉能力的因果改善。

原始train7558的前256完整记录未改，保留`source_split=train, manifest_set=train_all7558`，包含254食谱；开发为既有200题/200食谱，二者食谱无交集，仍分别报告。初始train256复用已封存1024真实参考的原前256行。正式新输入2024分为九固定任务，pilot额外九任务×2题=18；全部五列/两bank共2280单元。没有新增greedy、训练、optimizer/backward、候选过滤、真值注入或确认集调用。

q*、J*、gold rank和三分层只是真值相关分析参照，不能部署，也不是heldout准确率上界。MAP精确tie取canonical最小索引，近tie只报告，不读gold决定选择。SFT共同正β J为反事实诊断，原始CE/5与条件NLL分列。新增可用/实现J增益采用同组总和之比，恰零分母NA，负值不裁剪；不均值逐题微小分母比，不设事后阈值或自动挑下一方法。

## 独立验证结果

- 本机完整64项CPU测试：64过，0失败、0错误、0skip、0expected-failure/意外成功，未导入Torch。测试不是远端执行门替代品。
- 另做60位Decimal的192个非退化gold案例；qgold/KL/Jgap/Jref/可用与实现增益均独立重算，最大残差不超过1.78×10⁻¹⁵。72次精确/近tie选择不随gold改变；8232种字符串与冻结旧开放答案评分器等价。
- 真实7558源SHA、原256完整记录、254食谱及dev200来源/单位通过。实际cached source的六回执、同run signature/audit/spec/input/初始模型及逐行index链均核对，缓存不是重写标签或重新评分。
- 实际本机构建临时draft与公开draft SHA完全相同；状态draft、两个时钟为null，未冻结/部署/执行。
- 三组独立资源公式探针（首题慢、后续题慢、历史floor主导），均覆盖完整九任务并与手算一致。

## 已关闭的关键防护缺口

1. 缓存源不仅有`passed`，还串同一实际run的signature、source spec/manifest/cache、模型、input、版本和六项旧SHA。复制保留完整source payload，缓存新model/forward计数为0，不伪造新runtime指纹。
2. live scorer在加载processor/model前先校持久授权；授权绑定同policy/source/首时钟的零skip CPU与缓存独立审计。formal授权再绑定全部九pilot和整体资源门，拒绝错phase/task/policy/verifier、未来授权或先跑后补门。
3. 独立verifier自行pin原train/dev来源与16源闭包、旧三源SHA及实际正在执行的数学/验收源。逐任务核C5完整EOS24、literal检查、零更新、392项非空LoRA指纹和实际模型/照片文件。
4. 本地完整分析先核38命令、所有计数/PGID/绝对截止及最后commit时间，再读取新分数。缺任务、失败/重复命令、活动子进程或晚于commit均拒绝。远端物理报告必须包含当前metadata及签名派生的全部照片/权重/辅助文件回执，空集合或少物理项不能以一个`physical=true`蒙混。
5. 五个旧dev greedy逐题核模型/input/pixel编码审计、实际五项row decoder、seed/runid及完整generation配置可比性；包括invalid的全部200均保留，另报正确性2×2。train无匹配旧greedy，明确为None。
6. 大物理文件流式SHA而非整块缓存；本地图像/权重未导出时明确绑定真实远端物理回执，不宣称本地tensor/CRC复验。

## 资源门与时钟

每任务采用`row0+(N−1)*max(row1, historical_floor)`，加实际外层命令墙时减两行耗时的加载/指纹/IO；九任务合计乘1.5，再加600秒验证余量。历史floor为真实旧reference-only `2485.4406706159934/1024≈2.427188秒/题`，其outcome SHA已纳artifact。

源码明确全位置literal logits只在每任务index0调用，故一次首题成本不应重复乘所有N。两个pilot题不同，首末差不能解释为纯literal/热身因果，row1也不是稳态最坏上界；历史floor及安全余量降低过乐观预测，但不构成耗时保证。真实外部GNUtimeout仍执行首preflight起10800秒和原绝对UTC更早者，清理最多10秒，版本/修复不重置。所有评分子进程保持原PGID，无detach；CPU-only原生watchdog探针故意创建独立受限测试组，验证TERM忽略的child/grandchild均被KILL，其最长约1.3秒且不触碰模型。

**本机没有`timeout`命令，因此未实际运行GNUtimeout原生探针；此项远端仍pending。** 本机64测试没有因此伪记skip或把静态审阅冒充原生清理实测。远端还须实际64项零skip、CPU/native、五模型物理身份/版本、缓存审计、九pilot和整体空间/时间门，才能启动全部九formal。

冻结只允许把同一草案status改locked、写实际共用first-preflight和+10800秒截止，并追加不可变审阅/CPU/分析/notes回执SHA；不得改16源、62现有artifact值或科学条件。该批准不授权新硬件、额外数据、换checkpoint、放宽总预算或缩减任务。

本次审阅0 SSH/GPU/模型调用，未改作者源码。原人工歧义审核仍是最终结论限制，没有被增设为技术工作前置阻断。
