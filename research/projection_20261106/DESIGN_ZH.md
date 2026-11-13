# 方法候选草案：同一解析 RL 目标的两种投影

**状态：仅本地代码准备。没有 SSH、GPU 验收或训练，没有新增 GPU 预算授权，也未排队。** 当前完整展示诊断及其源码/协议不变。这里的新训练方案必须等 root 另行冻结完整预算、对照、绝对截止、选点规则与独立审核后才能运行。

## 一个明确问题

相同起点、256 条训练访问、64 次更新和相同可验证奖励下，原 Exact+KL 的不稳定收益是否与“将有限动作空间中的理想策略投影到共享 VLM 参数”的方向有关？改变投影方向，能否改善原贪婪生成及完整展示可靠性，而非只降低训练目标？

这是已有策略改进/投影思想在本任务中的受控对照，**不是新 RL 算法，也不是完整 MPO 复现**。新方法把解析 RL 目标固定成软标签后做监督投影；即使提高分数，也只能归因为这种训练目标/投影方式的受控改变，不能称“RL 算法击败 SFT”而隐藏它本身的监督拟合成分。

## 保持什么，改变什么

共同设置：原选定全量 SFT768 adapter；Qwen3-VL-2B-Instruct；同一题/图片/显示位置/训练流及既有 qref 缓存；同一个 β=1.6363096678310884；只有整组排序全对奖励1，否则0；所有24个规范候选均在看奖励前固定。每个规范答案按完整 answer+EOS 的原始 logp 求和，再对24个动作归一化得到 qθ。不重采样、不强制成功、不插入 gold、不删坏题、不引入额外数据或教师答案。

原 Exact+KL 最大化 `E_q[r]−β KL(q||qref)`，等价于最小化 `β KL(qθ||qstar)` 加一个与 θ 无关的常数。新目标仅改为 `β KL(qstar||qθ)/C`。其中

`qstar(a) = qref(a) exp(r(a)/β) / Z`，

C 是所有候选相同的完整 answer+EOS token 数。**不能先除 C 再对候选 softmax**；C 只在最终序列目标与梯度中出现。β 在新目标前显式保留，避免无意中改大/改小目标量纲。每次累积4题，因此每个候选 backward 用 `−β(qstar−qθ)*sequence_logp/(C*4)`。

两个 qstar KL 的 logit 梯度不同；qstar 在首次 optimizer 构造前只算一次并冻结，不能每一步随着当前 qθ 重建目标。原 raw sequence SFT 只对金标字符串求 logp；新投影使用24项的 `qstar−qθ`，包括归一化导数，不等于原始序列 SFT，也不等于只保留 qstar 权重而丢失 logZ 的实现。

## 应主动揭示的数学限制

当唯一正确动作参考概率为 p，令 `t=exp(1/β)`：

`qstar=(1−λ)qref+λ onehot_gold`，`λ=p(t−1)/(1+p(t−1))`。

因此这也可理解为训练样本各有固定权重的、参考策略与金标的软目标监督拟合。p 极小时 λ 仍小，**并没有自动解决低成功概率样本的弱信号**。

在共同起点 qθ=qref 上，forward 投影的梯度仍与原 Exact+KL 梯度共线，倍率

`α=β(t−1)/(1+p(t−1))`。

当前256题真实参考缓存的独立 CPU 审计给出 α约0.7484–1.3782。即使保持 β 与 LR，初始每题梯度大小也不完全相等；**不能说只改变方向、绝对步长完全相同**。二者区别应观察共享参数优化后的目标兑现、跨输入权重和泛化，而不能把一开始的共线方向包装成新推理机制。数学独立审计详见同目录 `MATH_REVIEW.md` 与其参考程序/JSON。

## 实现和工件

- `projection_math.py`：纯 CPU 数学、冻结的 tuple/FrozenQStar、β缩放与EOS计数约束。
- `projection_train.py`：新路径下适配旧训练循环；导入旧输入、scorer、可微完整 logp、梯度统计等已验证实现；不改旧文件。不重新生成 qref，只接受原缓存及签名 sidecar。
- 所有 CLI 字段（包括 revision、图像/text budgets、logp tolerance、mode）必须与新锁定协议完全相同；协议本身、源码、manifest、原 qref/config/completion 内容 SHA 和初始 base/adapter identity 均绑定。
- 实际图片字节、原缓存输入签名、全24答案与EOS、源金标映射、q/logq与raw token logp均在 CPU 核验；冻结qstar落盘及签名序列化通过后才加载模型。每步再检查源码/数据/参考/qstar不可变。
- fresh AdamW(.9,.999,eps1e−8,wd0)，LR5e−6，clip1，常数scheduler，accum4；eval关闭dropout但保留autograd。formal固定256/64/save16；**独立锁定smoke** 固定原full manifest前4题、既有reference4_smoke、4访问/1更新/save1。不允许用smoke adapter初始化formal。
- 保留候选 no-grad forward、可微forward/backward次数、GPU峰值、每步实际更新/梯度、全部访问、训练目标及两种KL、目标/参考SHA、checkpoint及optimizer/RNG工件。相同训练访问不等于相同计算量，必须另报实际墙时与候选前反向数。
- 当前已通过12项本地CPU测试，含24维logit与非线性共享参数 finite-difference、accum4、EOS导数和C、β系数、目标冻结、初始共线/混合、原reverse目标恒等式、旧source SHA、JSON-native CLI与draft-before-GPU拒绝。**这些不等于真实GPU backward/optimizer验收**。

## 正式对照和必要门槛（尚未冻结）

root 当前候选：在两个既有流新增 FWD；再用同256题的新预定shuffle构造第三流，给 raw SFT、原 Exact+KL、FWD三臂，约5个新增formal arms。这个候选尚未承诺预算，也未写入运行队列。第三流不是新的全量 SFT 初始化，仍须披露共同起点限制。正式实现应为所有新臂冻结曝光、seed、endpoint/选点规则，既有对照要逐项核对原artifact，不因新臂低分才追加更多seed。

必须先完成：独立源码/数学审核；CPU完整缓存target验收；独立4题/1步真实GPU smoke与重算验证；新阶段整批计时gate、排它锁、新绝对截止和真实硬停保护；失败证据与无自动续跑政策。外层新coordinator负责整批锁/健康gate/硬截止；当前trainer草案不能独立替代它。

新200题确认集继续封存。主结论看同模型/输入/曝光/贪婪解码的稳定差值；闭集MAP或四选项改善单独报告。若只是训练目标更好、dev或完整展示不变，结论就是“投影优化兑现与泛化脱钩”；若只有单流提升，仍是不稳定信号。任何提升均不能直接称公开SOTA。

## 现行外层协调器候选（替代上面的五臂想法，尚未冻结执行）

为了节约预算，root 当前将下一批收缩为 **仅在两个既有 flow 各训练一个新 FWD 臂**；旧 SFT/Exact-KL 的 fixed64 作为已知开发结果的历史对照，明确并非重新同步随机分组。所有新臂也只评 fixed64，不用 dev16/32/48 选点，不做第三 flow，不偷偷将旧 flow45 SFTstep48 当 fixed64 对照。

新 `projection_pipeline.py` 草案拟执行：完整远端 CPU 测试（当前29项，必须29通过/0skip/无CUDA初始化）→独立锁定4题/1步 FWD smoke→独立 `verify_projection_run.py` 验收→两完整训练臂与全部评价的整批时间 gate→FWD44/45 各256访问/64更新→每臂200题 greedy 与原前16食谱×完整24展示→覆盖、原始回复独立重算、逐食谱差值及完整工件。

新阶段候选最多4小时、按1.6元/小时增量上限6.4元。**尚未最终公告、冻结或启动**。绝对截止以首次 preflight 为起点，最长14400秒，不可在 child launch、第二flow或重启时延长。真实 SIGALRM 同时约束CPU和GPU；到期后仅有最多10秒 owned child/log 清理。排它锁阻止与其他GPU任务并发；fresh stage与child目录，无resume。所有三个child policy（smoke/FWD44/FWD45）须在第一步更新前完整冻结，不能根据smoke或成绩修改正式参数。

时间gate用真实一完整步smoke：128步训练、2次加载与8个checkpoint的保守非step余量，外加两套200+384评价；预测每条耗时用上一批完整四模型1536行输出的最慢相邻arrival间隔，包含预处理/生成/写盘。整体1.5倍余量加600秒CPU验收。smoke必须真实观察每题24次backward和25次scoring forward，不能用稀疏/零backward计时低估正式全枚举。gate不通过则两个formal臂均不启动；gate是运行估计，硬截止独立执行。

外层除了锁历史 raw/config，还重验原四控制的run_signature/outcome：相同初始权重、对应训练流/seed、256访问64步、相同参考缓存字节与各自原β。源码、所有测试（含真实Torch CPU参考）、child政策、manifest/qref/config/completion、历史训练/评价工件均锁SHA。完整报告两flow及其所有负结果，当前只有历史开发数据上的探索对照，不能升格为稳定RL泛化证据。
