# Forward / Reverse KL 投影：独立数学评审

**状态：数学、固定256题CPU参考及trainer草稿的静态/metadata门槛审查通过；真实Torch CPU参考尚待远端0skip验收，协调器和微型GPU仍未验收。未运行任何新训练、GPU或SSH，不构成执行预算或部署批准。** 本文不修改正在健康执行的orbit批次或任何冻结源码。

本次读取 `NEXT_METHOD_DECISION.md`、旧 `finite_policy_train.py`、已完成真实梯度工件及 `TRAINING_TARGET_AUDIT.json`。独立核验程序为 `independent_projection_reference.py`，结果 `INDEPENDENT_MATH_REFERENCE.json`。当前被评数学源码SHA为 `fa6e354e26ff7dd62fa6ab235abf7e9912d9793cf2d5803d68192dc23e570184`；源码改变后须重跑参考，不沿用本哈希的通过记录。

## 1. 目标、β及梯度正确

保持同一固定参考 `qref`、β=1.6363096678310884和完整排序正确0/1奖励：

- 理想策略 `q*=softmax(log qref+r/β)`，在更新前冻结，不能随当前模型重算参考。
- 原Exact-KL最大化 `J(q)=E_q[r]−β KL(q∥qref)`，等价于最小化 `β KL(q∥q*)`。
- 新对照最小化 `L_F=β KL(q*∥qθ)/C`。保留前面的β使两个目标采用相同量纲；不能漏乘β而把幅度差异隐去。
- 沿用负加权完整序列logp接口时，detached ascent系数必须为 `c_i=β(q*_i−q_i)`。外层loss为 `−Σ c_i logp_i/(4C)`；`Σc_i=0`抵消canonical logZ导数。
- qθ始终是完整答案及EOS的序列logp之softmax。`/C`只缩放最终loss，不能在softmax前除长度。24候选必须全部存在；没有gold注入、成功重采样、部分奖励或手工奖金。

当前数学源码实现与上述一致。独立24维和非线性共享三参数的有限差分最大误差2.71×10⁻¹²；固定256个目标的系数和最大残差5.07×10⁻¹⁶。

## 2. 初始方向并没有创新，变化是权重及路径

设 `p=qref(gold)`、`t=exp(1/β)`、`Z=1+p(t−1)`。在q=qref时：

`g_forward = [β(t−1)/Z] g_Exact`。

因此每道题在实际共享VLM参数中也共线，而不只是24个可独立logit的玩具模型。256题实际缩放α的最小/中位/最大值为 **0.7484 / 0.8221 / 1.3782**，系数共线关系最大绝对残差3.33×10⁻¹⁶。不同题α不同，所以合并后的batch梯度通常不共线；后续轨迹也可以不同。此变化是所测投影的组成部分，不能宣称初始每题发现新方向，不能只说“学习率完全相同就排除了幅度因素”。

低p题仍有p权重，本方法没有消除稀疏/困难题的全部弱信号。β保持原值且不在dev调节；否则目标与投影两项同时改变，无法归因。

## 3. 它确实可以精确写成蒸馏与监督的混合

唯一正确动作条件下，

`q*=(1−λ)qref+λ onehot_gold`，其中 `λ=p(t−1)/[1+p(t−1)]`。

所以forward loss除去常数后，恰为 **β倍的参考蒸馏CE与hard-label CE的加权和**；权重由原参考概率与β解析确定，不是额外启发式。256题λ的最小/中位/最大值为0.000308 / 0.403681 / 0.457127。最难题的硬标签权重仍很小。该混合恒等式独立误差至多2.22×10⁻¹⁶。

正确定位是“同一可验证reward产生的解析策略改进目标，两种共享参数投影的受控对照”。这是RL推导的固定软目标投影/蒸馏控制，不宜包装为新RL算法；若胜过原RL，也不能自动称其证明RL优于SFT。soft-target拟合与policy gradient的差异必须公开。当前raw SFT优化完整字符串logp，另与闭集hard-label CE存在canonical mass项的差别，报告应保留这一口径。

## 4. 可支持和不可支持的结论

固定256训练参考的非参数q* MAP由75.39%变为80.86%，只是已知训练gold下逐输入不受共享参数约束的理想策略，不是可部署模型，也不是dev/test准确率上界。旧256题online visits处于变化的θ，不能把其平均gap写成固定末点的全256题gap。

如果新方法减少训练gap而dev/orbit不改善，可得“更好拟合理想策略没有转成这一设置下的泛化收益”，不能称方法有效。如果只在一个flow改善，或只在闭集MAP/四选项改善，应完整保留不稳定/解码口径限制。不能读取新确认集来选β、seed、checkpoint或试验预算。

## 5. trainer/协议落盘后必须核验

1. qstar的参考、reward、候选token/EOS、输入hash、base/adapter起点与旧臂一致；qstar固定一次并保存SHA。flow45使用相同256问题重排，不能按标签位置把flow44的参考错配。
2. detach系数但每题重算当前q；24个完整前向及实际非零候选反向计数可核查；无候选置信度阈值、无跳过困难题或零梯度batch。
3. 同256访问、64更新、optimizer/scheduler/clip/LoRA/输入、flow44/45和fixedstep64选点。先证明旧reverse-KL路径在新版本微型验收中与冻结实现一致，再将其作为可比基线。
4. raw SFT只需gold反向，而两种投影通常24候选反向；**同访问不是同算力**。记录总候选前向/反向、墙时与成本，分清成本效率和固定曝光效果。若复用旧基线，准确披露历史运行与缓存来源，而不是声称同次随机试验。
5. CPU有限差分、真实PyTorch共享参数小例和真实模型4题/1update smoke均通过后，另行冻结完整队列预算、绝对截止、锁、失败证据、选点与所有臂展示政策。当前orbit的2h上限不能给这轮训练借用或重置。

后续若想单独区分“投影方向”与“由λ造成的硬题权重变化”，可另立闭集hard-CE/固定权重蒸馏对照，但这会新增实验问题和预算；本轮不能在看分数后临时补对照以挑选有利结论。

## 6. trainer草稿的首轮静态审查

已读新 `projection_train.py`。完整24候选的计算、现有scorer及autograd复用、sum-zero系数的负加权logp反向、四题累积、AdamW/clip、每次更新记录及固定目标签名结构未发现数学错误。曾发现 `Path` 类型的 `args.policy` 直接进入JSON签名会在昂贵模型加载后报错，作者已修正为JSON原生字符串，并用CPU测试覆盖；全部CLI字段也已改为精确匹配锁定policy，包括revision、输入预算与数值容忍度。

本机已实际运行当前 `test_projection_math.py` **12/12通过**，其中包括拒绝draft policy且在任何模型/数据访问之前退出、JSON signature序列化、旧冻结源码SHA及梯度有限差分。仍需将既有reference cache/config/completion SHA与初始identity明确绑定到新执行政策、独立的4题1update smoke模式、真实PyTorch参考以及外层绝对/墙时预算和排它锁；在这些完成前，不能把数学通过等同于完整训练管线通过。本文未授权或执行新训练。

## 7. 更新后的final draft审查与真实Torch待验收

当前trainer SHA `272010e83198084e2e6e8bdb4545f64ebae6d8de0008e19b832324d626f584f0`、math SHA `fa6e354e26ff7dd62fa6ab235abf7e9912d9793cf2d5803d68192dc23e570184`、math tests SHA `2913f03224c761fed5a652fa29fc3dfccfaf239329fd038c20376f78a7c4d95a` 已再次只读审查。

上一节列出的metadata和smoke模式缺项现已修复：manifest及原reference/config/completion的四个绝对路径SHA均为必需输入并逐步复核；初始base/adapter identity与原缓存绑定；全部CLI精确匹配policy；原参考全部CPU验证、qstar冻结、JSON签名预检均在load_processor/load_model之前；formal固定256访问/64步/save16，独立锁smoke固定4访问/1步/save1。三份仍为draft的模板，每份8个源码/4个原工件SHA及initial identity均由本次审查与本地原快照逐项核验通过。模板没有wall预算值，状态会被trainer立即拒绝，不可直接执行。

独立新增 `test_projection_torch_reference.py`（SHA `76d5efa4fac23e6a23d98ecc6672fec7599c23603d74041de52133584e4d9f38`），包含三个真实PyTorch CPU测试：共享非线性参数下直接forward-KL autograd与detached系数梯度一致、实际共享参数的初始forward/Exact/reverse共线、四题累积后强制触发clip分支并比较fresh AdamW参数及全部状态。额外的100倍压力缩放仅用于让CPU参考进入clip分支，不改变正式实验目标或LR。

本机运行总15测试，**12通过、3因本机无Torch明确跳过**；不能写成真实Torch测试通过。远端下一次CPU预检必须15/15通过、0skip、0error/failure，且测试前后CUDA未初始化。此文件未修改trainer绑定的八源码，未来外层coordinator须额外绑定该独立参考文件SHA和实际预检回执。

数学与trainer静态/metadata门槛现可放行给下一层验收；仍不表示真实模型反向或训练管线放行。新批次总预算/绝对截止、训练和评估排它锁、硬停止及无续跑政策由root另立coordinator，不能借用当前orbit预算。4题1update真实模型smoke和独立结果重算、所有正式臂/flow/选点预算预注册完成前不得正式训练。
