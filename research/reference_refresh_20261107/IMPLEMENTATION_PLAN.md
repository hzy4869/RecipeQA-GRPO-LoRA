# Reference refresh：本地实现方案与风险（按已落盘root PLAN实现）

本文件只规划新namespace实现，不是冻结协议、不授权GPU。根代理拟定研究：两flow×SFT、固定初始β、固定初始β/2、两轮每轮β刷新，共8臂；同SFT768、同256题两pass/512访问/128更新、accum4、LR5e-6、C5、clip1，固定128为主终点。最终科学定义以root `PLAN.md` / `ANALYSIS_PLAN.md` 为准；有不一致先修本实现草案，不改已冻结旧实验。独立20h/32元已公告，仍未first/freeze/远端执行。

## 实现最小闭环

1. **显式派生与闭包。** 新 `refresh_train.py` 复制旧coverage trainer；保留24 EOS动作、条件q及精确系数、逐候选autograd、CE归一化、clip/AdamW更新数学。新contract明确允许256唯一输入×2pass，不能沿用旧1024唯一访问限制或猴补旧globals。记录原SHA、AST相同函数、训练loop必要差异与完整lineage。新coordinator/CPU/launch/builder明确新namespace、8臂与20h第一时钟；旧源逐SHA绑定且不修改。
2. **两层身份。** 256个唯一 `(qid,presentation_id)` 是输入身份，512个 `(pass,within_pass_index)` 是访问身份。每flow先冻结256记录顺序，第二pass必须逐字段复用同一顺序与展示；flow45在同256集合一次固定shuffle，不从新1024 shuffle前缀取题。独立核原full7558前256、254食谱和SFT768实际已消费证据；不把重复访问当新数据或512独立问题。
3. **q0缓存只映射。** 使用已完成、同SFT768/版本/候选/像素/正文输入的旧reference。按完整输入身份重排/取subset并证明源行全部数值字段不变；保留source index0 literal-check起源。训练两个pass查同一256缓存行，不伪造512条新评分，不改候选次序或visit metadata偷换输入。
4. **刷新是真实中间模型。** refresh臂在预声明64更新完成、checkpoint64及其audit写完后，仍在同进程同model上对全部固定256输入重新24 EOS评分；签checkpoint64物理identity、adapter/visual指纹、输入/hash/pixel/versions、原run signature和逐行token logp。零optimizer/backward/采样生成，成本另列。全部缓存及独立可核数值/来源齐全后仅把第二pass的stopgrad reference换为新q；固定β与β/2和SFT不换。
5. **不中断优化状态。** 同一AdamW/scheduler/RNG继续从更新64到128，不重新load_model、不新建optimizer、不重设seed或恢复checkpoint。刷新前后独立保存optimizer/scheduler/RNG内容指纹、各参数step计数、adapter/vision指纹；断言无变化。若只保存旧checkpoint文件SHA无法证明内存状态未重置，需新增实际内存状态摘要和足以独立复核的小张量/安全序列化审计。后续最终128 optimizer计数必须连续。技术失败整臂保留，不做自动续训或借版本重置时钟。
6. **端点评分与主任务分离。** 8臂各最终dev200开放greedy与同16×24 orbit。固定train256的canonical评分可在同一trainer完成128保存后、仍加载最终model时只读执行，另存raw24×5 token logp、checkpoint/input/q0来源及前后不变证明，节省独立load。独立验收应把这256行当最终统一θ评测，不能用512在线轨迹代替。dev不另canonical评分。所有目标均从同q0并分别用β与β/2重算，刷新臂原生在线J另列。
7. **完整门。** 本机只做CPU/合成。远端新first-preflight起算后真实零skip CPU、native孙进程watchdog、8个普通smoke+2个真实刷新path smoke及各自独立有效更新验收、全8臂训练+末点评分+dev/orbit+刷新开销/存储门齐，才正式开始。少臂或中途好点不构成阶段完成。外部GNUtimeout同PGID、不detach；所有目录/cache/tmp只数据盘。

## 产物/责任接口

| 产物 | 准备内容 | 责任建议 |
|---|---|---|
| PLAN/ANALYSIS_PLAN/预算时钟与最终判据 | 主RQ、两pass顺序、smoke刷新边界、primary128、误差/CI与是否复验 | root锁定；本工具不代写冻结结论 |
| refresh_contract.py、refresh_train.py、lineage | 数据/缓存/4方法、更新边界、真实刷新与连续优化状态、最终train bank | 本agent派生实现 |
| builder/coordinator/launch/CPU及测试 | 8臂完整队列、锁、cache封存、授权门、计时/存储/进程组 | 本agent；尽量复用已审结构，声明所有新改动 |
| 独立run/reference/result verifier与测试 | 不导入producer数学，核两个reference阶段、旧cache同源、continuity、128主端点 | 建议root另委agent尽早并行，先固定schema |
| 最终分析 | 8×584=4672 greedy/orbit；8×256=2048最终train canonical；完整成本与以食谱聚类的对照 | 独立实现；先完整门再读结果，不按分数选点 |

## 冻结前必须解决的风险

- **解析终态相同不等于神经训练相同。** 只有每轮每题自由分布精确优化时，两轮β刷新理想q*等于固定初始β/2；刷新对fixedβ的收益可能只是更弱累计KL。主新方法差应为refresh−fixedβ/2，并保留fixedβ与SFT，不以已知数学等式冒充新算法。
- **中点优化尚未充分兑现。** 已完成bank显示原train可用J约49%兑现，不能宣称已达固定KL目标；新的刷新可能带来路径/有限优化效应。统一初始目标下的最后q、gap与绝对greedy是诊断，不能由在线原生目标不同直接比大小。
- **两pass与缓存索引。** 旧validate_budget/reference实现都要求唯一prefix；直接扩limit512会失败或错配。必须分开唯一记录与访问流，pass2保持同一展示。参考刷新不能拿pass1在线q拼接成末点qref。
- **RNG与optimizer证明。** 刷新评分原则上deterministic eval/no dropout/no采样，但仍实查状态；不通过手工重设RNG伪装连续性。如何序列化连续性证据需在独立schema里先定。
- **smoke不能漏真实刷新逻辑。** 4题1步只测普通更新，覆盖不到64边界；正式8smoke须用预定义缩小round边界并实际经过checkpoint→重评全部smoke输入→继续更新，或另有最小refresh两轮smoke。其余臂曝光一致，成本全记。具体由root PLAN先定，不在运行后补一个容易通过的门。
- **闭包与历史缓存数值身份。** 新源SHA改变不意味着旧cache重新推理；必须区分只承载数学相同旧score helper的scoring closure与本次trainer closure。只允许明确字段更改并封lineage，不能宽泛忽略全部source/input字段。
- **资源预估易漏二轮刷新与最终bank。** 正式8臂16 checkpoint、8smoke实际checkpoint数、2个flow各refresh256×25前向、2048×25末点前向、4672开放greedy、装载/物理核验/CPU独立验证与安全余量全部计入20h/存储门。预估约12h不是保证；全门不够则正式全部不启动，不能静默删臂。
- **统计与叙事。** train256含254食谱，dev200为200食谱，orbit16食谱；两flow共享起点/题池。不把512访问或两flow当独立扩样。主对照一组预先声明，其他包括SFT、fixedβ和不同目标J为次要；完整公布8臂及负结果。
- **时间/授权边界。** 20h/32元目前是新批已公告设计，仍须root冻结first+absolute后才能远端preflight。原bank3h/coverage16h不借用、不重置。若新目标未达，不能靠不断扫β或开confirmation调参完成长期目标。

## 当前状态

只读旧coverage producer/contracts/verifier和已完成endpoint独立报告。0 SSH/GPU/模型调用/训练，无新first clock，没有改旧冻结源。新contract/trainer/编排及独立verifier已经本地准备；尚待完整来源审阅、真实draft与远端实际门。没有另加loss或额外模型。

## 实际实现补充

新trainer不生成冗余adapter_final，最终identity直接最后固定checkpoint。普通8smoke+两path-smoke共12checkpoint；正式16checkpoint。连续快照包含梯度，因此formal与path合8文件约1.68GB，不因4题而减少参数状态大小。全部入存储门，不能删旧失败工件。

旧coverage本来已用恒等LambdaLR，源 `coverage_train.py:366` 构造、`:468`每步调用；原signed optimizer.schedule='constant'。新trainer沿用同一状态机；额外392step/last_epoch只作连续性审计，不引入调度。

主实现接口已由CONTRACT_DRAFT.md收口；全阶段74命令，前73科学/技术命令完成后，最后独立stage-only完整验收也在同一20小时native watchdog内执行。第74成功且parentSHA绑定后才提交independently_verified终态；不能以pending全局审核充当完成。当前本机stdlineage与保护测试不等于真实模型smoke或formal准入。
