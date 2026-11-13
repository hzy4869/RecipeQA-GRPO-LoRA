# Coverage代码与合同独立审查（最终静态版）

**结论：PASS——当前代码/合同可以进入既定远端验收阶段；没有发现新的算法或集成阻断。** 这不是正式训练或真实GPU验收已通过的声明。审查仅用本地CPU，无SSH、GPU、模型推理、冻结或test/确认集访问；只修改本审查及其回执。最初草稿的待修项已由作者落实，历史版本保留在Git中。

## 绑定的实际证据

`CODE_REVIEW_CHECKS.json`记录全部**20个生产/旧依赖源码SHA**、8函数与整段训练循环AST、完整独立stdlib测试stdout/stderr以及两份研究/统计计划SHA。生产方与独立verifier的source集合精确一致；测试前后20文件无漂移。关键文件如下：

| 文件 | 最终审查SHA256 |
|---|---|
| `coverage_train.py` | `10a9cb4830ade62f6e2d91ecde21ff5a282f15f04ee2649cc99ebf6d329bb06c` |
| `coverage_contract.py` | `02f08db4d108718428ad34f315f3bd45ffecaf56b6aac842c9031a52dd6d0f3b` |
| `coverage_pipeline.py` | `d51833b6f7a4a9c7b157efd9d0313a5f4184a4d6d9b0abaa59a37b7f1e7d24b3` |
| `verify_coverage_run.py` | `4d5499f371f609903768a8ae20a178e48222b700a748fb34dc475adbdd1bfbe7` |
| `launch_stage.py` | `a90f5bd3a3dacfe1255a36dff3cc9e7fd9d41ad31a716c3cc288c4e12adcaf47` |

旧trainer仍为`34e29cc48744626dec27f99afdd5bdf34660130cf45f17e6d3ff205786a446a5`，匹配已下载manifest；旧独立verifier为`c2408abd93fb9b322262213a65b9d311abe88ef4bc7550aec1d164626f73a791`。新旧`log_policy`、`categorical_draws`、`policy_coefficients`、`append`、`pixel_identities`、`cpu_gradient_norm`、`prepare`、`save_checkpoint`八函数AST完全相同；整个实际训练for-loop AST也完全相同，SHA为`7f2805ddb0fc9c7855d9ebb93a4e98e6be853f95b06e4c12d39f33ec7de5cc81`。

我独立运行了14个生产合同/队列测试及13个独立verifier测试，**27/27通过、0失败/错误/跳过**，完整回执中保留实际时长。没有运行本机缺少的3个真实Torch测试或GNUtimeout原生进程树probe，也没有把这些未执行项记为通过。

## 原草稿待修项如何关闭

1. **科学实现与source closure。** 新文件是显式派生副本，不改旧代码、不monkeypatch。4/1024、save64、共享剩余时钟、固定C5及政策签名是新合同；训练数学和实际更新循环保持原样。20源已包含`orbit_diagnostic.py`、`predict.py`、`validate_run.py`，不会遗漏输入生成/重评分/评测/时钟的直接依赖。PLAN/ANALYSIS_PLAN通过builder进入stage与每个child的artifacts，不只口头称冻结。
2. **flow/历史曝光。** 合同固定flow→seed/manifest、原模型revision与SFT768起点；pipeline核parent/child共同首次preflight及absolute deadline。旧真实visits的step0..767各8条、backward-returned消费标记和原SFT run signature进入生产及独立审计。另已CPU核1024题/963食谱全部在6144次初始已见曝光中；flow45新前256与旧flow45仅82题重合，分析计划明确禁止误当同前缀复现。
3. **reference两层冻结。** 先冻结child CLI/路径、构造函数、源码、数据与顺序；真实1024 qref只评分一次。三种确定性重排/子集连同source共15个cache/sidecar/lineage文件在首smoke前封SHA，回执自身SHA也进入parent immutable检查；每child实际run signature再绑定reference/lineage/封存回执SHA。无需改写已冻结childpolicy。重排只改行index和refspec SHA，其余scores/token IDs/input audit等不变；spec只改manifest/selected-record/pixel集合。独立verifier不调用producer重排函数，重建所有来源与许可字段。
4. **独立验收。** 新verifier显式接受4/1和1024/256/save64，拒绝旧256/save16政策；数学来自旧独立推导，不导入新trainer/contract。reference literal检查按原source index0，当前训练检查按当前visit0。reference-only真实1024/零optimizer路径独立验收，各训练审计也重新检查共同source；shared25600参考前向单列，不重复加到每臂成本。所有实际checkpoint有safetensor指纹和optimizer/scheduler/RNG ZIP CRC审计；无有效更新作为已披露负结果可以通过。
5. **资源与deadline。** 16正式+4smoke checkpoint、8份最终adapter、metadata和余量由旧实际字节估容量；每64保存，128/192不评测。四smoke各自独立验证后，才估四个256步训练、3136预测/12次加载/保存和独立验证的完整时间门，拒绝整批缩臂/缩样。外层GNUtimeout管理同一进程组，科学子命令不detach；原生CPU probe确实创建TERM-ignoring子孙并检验KILL后的停止状态。`/proc`已回收竞态有明确处理。真实probe、容量与计时结果尚须远端产生，源码静态审查不代替它们。

## 无额外代码阻断，但以下真实门仍必须通过

- root在实际数据盘核源/工件SHA并冻结统一首次preflight、16小时和UTC绝对截止；环境/缓存/输出均受数据盘检查。后续技术修复沿用原时钟，不能借版本重置。
- 远端**精确30项CPU、0跳过**，包括3个真实Torch autograd测试，且CUDA未初始化；另行记录真实GNUtimeout子孙清理probe。Mac的27项不是30项。
- 真实共享reference1024独立审计、四个4题/1更新GPU smoke及其实际二进制验收、完整容量和全四臂剩余时间门。任一失败保留证据，不按分数停臂或改科学内容。

## 分析约束保留

fixed256是唯一主终点，fixed64仅诊断，无最佳步选择；两flow共享起点/dev，16食谱不是384独立样本。相同曝光不是相同候选算力，1024已见题不是新数据，增加覆盖与更新同时发生。平均约3pp且两flow严格正只触发下一批预定≥3flow复验，不自动开确认集或证明胜出。分析计划还绑定了共同初始模型143/200=71.5%的已知原始证据，必须区分绝对提升与比continued SFT更少退化。

训练J可因较小KL而升高，一致性可以稳定地错；两者不替代greedy Exact。未见优势不证明真实差为零。旧test已知、dev反复使用、人工歧义审核pending均须披露；新200仍封存，无同口径官方比较不称SOTA。本次推荐没有把无效grounding B测量当视觉瓶颈的证据。
