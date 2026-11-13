# 新批次草案：RL 是否改变了模型对图片摆放位置的依赖？

**状态：仅本地草稿，尚未 SSH、尚未运行 GPU。** 旧 12 小时批次已结束；本方案是独立新批次，不能靠新目录或重启延长旧预算。

## 研究问题与动机

在同样 256 条训练访问、64 次更新后，Exact+KL RL 与继续 SFT 在原来单种摆放上的准确率接近。这是否掩盖了它们在“同一组照片换个位置，答案仍对应同一真实图片顺序”这一性质上的不同？

我们只改变四张照片的显示位置，食谱文本、真实照片集合和真实排序均不变。把预测标签还原成图片的语义身份后，检查 24 种位置是否给出相同答案，以及这种相同答案是否正确。**一致性可以是稳定地错，必须与正确率一起报告。**

本批只诊断展示位置依赖，不将其写成已证明图文对齐机制。后续照片到食谱步骤的对齐干预需要另立协议。

## 固定实验条件

- 模型：Qwen3-VL-2B-Instruct，以及旧有限策略 v2 的四个既有 **step64** adapter：flow44/45 各 `raw_sft` 与 `exact_kl`。同样的全量 SFT 初始化，随后各 256 个问题访问、64 次更新。本批零更新。
- 选点政策：固定预算 endpoint64，在首个新预测前锁定；不根据新结果选 checkpoint。**这不同于以前 dev-selected 的 flow45 SFTstep48**；旧比较不能与本批混写。flow45 两者旧 step64 均为 71.5%，这个成绩不是选用它们的依据。
- 数据：原 `data/manifests/val.jsonl`，SHA `c998796fbc07a24c7a24062d6d0aba07500c75b34568fd7ba67f8c38daae03c9`，**按原文件顺序的前 16 题，16 个不同食谱**；禁止按成绩或图片难度筛选。
- 每题完整 24 个 `display_perm`，词典序；每模型 384 条正式预测，四模型共 1536 条。另有每模型固定前 4 条的独立计时 pilot，共 16 条，pilot 不当作额外统计样本。
- 文本/视觉预处理与原 `predict.py` 完全一致：原 answer-only prompt；greedy，32 新 token；min/max pixels 65536/262144，recipe budget1536，input budget4096。评价种子分别沿用 flow44/45；greedy 无采样。
- 不打开旧 test 或新 confirmation200，不改 reward，不训练，不做候选约束、投票或最佳输出筛选。

## CPU 坐标构造与独立校验

`display_perm[p]` 表示显示位置 p 的真实图片身份。构造前从原实际 `image_paths_display` 与原 permutation 还原 semantic→path，交叉核对已有 chronological paths；再按新 permutation 同步重排图片路径、状态、`presentation_id`、`target_labels`。保留 `qid`、recipe ID、食谱内容、真实 `target_order`。Gold 标签通过逆 permutation 得到，并立即正向还原检查。代码支持非 identity gold，避免“所有答案默认 ABCD”的隐藏错误。

runner 调用已冻结 `scripts/predict.py`，随后调用已冻结 `scripts/validate_run.py --require-orbit` 核对每模型 16×24 无遗漏、重复或身份变化。再由新脚本从完整 raw response 独立解析和计算六对关系，不依赖 predictor 的保存分数；两种 scorer 不一致立即失败。技术失败使 coverage 失败，不能充当模型无效答案或静默删除。

## 指标和展示

每食谱报告：K（24 种显示中正确次数）、K/24、是否 24 次全对、semantic modal consistency（最常见有效语义排序次数/24）、六对排序关系正确率、有效回答率。汇总对这 16 个食谱等权；给 K=0…24 的完整分布及同 flow 的 RL−SFT 逐食谱差值。1536 次显示不是 1536 个独立样本；本批不提供以显示为单位的显著性检验。

四个模型结果全部展示，无论正负。只有两种优化 flow，且共享 SFT 初始化；不能据此声称稳定 RL 提升、机制因果或总体结论。是否值得更大样本/新干预应在看完这批后新冻结，而不是继续在这一批上选题调分。

## 资源、计时与停机

新批次最多 **7200 秒，约 3.2 元**（按 1.6 元/小时，不含已发生租卡空闲费用），另设 launch 前锁定的绝对 UTC 截止，以更早者为准。开始后输出目录必须新建且不能存在；获得原训练、评估、有限策略及本批排它锁；GPU 必须无计算进程、低占用且温度正常。

先完成四个固定 pilot。记录输出的单调时间；相邻输出间隔包含下一条的 CPU 图像处理/编码、GPU 生成、写盘，因此成本不只看 `generation_seconds`。整体 gate 用最慢 pilot 样本间隔外推 1536 条，加四次首条加载成本、1.5 倍余量和 300 秒 CPU/IO。若不够，正式四臂均不启动；若足够，则完整执行既定四臂，不按分数停臂。Pilot 不是尾部耗时保证，独立硬截止仍生效。

任何技术故障/超时均保留已生成日志、raw 和 state，终止所拥有的进程组；不能通过重启、接着生成余下记录、缩短样本或只保留部分模型，把失败批次改成“完整”批次。修复必须用新版本/新协议并解释故障和预算。所有模型、环境、图像、缓存、临时文件和输出只留远端 `/root/autodl-tmp/rl_recipeqa`。

前后散列核验磁盘上的 base/adapter 权重及辅助配置完全相同；已冻结 predictor 使用 `trainable=False` 和 `torch.inference_mode()`，无 optimizer。明确这是磁盘权重指纹与代码路径证据，不额外宣称做了内存参数 checksum。

## 工件与冻结前门槛

预计工件：冻结协议/源码 SHA、GPU 健康记录、输入 cohort/gold mapping receipt、384 行 manifest、16 行 pilot raw 与四臂 1536 行正式 raw、原 predictor config、全部 subprocess 日志与时间/返回码、四个覆盖检查、独立重算逐行分数、逐食谱统计、summary、模型前后 identity 和完整文件 SHA receipt。

当前本地 9 个 CPU 测试通过（覆盖 24×24 映射/评分、非 identity gold、图片身份反转、缺失/重复/技术失败拒绝、稳定地错与正确率区分、四模型计时 gate）。运行前仍需 root 独立评审、在远端只用 CPU 构造真实前 16 题并检查文件、设新绝对截止、刷新冻结 SHA、公布新批预算后将 JSON 状态改为 `locked_before_first_orbit_prediction`。**本 agent 未作任何远端连接或执行。**

终止审查补充：最后 coverage 后、最终 fingerprint 前后及标记 completed 前均检查剩余预算；SIGTERM/SIGHUP 转为异常，进入进程组清理与锁释放。SIGKILL 无法由 Python 捕获，不能声称这种强杀情形也保证无孤儿 GPU 进程；应由后台观察器再核验。

硬截止补充：SIGALRM 真实 wall timer 覆盖 CPU preflight、哈希及汇总，取新7200秒wall与冻结absolute较早值。截止后仅执行 owned child 的 TERM/KILL 清理（最多8秒）与日志线程收尾（最多2秒），不启动新研究任务；finally取消闹钟并恢复旧handler。
