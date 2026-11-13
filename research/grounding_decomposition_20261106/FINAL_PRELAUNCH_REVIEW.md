# 配对 grounding 分解：最终启动前独立审阅

**决定：代码与评分逻辑通过，可按已公告的30分钟／约0.8元上限进入远端部署、冻结和受门控执行。没有发现需要阻止这一有界诊断的关键错误。** 此决定只适用于下方源码字节；本审阅没有 SSH、GPU、模型/processor调用，没有冻结政策或启动实验，也不代表远端输入预检已通过。

## 实际核验

独立执行当前 `test_decomposition.py`：**25/25通过，0跳过、0失败、0错误，0.040秒**。新版CPU gate绑定预期25项、唯一实际计数和最终终态，拒绝空收集、24/26项、重复计数、跳过、失败/错误、缺终态或非零退出；不再把退出码0独立当作测试成功。runner将结构化计数和原stdout/stderr、suite SHA写入回执。源码使用stdlib及既有开发metadata；假tokenizer测试没有冒充真实Qwen处理器验收。

另独立以零起点、有缺口的source IDs `[0,3,9,17,24,40]` 和全部24种照片展示排列核验可见编号、B导出排序、最终顺序以及C/Cπ答案，全部往返一致。未读取test或confirmation来选择条件。

## 启动范围与正确性

- 固定既有开发200题前32食谱，SFT44与Exact-KL44均为预定step64。两模型各新跑A/B/C/Cπ各32次，共256条；新A为主配对对照，历史A仅复现与身份控制。没有按新分数选点、挑错题或添加模型。
- A/B复用原A均匀正文cap、全部步骤标题与四图；真实CPU预检必须重现历史prompt/input hash、长度及grid，比较当前A/B的可见recipe UTF8字节和实际pixel/grid张量。每次多模态推理还重查pixel/grid摘要。任何超限均失败，不静默另裁正文或换图。旧历史pixel字节不能追认相等，故保留新A。
- B绝对来源匹配、六对相对先后、严格可导出顺序、最终Order、自洽性及格式有效性分别计分；unclear/missing/out-of-range留在固定分母，tie不按标签打破。最终Order不因数字字段格式失败而自动丢弃；格式率另报。来源步骤是数据归属标签，尚不是人审确认的唯一视觉语义。
- C/Cπ为独立、无图片的权威数字表输入。Cπ固定循环 `(1,2,3,0)` 后独立计算自己的金标，既不共享C的会话，也不拿原图排序判分。C高、B低不能唯一归因于视觉grounding；它们改变了上下文、信息和输出要求。
- pilot恰为前2食谱×4条件×2模型的16个规范观察，计入256且不重跑。整批gate使用8个模型/条件cell的较大实测时长、两次剩余完整模型重载、1.5倍安全系数和120秒余量，估计剩余240条。gate失败即保留不完整阶段，不删条件。汇总拒绝缺失、重复或技术错误，不能用部分结果报整体。
- 9份科学/辅助/test/launch源码均绑定SHA；已知开发manifest、两份历史预测及config、固定128张照片、实际模型及adapter字节均核验。排它锁覆盖旧训练/评测、projection和本诊断，GPU不空闲则拒绝。环境、缓存、图像路径和输出均要求数据盘，输出目录必须全新；不支持resume、自动retry、训练或改奖励。
- 外层 `launch_stage.py` 使用GNU `timeout --signal=TERM --kill-after=10s` 监督runner及CPU子进程组，弥补本进程SIGALRM可能等待原生CUDA调用的缺口。剩余秒数来自第一次preflight与绝对UTC截止的较早者，不能重置为新1800秒；至多10秒后强制结束，只允许清理。正式run要求政策SHA匹配的watchdog标记，须经该launcher启动。

## 部署时仍必须真实通过的门槛

root须逐项核对本表与远端源码SHA、实际模型/图像字节及冻结政策，留存GNU timeout版本/launch回执；记录唯一first-preflight及不晚于其后1800秒的绝对截止。**真实远端25项CPU、32题processor/recipe/pixel/EOS/token预算预检、16条pilot和全批计时gate均未被本地审阅替代。** 失败保留版本和证据，修复不重置本轮时钟。

本诊断不触碰新确认集、不更新权重，也不能产出“RL已提分”“定位了隐藏CoT模块”或SOTA结论。32个已知开发食谱、单个优化flow、29/3正文裁剪分层与未完成人审都必须随结果披露。

## 审阅源码SHA256

后续任何字节变化均须明确复核，不能沿用此表声称已审阅。

```json
{
  "research/grounding_decomposition_20261106/run_decomposition.py": "c94630506a0dd064b02245ecd294a2f8c053446a83d0bbd7ee8a0282df6138e1",
  "research/grounding_decomposition_20261106/decomposition.py": "ff7c83277c73c3d157e597018683ebdd326ef0d6545a5de739e6eb5567b0d69a",
  "research/grounding_decomposition_20261106/build_draft_policy.py": "f43bd14e6151b6672bc66b6cd6d02c55cc4e35f4c6382cb317c48189c99d37f4",
  "research/grounding_decomposition_20261106/launch_stage.py": "5f65ae237752fb5f9e6a422ebeae9a84cd5936e576fa751ca93d8bd745cc76f5",
  "research/grounding_decomposition_20261106/test_decomposition.py": "a9416051ae5aef5d40131de40d39ffae5c2dbcab1ec33694d30ac60d7187bd71",
  "research/grounding_equivariance_20261106/orbit_diagnostic.py": "d732f89f6b9cf386b2796942b45c0f792a3208cd2b88281148e76ef029d5055a",
  "src/recipeqa/modeling.py": "5fedb7a60877e965861aa9174179880754cb0555c4167265fdd8c3781af39771",
  "src/recipeqa/metrics.py": "5287d44a3bb4ede3a44ce0a6625aee399260afecd42c984441a8e07492a2423a",
  "src/recipeqa/process_metrics.py": "ebc9e551dd489c6c14d979755029ea79dcf5d88238f5c20076742dda73ea4b0b"
}
```
