# Projection新批次协调器独立评审

状态：**静态代码/数学/metadata schema审查通过，最终冻结回执及远端CPU→真实smoke验收仍待完成**。只读本地审阅 `projection_pipeline.py`、`run_cpu_preflight.py`、`test_projection_pipeline.py` 及待冻结stage政策，不SSH、不GPU、不修改作者源码。4小时/约6.4元为root报告的新批预算，不借用旧orbit或旧12小时上限。

当前设计口径：新增两个flow的FWD固定64步，四个旧SFT/EK fixed64是明确披露的历史对照；dev200和首16×24 orbit是已知开发数据；不使用新确认集，不按分数改选点或停臂。

初读队列实际计划命令数为 **15**：CPU preflight 1；smoke train+verify 2；两个正式flow各train+verify、dev predict+coverage、orbit predict+coverage，共12。12只是两个正式flow的命令数，不是整批总数。

## 已核验的核心路径

1. `validate_stage_policy` 要求14400秒、固定两个旧flow、fixed64、已知dev200及其首16×24 orbit；绝对截止必须来自 `first_preflight_started_utc`，长度≤14400且不能未来起算。实际remaining取绝对截止与本进程墙时较早者，不能在coordinator启动时重置早前preflight时钟。
2. 调用已冻结orbit的真实SIGALRM，覆盖CPU散列和汇总阶段；正常TERM/HUP异常化进入owned进程组清理。异常时先撤alarm，再TERM/KILL和有界wait，保存失败状态；无恢复/自动重试。八个非阻塞排它锁及GPU空闲/温度/内存gate在首个新训练前取得。
3. CPU preflight使用 `CUDA_VISIBLE_DEVICES=''`，实际discover全部 `test*.py`，固定计数且0skip/0failure/error/CUDA未初始化才过。当前源码完整发现 **29项**：本机26通过、3项因无Torch明确skip；远端必须29/29且0skip，不能把本机结果写成通过远端门槛。
4. 三个child政策在第一更新前全部独立锁定、SHA绑定，smoke4/1输出与两个formal256/64输出分开，初始化身份均为原SFT768。正式模型不从smoke adapter暖启动。manifest/ref/config/completion均绑定；每阶段重查policy/source/artifact/child SHA。
5. 历史四控制的signature、原起点、同manifest/hash、seed、256访问/64更新、LR、qref和原目标β与对应新flow对齐；全部旧dev/orbit原始响应再从display坐标独立计分。明确它们是已知开发数据上的历史控制，不冒充同时随机试验。
6. gate先完成真实4题1步smoke及forward专用独立verifier，再用完整四个历史orbit的384条arrival时间估计两正式训练和1168条预测：`1.5*(128*smokestep+10*非step开销+1168*最大arrival间隔+4*首条加载)+600`。包含CPU预处理/写盘、两次训练加载及八次checkpoint余量、四次预测加载、独立验证余量；仅资源依赖，没有分数。未通过则两正式臂都不开始；硬截止独立于估计，不能用pilot声称未来耗时保证。
7. 每正式flow固定64 checkpoint→200贪婪→384orbit，并做coverage和raw重评分；没有dev选点、重采样或低分停臂。比较给完整两个flow与四历史控制，nested layouts以16个recipe为单位，不把384条当独立样本。

最终检查stage的 **18个source SHA、39个历史/参考/data工件SHA及3个child政策SHA全部匹配**，无缺项。评审并行期间出现过陈旧回执，作者现已刷新；最终版实际逐项复核通过，不沿用旧哈希。

最终审阅源码SHA：pipeline `358a056890a445d8c0ba2a986356345721137b2af787a8880fc29e5fd1e8f2d8`；CPU preflight `328d9b482c984fedb1b1e0740e0cdda4faf41111fa9d2204fae06d41a69080c4`；pipeline tests `f03a5507d55de839519a25d9c43aa1903b73d38972df1cdc1a667d4f0b1f7bd1`。后续变化须复核。

末次小改也通过：整批计时外推前强制真实smoke确实完成96次候选反向与100次当前scoring前向（4题×24反向、4题×25含检查的scoring前向），避免用稀疏低成本smoke低估完整formal；四个历史timing标签集合必须恰为四个不同控制，不能用重复arm冒充完整基准。对应duplicate拒绝分支加入既有测试，计数仍为29。该门槛只检查实际成本覆盖，不读取分数，也不允许在不满足时换题重跑。

未发现阻断性的算法/接口/计时逻辑错误。放行范围仅到最终冻结及远端验收层：root仍须锁定首preflight起点与绝对截止、有限child wall预算、当前源码/测试/child政策SHA，再得到真实CPU29/0skip和4题1步模型smoke独立验证通过，计时gate通过后才能启动两正式臂。旧test已知、新确认集封存、共享SFT起点、soft-target监督投影归因和历史控制限制必须保留。
