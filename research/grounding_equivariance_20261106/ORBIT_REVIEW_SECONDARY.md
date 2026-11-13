# Orbit 诊断第二独立审阅

状态：最终修正后代码、九项 CPU 测试及草案协议已审，无剩余代码或科学设计阻断项；最终冻结 SHA、远端 CPU 预检及新绝对截止仍由 root 完成。只读审阅，未改 runner，未启动 GPU。

执行 `python3 -m unittest discover -s research/grounding_equivariance_20261106 -p test_orbit_diagnostic.py`：6/6 通过。覆盖非 identity target、完整 24 展示、图像身份往返、原 gold 错误拒绝、16 固定 recipe、格式解析及全部 576 个坐标/输出组合、完整覆盖、技术失败拒绝和四模型 pilot 时间估计。

## 科学设计

四模型固定 step64 端点，避免把之前 flow45 SFT 按 dev 选择的 step48 和 RL step64 混为公平终点。固定 dev200 文件前 16 recipe，不按成绩选样；每题全部 24 布局，同步更新显示图片路径/状态、display_perm 及逆映射 gold，文本上下文不变。每臂 384 条相同输入与相同解码；pilot 每臂额外 4 条相同输入，不用于方法选点。24 布局嵌套于 16 recipe，分母正确；consistency 可包含一致错误这一限制已写明。没有新奖励或训练。

## Launch 前需要修正

1. 2 小时及绝对 UTC deadline 由子进程 timeout 与 remaining 双闸控制，但最后 coverage 完成后，汇总、最终模型权重 fingerprint 和 artifact SHA 阶段不再检查 remaining。建议在最终 fingerprint 前后、最终 complete 状态前调用 remaining；否则可能在 deadline 后仍标 complete。超时部分证据应保留并标 budget stop。

2. 子进程使用 start_new_session=True；父进程目前没有 SIGTERM/SIGHUP 处理。普通终止可能绕开 Python 的 except/finally，留下 GPU 子进程。建议信号转为显式异常，进入 stop_child 及锁释放路径。SIGKILL 无法由 Python 捕获，应如实说明；若宣称父崩溃也有严格截止，需另设独立 watchdog。

## 运行前条件

协议落盘后仍需审查四端点权重身份与原训练 run 的绑定、source SHA、全新输出目录、UTC 绝对截止、2 小时及预算约 3.2 元上限的事先披露。runner 缓存检查对 XDG/PIP/TRITON/CUDA 部分变量允许缺省；应确认远端 env.sh 实际全部指向数据盘，避免缺省 cache 落到系统盘。

没有发现图像顺序/答案逆映射错误、gold 进入决策、按低分停臂、按次级成绩重选点或把 384 布局当独立 384 recipe 的错误。16 recipe 是 post-hoc 开发子集，该实验仅作展示敏感性诊断，不构成总体 RL 收益确认，也不能直接归因为图文对齐机制。

## 修正后复核

上述两项已由 runner 作者修正：在最终 immutable、模型 fingerprint、summary 及 receipts 阶段补充 remaining 检查，完成状态前再次检查；SIGTERM/SIGHUP 转为 InterruptedError 进入子进程清理，并在清理期间忽略第二次普通终止，最后还原原信号处理器。独立复跑新增 deadline 和信号检查后的 8 项 CPU 测试，全数通过。SIGKILL 无法捕获的边界已在代码中明确。

草案协议四个 prior_endpoint_config 文件的 SHA、adapter 身份和共同 base 身份均已与补取后的真实快照独立核验一致，包含 flow45 SFT 的固定 step64，未误用旧 selected step48。草案 status 和 absolute_deadline 尚未冻结，且修正后源码 SHA 需要刷新；这属于启动门槛，不能当作已经启动。此前两个必要修正已关闭。


## Final CPU budget protection review

SIGALRM real-wall timer now covers preflight, CPU hashing and final summaries in addition to child wait timeouts. Both inner subprocess and outer exception handlers cancel the timer before bounded child teardown; finally restores the alarm and normal termination handlers. The real 0.01-second timeout test passed. All 9 CPU tests passed in an independent rerun. Runner/test SHA match the current draft protocol. SIGKILL remains outside Python signal handling; cleanup after expiry is allowed only for bounded teardown and evidence persistence.

Final reviewed SHA256:

- orbit_diagnostic.py: `d732f89f6b9cf386b2796942b45c0f792a3208cd2b88281148e76ef029d5055a`
- test_orbit_diagnostic.py: `cb1691c23b22463d6983b032881c447f09e3b18abfd3c026a0c7205520354532`

Result: no remaining code/scientific-design blockers. Root must still freeze a new absolute UTC deadline, change policy from draft to locked, and complete the remote CPU/cache/cohort gate before GPU launch.
