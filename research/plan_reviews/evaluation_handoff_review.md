# 固定评测与训练后交接复审

2026-11-05；此记录区分已完成的代码/CPU检查与尚未执行的正式评测。原训练、预测、统计源码及原协议锁未修改。新增评测源码交根代理审阅后停止编辑。

## 身份、覆盖与恢复

- `evaluation_pipeline.py` 在 P6 `completed / three_branches_selected` 前直接退出2，不等待、训练或调用GPU。完成后核原协议SHA、全部原冻结文件、Base/W实际权重与辅助内容、三主选点与各`best_checkpoint.json`、训练工件验收及选中adapter实际内容；可选4pass只进入独立补充E0表。
- 新评测wrapper、greedy比较、review-view生成器、训练动态图脚本四文件的SHA写入selection snapshot；snapshot连同模型、manifest、review派生来源与线程设置绑定恢复。新源码不是对原训练源锁的覆盖。
- E0每方法905题/597食谱；E3每主方法60食谱×24布局；E4每主方法40食谱。调用现有validator逐项声明run/checkpoint/seed/prompt-kind/样本数与完整orbit，技术失败不冒充invalid。配对统计要求全覆盖一致。
- 每个成功stage只有在原命令、声明artifact键集合及所有artifact内容SHA全部匹配时跳过；缺receipt、文件变化、selection变化或旧GPU子进程仍活跃均停止。失败不自动重试；人工查明后才使用`--resume`。

## E3真实重复与E4可见文本

- E3前固定monitor前10题，三主checkpoint各两个独立预测文件，共60次greedy。每个checkpoint的两个文件使用同run_id；逐题比较原响应、含parsed-order的完整score、实际`input_audit`、解码与生成长度信息，配置只允许output路径不同，实际权重身份必须一致。不一致写报告并暂停E3，不自改backend。该控制不证明全部输入或训练过程逐位确定。
- E0与E4统一`eval_<name>`、model_id、checkpoint及seed，E4 join另要求40条answer-only齐全；metrics代理独立核对了这一identity接口。
- 预测/coverage始终用原process40和原校准5条控制manifest。只有human-summary改用模型实际可见正文派生view。逐record除`steps[*].body`与新增review元信息外全字段相等，固定正文cap有roundtrip/输入上界证明；正式process预测完成后还将用真实`input_audit.prompt_text_hash`和cap再次核验盲审文字。
- 主40派生SHA：`89f8397c463513bc6cf2ff37096ecae01ecd3f9d30eb473f334a2b0e5ff18691`。校准5派生SHA：`decf887c7b36026032c881dc11f3cb334dd68953b0a29adb823921704629974b`；其原控制payload SHA：`f0277bd3e6b26797f6c600ea3091d24512e26a4b1bf1fb2c5b9e55b4e41b6240`。
- 校准5的cheese-fondue有正文裁剪，源全文不合格报告保留`passed=false`；派生后的5条正文/cap均与已执行的选中W真实input audit相等。校准W/process256响应与盲审包单独存`results/e4_calibration`，不进入主40/120分母。
- 两处E4 summary显式固定bootstrap2000/seed20261107、blind seed20261108。没有提供人工labels；自动完成状态为`automated_complete_human_pending`，不把形式一致性当作依据语义正确或内部faithfulness。

## 持久交接的停止条件

`wait_and_evaluate.py --parent-pid 11447`只跟随已亲自观察到的原P6进程，记录PID、Linux start_ticks、协议及四个新源码SHA。P6失败、PID替换/复用、运行中父进程消失、完成声明缺阶段/选点/成功receipt、源码变化、超时均写`stopped`；不重启训练或自动恢复评测。

只有P6提交真实完成且原父进程退出后才运行CPU评测preflight，随后要求`nvidia-smi`无compute进程，再启动一次评测。四个线程环境统一为4。已有handoff/evaluation记录则拒绝另起；旧记录和仍活跃子进程需人工核查。此脚本没有在本次实施中部署或启动。

## 实测与边界

- 本机三个新ops入口`py_compile`及`--help`通过；`python3 -m unittest discover -s tests -p test_evaluation_handoff.py -v`共7个CPU控制测试通过，涵盖父进程退出/失败/PID复用/receipt、GPU非空拒绝、独立greedy差异、保护字段及实际prompt/cap检查。
- 经现有SSH master只读、CUDA不可见、源码仅在内存执行的远端CPU探针通过：实际process40派生内容hash/保护字段等价、实际校准5派生等价及真实W cap/prompt hash相等。全程未import torch，未写evaluation/handoff状态。
- 同一次探针：评测`--check-only`按预期输出P6仍为`running / exact_seed42`并退出2；交接`--check-only`确认原父11447仍活跃，输出`ready_to_wait`并退出0。它表示可以观察该父进程，**不表示完整P6选点preflight或正式评测已完成**。
- 训练/开发动态脚本接口与metrics代理最终版本一致；要求两个required-complete flags和无required级unavailable，optional缺失保留。主图友好名称由真实summary的run_id动态映射，原始身份不变。
- 尚待根代理部署并完成P6后实际运行GPU评测；独立人工输入审核、5例校准与120响应双人盲审仍pending。后续不能依据test结果调整checkpoint、训练或奖励。
