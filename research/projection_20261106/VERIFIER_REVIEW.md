# Forward专用独立verifier复审

状态：**静态数学/接口审查通过，真实模型smoke尚未执行或验证**。只读本地检查，未SSH/GPU，未修改verifier或trainer。当前审阅verifier SHA `45ed0d5b6c8e2fd9fce9f9c030283a9916af15b6ecff6f6a49dd90546f7c7dec`，测试 SHA `68492e663a72bac615c206150b78ee70c18b048984642b403fa76e3b6ae9fd66`。

- 独立二元倾斜 `D=1+p_gold*expm1(1/β)` 重算qstar，β(qstar−q)系数与forward KL/C均正确，未导入trainer/math/scorer/Torch。
- 对照新trainer的signature、state/outcome、冻结目标、reference sidecars、visit/step、checkpoint文件字段，未发现schema或CLI错位。formal256/64和smoke4/1、原β、C/EOS、24动作、固定gold映射与无sampling均有独立检查。
- 原始EOS token score重建当前q/ref q，核对实际反向候选的机器零选择、四题累积计数及所有更新执行；缓存与目标保持同一参考，未以开发标签构建训练目标。
- 初始/末点safetensors字节独立重建LoRA运行时指纹；所有Torch状态哈希、容器CRC均检查且不unpickle。这个证据不能重演实际Adam张量算术，文件和报告明确披露。
- 无有效权重变化可被正确披露且仍为科学有效结果；verifier不要求提分或强制训练成功，避免低分删臂。
- 7项纯标准库verifier测试在完整本机29项套件中均通过。其余3个真实Torch测试本机缺包而skip；远端必须29项全过、0skip且CUDA未初始化。

当前stage draft中曾有旧verifier SHA，作者已在源码停改后刷新；本次最终实际核验18个source SHA全部匹配，包括本verifier。该审查仍不代替真实模型smoke或给GPU执行批准。
