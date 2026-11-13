# SFT恢复差异的精度核查

核查日期：2026-11-05（香港）。对象：`runs/pilot_sft_debug_v2`，连续20步与从checkpoint18重启后完成20步的比较。未修改运行源码、模型或checkpoint，未启动GPU任务；以下新增探针均在远端以`CUDA_VISIBLE_DEVICES=''`和2个CPU线程执行。

## 结论

**目前没有发现LoRA保存／加载降精度或AdamW状态被转成BF16的证据。无需因此阻断W准备。** 已证实模型、优化器的相关CPU重载路径保留原dtype和数值；不能据此证明GPU跨进程继续训练逐位一致，也尚未定位差异的具体算子。

连续与重启轨迹的最终LoRA最大绝对差约`3.2094e-5`，optimizer最大差约`5.793e-4`。恢复后的输入、input hash、学习率、scheduler及最终RNG一致；第二个恢复更新的梯度范数仍有差异。这些是主执行者的真实比较结果，不能标为bitwise恢复，也不能因为绝对数值看起来小就自动标为numerical_close。

## 本次CPU实测

1. 读取实际安装的Transformers 5.22.1源码，Trainer的PEFT恢复调用为`model.load_adapter(checkpoint, active_adapter, is_trainable=True)`。PEFT 0.21.2该接口的`autocast_adapter_dtype`默认值为`True`。
2. checkpoint18的adapter包含392个张量，全部为`torch.float32`。以原始BF16基座创建同样的语言LoRA，392个可训练张量同样全部FP32。
3. 在CPU按上述真实接口加载checkpoint18，再用`get_peft_model_state_dict`提取状态，与原safetensors逐项比较：键集合相同、392/392张量dtype和`torch.equal`数值相同，差异数为0。此探针检查数值与dtype；没有把它夸大为全部GPU执行状态的逐字节证明。
4. checkpoint18的AdamW包含392参数的首组及0参数的第二组；`step`、`exp_avg`、`exp_avg_sq`各392个张量，全部FP32。按同样的两组构建CPU AdamW并执行`load_state_dict`，所有状态张量的dtype和数值差异数为0。
5. 实际Trainer的设备迁移代码仅调用`model.to(device)`，不在该路径指定BF16 dtype。SFTTrainer中另一处BF16 adapter转换受`_is_quantized_model`条件控制；本项目未量化，不走该转换。

这排除了本次探针覆盖到的CPU保存／重载精度转换问题；不等于完整审计了每个GPU内核和所有未序列化执行状态。

## 确定性边界

原run的resolved配置为：BF16开启、FP16关闭、`full_determinism=False`、`tf32=None`、`torch_compile=False`、`dataloader_num_workers=0`。相同env.sh下未设置`CUBLAS_WORKSPACE_CONFIG`。

独立CPU探针看到的默认值是：deterministic algorithms关闭、cuDNN deterministic关闭、benchmark关闭；flash、memory-efficient、math SDPA均允许。这些是新探针进程的默认值，**不是对仍运行GPU进程内部标志的直接读取**。

[PyTorch 2.8确定性文档](https://docs.pytorch.org/docs/2.8/notes/randomness.html)说明固定随机种子还需要排除非确定性操作，并提供确定性算法及CUDA workspace设置。[SDPA文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)也说明不同后端可有浮点数值差异，且存在非确定性的CUDA情形。因此当前现象与未强制确定性的BF16/CUDA计算相容，但这些文档不能证明本次差异就是某个SDPA内核造成的。

## 最小处置

- 继续W准备及后续已授权pilot，保留连续与恢复两套原始工件和CPU探针结论。
- 当前准确表述为：**参数及优化器重载精度核验通过；数据位置、学习率、scheduler和RNG恢复已核；跨进程继续训练未达到逐位／当前容差下的数值等价。**
- 正式三分支统一使用冻结的数值配置，记录真实恢复attempt和成本，使用实际终点结果。不要只给一个分支临时开启不同后端或精度。
- 若最终必须进一步追求逐位复现，可另做固定内核、确定性算法和CUDA workspace的小控制；它是定位工具，不是此时改变W或扩大项目范围的必要条件。

本审查不将已观察的差异自动归为科学失败，也不宣称差异对最终准确率没有影响；后者需要实际预测比较。
