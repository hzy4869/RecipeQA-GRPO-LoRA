# 预声明统计与报告口径

本文件补充PLAN，所有远端preflight及正式训练前确定。当前仅本地准备，没有本批方法成绩。最终锁定政策须同时绑定本文件、实际独立统计源和测试SHA；种子数相同不能替代RNG engine一致性。

## 主要estimand

主评测为原manifest顺序的200个开发食谱，每题一个开放greedy回答，invalid/missing按冻结覆盖/评分政策处理，不能从分母删除。固定列次序为：SFT44、fixedβ44、halfβ44、refresh44、SFT45、fixedβ45、halfβ45、refresh45。原baseline143/200仅作已知绝对参照，不新增baseline推理。

两个预声明主效应均为百分点：

1. `D_half = 100 * mean_recipe(((refresh44-half44)+(refresh45-half45))/2)`。
2. `D_sft = 100 * mean_recipe(((refresh44-SFT44)+(refresh45-SFT45))/2)`。

二者及每flow的有符号结果全部报告，不能因成绩改主对比、只选好flow或更换开放解码。方法选择及随后独立复验须另冻结，不能用新confirmation选赢家。

## 联合配对bootstrap

- 引擎：Python标准库 `random.Random(20261107)`，不使用同seed的NumPy等价替代。
- 每个draw按顺序执行200次`rng.randrange(200)`，with replacement，共5000draw。每draw的同一索引同时作用全部8列，然后重算两个D及每flow差值。
- 点估计使用全部原200题；百分位区间取2.5%/97.5%，对5000个排序draw用位置`(5000−1)*p`、相邻值线性插值（p=0.025/0.975），不另重设seed或过滤draw。
- 保留八列逐食谱CSV、参数/引擎/分位数政策和全部两主对比的结果。独立实现用非恒定联合fixture核验索引共享、列映射与两flow权重；不能用只含常数的测试声称验证统计。
- 这些是开发探索区间，不覆盖重复使用开发、两个主对比的多重性、所有训练初始化随机性；不据“至少一个区间不跨0”宣称确认性发现。

## 诊断与边界

16×24 orbit报告每模型K/24、全24正确、modal语义一致率及配对食谱变化，独立单位为16食谱，主文不把384展示当N。其结果为预声明描述诊断，不替换200题主estimand。

末点train256的254食谱仅用于固定题同θ概率诊断：同q0的Jβ、Jβ/2、qgold、KL、解析gap及ratio-of-sums；三层严格用旧q0/旧β名单193/14/49。保持题均值为描述量，不附虚构的独立题CI。q*gold-dependent，不部署、不构造预测、不作为准确率上界。刷新后相对自身新参考的KL可记录，但不能替代同初始q0的两种J，不能因KL归零声称原J改进。

在线随θ变化的J与终点同题J分开。rawSFT实际CE/C、两RL实际J/C和共同参照目标分别命名。所有候选、实际更新、额外reference和模型加载/墙时单列；同曝光不是等计算。所有负结果和失败保留。原test已知、dev重复使用、人审pending与共用SFT起点在报告正文披露。
