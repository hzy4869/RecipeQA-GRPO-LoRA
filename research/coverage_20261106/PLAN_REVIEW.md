# Coverage方案独立CPU审查

结论：**科学设计有条件通过，执行实现尚待独立验收。** 审查对象为当前16小时草案；没有SSH、GPU、训练、代码修改、grounding新预测读取或确认集访问。A的选择基于此前256访问和两个flow的完整负结果，不依赖grounding B条件的测量失败来建立视觉瓶颈。原12小时草案尚未启动/冻结，root已依据完整计时裕量先公告独立16小时/约25.6元上限；这不是重启已开始的阶段来增加时钟。

## 可保留的设计

- 四个**新**run共享SFT768，两个flow各SFT/Exact-KL，且同flow严格同1024题、同顺序、同256更新；是公平的曝光对照，不是相同前反向算力对照。相同LR/clip也不保证相同有效参数更新，必须继续保留梯度/clip/update记录。
- 前1024题由清单位置选定、已被初始SFT见过；不能把继续训练扩展称新数据或初始SFT增益。新1024 reference由同初始模型真实评分一次；相同输入仅重排行而非重排照片，允许有透明的flow45缓存重用。
- fixed256为唯一主终点，step64只作预定dev诊断；128/192即使保存也不评测/选点。总3136预测的算术正确。完整16×24应以16食谱为分析单位。
- **沿用原LONG_TERM门**：mean[同flow RL256−SFT256]约≥3pp，且两flow各自严格为正，才进入下一批事前锁定的至少3flow复验；不采用旧建议2pp，不直接开新200。两个flow共享起点和dev，不能把结果当400独立测试或总体稳定性证明。
- 新数据覆盖与更新次数同时扩大；1024阶段相对旧256的变化不能归因“数据规模单一作用”。若无优势，报告该预定目标/学习率/规模下缺乏稳定支持；不以“不显著”证明真实差为零，也不宣称否定所有RL。

## 冻结前须落实的执行合同

1. **派生源码边界。** 新trainer应是显式、可逐段diff的派生副本；训练数学、反传/累积/Adam更新循环保持旧实现。允许改变的合同应列为范围4/1024、per-arm runtime、save64、政策/源码绑定和导入路径。PLAN仍有一句“newwrapper”，应与实际派生copy用语统一。禁止动态monkeypatch、编辑旧冻结文件或只把CLI数字改大。
2. **三处已知旧硬约束。** 旧trainer限制limit≤256、wall≤10800；旧verifier仅接受4/256、正式save16，且SOURCE_SHA及路径写死。新verifier须接受真实4/1024和save64、独立重算数学及全部实际checkpoint二进制/optimizer CRC，不是复用旧256报告改数字。较慢旧EK1024外推已略超过3小时，per-arm上限必须受16小时剩余绝对预算约束，不能保留10800秒硬失败点。
3. **reference字段与路径。** 对flow45重排，应在冻结政策精确列出唯一允许变化的字段（代码agent声明为`index`、`reference_spec_sha256`），其余候选token分数/IDs、展示/金标、输入audit等逐项不变；来源证明另存sidecar，保留原行payload SHA及source index。pool身份一一对应，不静默drop重复/坏题。新命名空间必须显式解析旧helper路径；`scoring_source_sha256`排除的trainer文件名也要更新，不能遗留旧字面量导致refspec错误或错误漏签。
4. **资源门与工件门。** 四smoke各从共同初始模型独立运行并验收，不能拿第一flow的烟测代替其余。正式四臂共16个checkpoint，容量门按实测adapter/Adam状态字节加全部smoke、reference、评测及余量估计，不能自动删除旧失败分支挤空间。计时门覆盖reference实际剩余、所有加载/保存、3136预测、独立验收和固定裕量；不删慢臂、缩短访问或减少orbit让门通过。
5. **绝对deadline确实覆盖进程树。** 仅GNUtimeout并不自动覆盖另建session/进程组的子孙；本轮选择所有子孙留在外层timeout进程组，应由真实CPU测试验证孙进程超时后也消失。coordinator内部alarm不得覆盖外层计时。首次CPUpreflight起算16小时与冻结UTC绝对截止取较早者，清理至多10秒；后续版本/路径修复不重置。每个科学子命令也不得在deadline后继续排新工作。
6. **冻结与新reference的先后关系。** 一级协议和child CLI/路径先冻结科学内容、源码、pool/order、reference构造/重排规则和预算；真实reference生成后，以首smoke前封存的`generated_reference_receipt`实际SHA、每次启动的parent immutable检查，以及各child run_signature内的实际reference/lineage SHA共同完成第二层绑定。这里不要求改写已冻结childpolicy文件：这是与先前“生成带SHA子政策”建议等价的签名层方案。最终独立审计须串联source-reference、lineage、封存回执与child签名及时间先后，不按qref/分数选样本/参数，训练中检查无热改。

## 停止与因果措辞

健康四臂完整跑完，无成绩停止、低分重采样或变奖励。技术失败保留源/政策/PID/state/stdout与部分工件，修复新版本但沿用本轮原时钟。最终未达3pp门或两flow异号，就保留本规模下未见稳定RL增量；step64较好也不能替换主终点。通过门只表明值得后续复验，不是成功结论。若只有SFT随曝光改善，应归因更多继续训练而非RL独有能力；若两者都改善而RL无额外差值，也不能称RL胜出。

Pair、有效率、orbit一致性及训练J均为次要机制量，不能替代greedy Exact主指标；一致性可以稳定地错，较高J也可仅来自较小KL。所有输入/输出留在数据盘，secret仅内存，新200继续封存，旧test已知的事实持续披露。总租卡账单未知，16小时阶段墙时不是全部平台费用。
