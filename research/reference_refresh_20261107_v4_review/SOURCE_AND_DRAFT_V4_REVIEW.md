# v4续行源码与实际draft独立审查

**PASS：可据已审draft物化final政策；尚不可据本回执直接宣称正式训练准入。** 审查producer提交`99fcfb5`的十份源码、计划及实际`DRAFT_PROTOCOL_V4.json`（SHA `8ab79c46f6ab655ee39f3b9637b274a0ee7028417f28675eb4f32b60960538fd`，364667字节）。本机只读审查与CPU检查；未SSH、GPU、模型／processor调用，未读取新部分预测或新正式轨迹，未改producer／旧冻结文件。

授权文件实际SHA为`d83ef36a3ac2e2aa86e1765309e62da18e0bed4c628f2d56e642a2fbc721c790`。原first仍是UTC2026-11-07 04:18:24.421606，新absolute为UTC2026-11-08 04:18:24.421606：累计86400秒／38.4元。旧v3政策、72000秒／32元和失败资源门均完整保留。新contract、父政策、child、launcher、授权和最终guard使用新时钟；旧证据按原issued／started／finished时刻及旧deadline检查，不调用旧`remaining_seconds(now)`。

科学设置未改变。训练源码15个顶层函数中，`validate_authorization`只改旧技术proof的owner与fresh CPU复用先决门；包括`main`训练循环在内的其余14个函数AST均与v3相同。数据池、重复访问、128更新／64边界、奖励／β、优化器、完整状态连续性、终点评分及统计函数保持原实现。资源估计、存储算法和1.5余量／600秒margin的函数AST也不变。十个外部源码依赖SHA未改。

实际draft的20份source SHA和129项artifact SHA全部与本机已有原字节核对，18个child SHA全部匹配。对所有child逐字段比较，CLI只改`output_dir`、`reference_cache`、`wall_seconds`；其他科学参数逐值相同。父政策只改状态、累计预算／授权、绝对新截止、输出目录、CPU计数、子描述与证据绑定，并新增显式continuation／复用字段。原first、全部八臂、原池、全部原artifact SHA未变。

74逻辑阶段严格分为54条新执行与20条历史技术事件：fresh CPU/native 1 + fresh Q0审计4 + 历史train／verify 20 + 八臂正式／评估48 + finalguard 1。历史事件逐项等于旧state原事件加三项显式来源标签，原argv、PID／PGID和时间均保留；新54条要求当前同PG并分别按新阶段顺序校核。未把旧时间改成新事件，也不要求跨两批时间单调。正式臂仍从共同原起点新训；不恢复smoke优化器。

复用验证分别绑定旧父SHA、旧25命令、旧20源码、旧18child／四Q0／十技术proof、有效更新、原授权issued时间与完整物理依赖。独立从四Q0加十技术proof重建的联合清单为**1401项**，与draft精确相同；398仅是十技术proof的子集。301个payload路径本机存在，1100个缺失，不能宣称本地完整tensor／photo重放。旧stage_v3目录从其显式旧policy／state路径解析；`Files.relative`的root-first规则及nested-root回归保留。

本机实际验证：

- 新namespace原有68项测试全部通过，零失败／错误／skip／expected failure，未importTorch；十份源码测试前后SHA一致。远端完整门预期71项，另三项旧Torch测试、本次真实native探针尚未在远端执行。
- 独立以335份实际已保存metadata文件（212896266字节）构造临时隔离root，`validate_reuse(..., physical=False)`通过旧25／复用20／十技术／四Q0；临时副本已清理。没有对缺失的大二进制或照片做physical重哈。
- 在该隔离metadata副本上，旧授权issued晚于真实train启动、错误Q0 flow、联合物理清单遗漏一项三种反例均被拒绝。
- 模拟UTCNov8 01:00（晚于旧20小时截止、早于新截止），新累计时钟剩余11904.421606秒；原first／deadline／上限／预算的四种错误改动均拒绝。历史25事件仍按旧实际时间合法。

当前无源码／draft blocker。允许final只作预声明物化：draft→locked、18个child描述路径移到fresh locked目录、child状态切换、SHA重新绑定，以及添加明确的新审查／授权artifact收据；不得修改科学字段、原first、完整八臂、18角色、74／54／20计数或已审源码。新增收据须检查来源与无hash循环，不能用review绑定自身final SHA构造循环。

启动前仍须对实际final父／18子和新增收据做短绑定审查。之后由root运行新真实CPU/native和1401项完整实际物理重哈、四份fresh Q0审计、GPU空闲／锁／数据盘门，以及按实时now／free重新计算的完整八臂时间／存储门。准备和等待都已消耗原first起的时间。旧70475.17秒估算与旧存储PASS不是新门PASS。第74项guard仍在同一个native timeout进程组内，保留一字节barrier与仅自身live许可；成功退出后才允许terminal complete。

此PASS只涉及源码、实际draft及本机有限检查，不是远端运行通过、方法有效、收敛、q*可部署或课程／报告新结果验收。
