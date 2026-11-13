# 24小时续跑资源门独立核对

本机旧证据身份一致，**不构成v4新运行放行**。20份旧源、18份子政策在repo与mirror的SHA均匹配；25条命令为冻结日程的前25条且全部rc0、无active子进程、尚无formal。4份q0与10份技术proof的身份／role／旧child／verifier绑定一致。旧v3仍是time gate失败，不能改写其状态或政策。精确路径、SHA／字节见同目录JSON。

六项时间必须全部保留：训练33615.870943秒；8次加载／checkpoint开销272.177852秒；2次256行refresh1873.459611秒；8次256行final bank7493.838445秒；4672预测及16次加载2974.322741秒；8次独立核验353.777495秒。总和46583.447087秒，按原1.5倍安全系数加600秒，仍为**70475.170630秒**。保留全部10实测及四个完整训练floor、旧完整reference成本和四条384-emission预测计时；不得挑快值或降低余量。

本轮授权保持原first UTC2026-11-07 04:18:24.421606，累计86400秒／38.4CNY，新deadline UTC2026-11-08 04:18:24.421606。仅作算术快照：UTC07:18:56.264913剩75568.156693秒，比旧估算多5092.986063秒；若估算不变，最晚门时间为**UTC08:43:49.250976**。这不是runtime PASS。必须在所有部署／CPU／核验准备后用realnow重新计算并要求estimate<remaining；旧69139.742397不能复用。

技术proof的required-artifact并集398项、无冲突；291项与immutable local snapshot元数据一致，107项延续已知remote-only边界。当前仅复核metadata及源／政策／proof文件，未重读轨迹／候选概率或本地加载缺失tensor。两个path的4个continuity物理SHA不同于逻辑内容指纹；应同时保留物理SHA／字节及原CPU内容重算证明，不可仅比较ZIP字节。v4须在真实远端文件上严格核398项身份。

已有12个technical checkpoint和4个continuity文件须保留，已占用空间反映在fresh free中，不重复计为未来写入；未来仍需16 checkpoint、4 continuity。现有保守checkpoint最大221037238字节、adapter69788264字节、actual continuity最大209912581字节，continuity上界为290825502。公式 `ceil(1.2*(16*checkpoint+4*continuity+512MiB)+1GiB)` 得7357864298字节。必须重新测远端实际大小与free；旧8549683200字节free不能充当新门。

最小续跑保留旧v3政策／失败门／原clock及4q0／10技术全部原收据，并单独冻结v4来源与授权lineage。旧CPU67与旧idle／锁／watchdog不是v4新证据：新源CPU/native gate、当前锁／idle／cache／数据盘限制、空新输出目录、fresh完整八臂时间／空间门均必需。formal从共同SFT768新起，不从technical checkpoint恢复。新absolute watchdog必须覆盖formal后代、评估、独立核验和final global guard，不能重置first或删臂减步。

本次无SSH／GPU、科学重跑、confirmation/private或新partial读取；只新增本RESOURCE审查两文件。尚待上述fresh runtime条件；未发现旧证据身份blocker。
