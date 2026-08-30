# P1-P9 实验可信度审查

审查日期：2026-08-30。审查对象包括实验驱动、聚合脚本、原始 CSV/JSON、
底层异步 I/O 实现和验收口径。`pass` 仅表示一次运行未报错，不能替代性能
或科学结论的验收。

## 总体结论

| 实验 | 当前可信度 | 可支持的结论 | 不能支持的结论 |
|---|---|---|---|
| P1 | 中 | 同型号双盘上，裸盘相对 buffered FS 的路径差异 | 裸盘普遍优于 O_DIRECT；256 MiB 请求完全同规格 |
| P2 | 低 | 三条路径的事件、系统调用和 CPU 开销不同 | 各软件层精确时间百分比 |
| P3 | 中高（单配置） | 4 MiB/depth 4 下 DMA 与 NVMe 确有重叠 | 完整 chunk/depth/延迟矩阵；真实 NVMe QD 结论 |
| P4 | 低（性能）/中（功能） | 保存、完成等待和恢复能够执行 | 平均 step overhead 小于 5% |
| P5 | 中低 | HugePage 消耗随槽数和块大小增长的趋势 | RSS 增量严格等于 slots x chunk |
| P6 | 低 | 三 seed 的 PMU issue-ratio 投影约 3% | 整机 Vector 利用率约 3%；存在可免费利用的算力 |
| P7 | 中低 | seed 42 的早中晚变化分布和热点迁移 | 三 seed 普遍规律 |
| P8 | 中低 | 已提交帧的字节核算约 25.6% | SSD/NAND 实际写量低于完整检查点 20% |
| P9 | 中（小样本） | 两个恢复位置可重建并继续训练 | 100 Delta、多策略和 GPT-2 XL 的长期正确性 |

## 关键发现

1. P4 的 `step_overhead=5.334981658` 是比例，代表约 533.5%，此前将其
   解释成 5.3% 是单位错误。async checkpoint 的前台等待均值约 19.9 秒。
   baseline 只有 32 个正式 step，async 有 302 个，独立进程吞吐差 3.6%
   受到运行长度和稳态差异混杂，不能覆盖直接测得的 checkpoint stall。
2. P6 的 `aiv_vec_*_ratio` 原始值单位确为 0--1，但聚合值是算子 PMU
   issue ratio 乘算子持续时间后除以设备时间窗。它不是设备级 Vector core
   occupancy。三次运行缺少 host/device 公共时钟 epoch，step 映射只是按
   step 时长从首个设备事件顺序铺排。约 3% 的数值可复算，但名称和推论不可信。
   `hbm.csv` 的设备平均值约为读 19.1--21.0 GB/s、写 20.2--22.2 GB/s；这是
   带宽而非利用率，且没有可靠峰值分母，不能据此判断 HBM 是否接近瓶颈。
3. P1 校准中 `/dev/nvme0n1` 与 `/dev/nvme1n1` 的 1 GiB O_DIRECT 读取均值
   分别约 296.1 ms 和 353.4 ms，相差约 19%。同型号不等于同等性能，A/B
   结果必须报告校准差异。SPDK 的 256 MiB 逻辑块被拆为 4 MiB 命令，而 fio
   使用 256 MiB 请求，因此该点不是物理请求规格一致的对照。
4. P3 的 async C 路径真实调用 `aclrtMemcpyAsync`、记录 ACL event，并在
   event 完成后提交 SPDK，时间戳能支持重叠结论。但 CSV `queue_depth` 是
   Python 到 reactor 的请求 ring 计数，不是 NVMe 在途命令数；当前延迟钩子
   位于数据写完成之后、元数据提交之前，也不等价于 SSD 服务延迟。
5. P5 在初始化后又映射 1 GiB payload，峰值 RSS 因而主要反映测试载荷。
   `HugePages_Free` 差值还包含固定 SPDK/EAL 开销。现有数据适合比较趋势，
   不能直接证明绝对 DRAM 只由槽数和分块大小决定。
6. P8 的 `nvme_bytes` 来自 `aligned frame + 4 KiB metadata + periodic FULL`
   的主机侧核算，没有读取控制器 SMART 或块设备写扇区差值。它应称为提交字节，
   不是实际 NAND 写量。10 步中一次周期 FULL 使均值为 25.6%，门槛未通过。
7. 可靠性汇总虽标记 `pass`，实际只执行 owner 1,000/10,000、Python 故障
   单测和一次 NVMe submission failure。`required_faults` 中的 ring full、slot
   exhaustion、timeout、duplicate completion、owner exit 并未全部做硬件验证。

## 数据完整性与统计限制

- P1 每配置 30 样本并提供 t 区间；但多个配置存在重复成功运行，聚合器按
  run_id 取最新值，未预注册选择规则，也未控制 SSD 温度、GC 和运行顺序。
- P3 只有 seed 41、4 MiB、depth 4、delay 0 的完整三模式组。
- P4 只有 seed 41、interval 10，且 baseline 与 checkpoint run 长度不同。
- P6 观察有三个 seed，但辅助计算仅 seed 41，且没有 none/CPU 对照汇总。
- P7 的有效输入是 seed 42 的 500 步训练，早中晚各采样 30 步。
- P9 只有 target 5 和 10；loss 偏差只有一个可比较样本。

## 后续验收要求

- P1 统一实际命令大小，随机化运行顺序，并按双盘校准结果报告归一化与未归一化值。
- P3 记录真实 NVMe outstanding 数，在设备提交/完成路径实现延迟注入后再跑矩阵。
- P4 用相同总 step、相同进程生命周期和三个 seed 重跑，直接以总 wall time、
  checkpoint stall 和非 checkpoint step 为门槛。
- P5 分离固定 SPDK hugepage、槽池、payload 和 Python/模型内存。
- P6 使用官方设备级时序指标或以 AIV active core-time / available core-time
  定义利用率，并用 profiler 证明辅助算子与训练真实并发。
- P8 同时记录提交字节、块设备扇区增量和 NVMe SMART data units written；
  P9 扩展到 100 Delta、多个恢复点和 GPT-2 XL。
