# 批次 A 实验结果汇总

更新时间：2026-08-29。设备为 Ascend 910B3；SPDK 实验仅使用 `0000:83:00.0`。
首次初始化/图编译不计入稳态样本；正式微基准均采用 10 次预热＋30 次样本。失败样本
不进入成功均值。

## 结果总览

| 实验 | 正式结果 | 结论状态 |
|---|---:|---|
| E3 Host staging | XL regular 30；13B regular 30；pinned 边界失败 | 部分通过 |
| E5 XL 4 MiB 单 owner | producer 1/4/8，各 30 | 通过 |
| E5 4 MiB 控制压力 | producer 1/2/4/8，共 30/60/120/240 | 通过 |
| E1 软件栈剖析 | FS QD1/QD4、SPDK QD1 代表性运行 | 工具链通过，逐层闭合未完成 |
| E8 XL Vector PMU | 3 seeds × 2 metric groups，共 6 × 30 steps | 通过 |

## E3：Host staging 与内存边界

实验是 HBM↔Host 内存往返，不触碰 SSD；`pcie_bytes` 是往返搬运量，不能解释为磁盘
写量。

| 模型/模式 | 状态字节 | 样本 | 往返均值 | P50 | P95 | Host RSS 峰值 | VmPin |
|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 XL / regular | 3,274,208,000 | 30 | 390.44 ms | 388.09 ms | 404.81 ms | 10.16 GB | 0 |
| GPT-2 13B / regular | 26,204,712,960 | 30 | 2,643.42 ms | 2,648.27 ms | 2,654.94 ms | 55.97 GB | 0 |
| GPT-2 13B / pinned | 26,204,712,960 | 0 | — | — | — | — | 分配/传输失败 |

13B pinned 单个完整 Host buffer 返回 ACL 107000/507899 边界错误，说明不能把完整模型
状态简单地置于一个 pinned buffer。当前 E3 可以证明完整 Host staging 的内存代价和
pinned 单缓冲容量限制；还不能证明 slot×chunk 的有界内存性质。

证据目录：

- `E3/host-staging-xl-regular/E3_20260829_131916_2d5420b5/`
- `E3/host-staging-13b-regular/E3_20260829_132355_4d65fae5/`
- `E3/host-staging-13b-pinned-boundary/E3_20260829_133817_ec631379/`

## E5：单 owner 多生产者压力

XL 状态为 3,274,208,000 bytes，chunk=4 MiB，pipeline depth=4，所有运行使用一个
SPDK Reactor owner；每个波次均写入、读回并进行完整 flat-HBM SHA-256 校验。

| producer 数 | 样本 | 波次均值 | P50 | P95 | 有效吞吐均值 | 读回校验 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 30 | 45,009.65 ms | 45,014.34 ms | 45,038.61 ms | 69.37 MiB/s | 全部通过 |
| 4 | 30 | 44,923.88 ms | 44,924.35 ms | 44,955.98 ms | 69.51 MiB/s | 全部通过 |
| 8 | 30 | 45,162.69 ms | 45,049.86 ms | 45,060.86 ms | 69.15 MiB/s | 全部通过 |

producer 增加没有改变单 owner 的整体物理服务能力，吞吐保持在约 69 MiB/s；该结果
支撑资源单 owner、并发请求受控和错误可观测，不支撑“并发生产者提高物理盘吞吐”。

控制负载 4 MiB 的已有结果为 producer=1/2/4/8 有效吞吐约 62.3/34.3/20.2/13.9 MiB/s，
用于补充控制面压力对照。

证据目录：`E5/model-xl-pressure-v3/` 和 `E5/owner-pressure-4m-formal/`。

## E1：逐层软件栈剖析

已完成代表性采集：

- 84.0.0/XFS Buffered FS + fsync：QD1 256 MiB 15,233.63 ms，QD4 26,066.97 ms；
- 83.0.0/SPDK async qpair + flush/metadata：QD1 256 MiB 373,561.33 ms（包含 perf
  重新执行开销，不与 FS wall time 直接比较）；
- tracefs 可观测 block request、syscall write、writeback 六类事件；
- perf stat/record 当前返回码为 0，但环境未提供可直接用于逐层绝对时间的闭合结果。

因此 E1 当前结论是采集链路和持久化边界已建立，不能写成页缓存、文件系统/块层、设备
服务时间的精确百分比分解。FS 与 SPDK 结果属于跨盘校准，不能作为严格同盘性能比较。

## E8：真实 Vector PMU

GPT-2 XL 每个 seed 均完成 ArithmeticUtilization 和 Memory 两组真实训练，各 10 次预热
＋30 个稳态 step，并从本次 `PROF_*` 导出的 CSV 读取指标：

| seed | metric | step mean / P50 / P95 | 95% CI | HBM read / write |
|---:|---|---:|---:|---:|
| 41 | Arithmetic | 593.31 / 560.40 / 680.80 ms | 18.35 ms | 20.55 / 21.71 GB/s |
| 41 | Memory | 647.08 / 643.95 / 736.89 ms | 15.23 ms | 19.14 / 20.17 GB/s |
| 42 | Arithmetic | 620.66 / 636.78 / 675.06 ms | 20.31 ms | 19.92 / 21.01 GB/s |
| 42 | Memory | 564.90 / 561.76 / 599.92 ms | 4.97 ms | 21.04 / 22.24 GB/s |
| 43 | Arithmetic | 583.22 / 567.54 / 630.19 ms | 12.81 ms | 20.78 / 21.92 GB/s |
| 43 | Memory | 643.99 / 618.21 / 766.57 ms | 30.03 ms | 19.09 / 20.17 GB/s |

真实 XL PMU 矩阵已通过“真实训练、真实 CSV、30 个稳态样本”门禁。结果只能证明当前
训练工作负载下 Cube/Vector/HBM 的实际活动和带宽，不能直接推出增量检查点可以免费使用
Vector Engine。

证据目录：`E8/formal-gpt2xl-3seed-v2/`，汇总文件为同目录 `E8_real_summary.json`。

## 失败与修复记录

- E5 首次 XL 运行因 2318 个参数逐项同步 D2H 校验在首样本前长期无进展；改为连续 flat
  HBM 分块校验后通过。
- 随后发现 flat 分配边界没有 4 KiB 对齐，跨 1 GiB chunk 时被 NVMe 参数校验拒绝；已
  改为参数起始和总长度 4 KiB 对齐，XL 矩阵通过。
- E8 XL 首轮 seed41 因 profiler 派生 JSON/SQLite 写满 home；失败样本保留，后续运行
  增加“先解析 CSV、再清理可重生成派生文件”的策略，六组最终结果均已生成。

## 尚未完成

- E3 的 1/2/4 slot × 1/4/16 MiB、5 s 慢盘和 pinned 分段方案；
- E5 GPT-2 13B 多生产者矩阵；
- E1 全尺寸/随机写逐层时间闭合；
- E2 真正 `aclrtMemcpyAsync`＋ACL event 与 NVMe 重叠；
- E4 真实训练端到端 step overhead；
- E6/E7 控制面故障和盘上提交开销；
- E10 Host 与 NPU 图内预处理对照，以及 E12 分类感知 R2。
