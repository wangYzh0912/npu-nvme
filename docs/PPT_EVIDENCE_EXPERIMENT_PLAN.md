# PPT 证据实验计划（E0–E16）

## 1. 固定口径

- 设备统一为 Ascend 910B3；83.0.0 仅由 SPDK 使用，84.0.0/XFS 仅允许写
  `/models/npu_nvme_exp/ppt-evidence-20260829/`。
- E1 采用同型号双盘路径对照：84.0.0 测 Buffered FS/O_DIRECT，83.0.0 测 SPDK；不
  将其表述为严格同盘结论。正式矩阵前用相同 256 MiB 控制负载校准两盘。
- 固定记录 Git/SPDK commit、MindSpore、MindFormers、CANN、驱动、固件、内核、模型、
  seed、NPU、NUMA 和 SSD PCI 地址；首次设备初始化与图编译不计入稳态统计。
- 微基准为 10 次预热＋30 次正式样本；真实训练每组至少 3 seeds、至少 30 个 checkpoint。
  输出 mean、median、stdev、95% CI、P95；样本数小于 30 不报告 P99。
- Buffered FS 使用 `fsync/fdatasync`；SPDK 使用 NVMe flush 和 metadata commit。失败样本
  单独写入 `failures.jsonl`，不进入成功均值。

## 2. 证据包与接口

统一结果目录为：

```text
results/ppt-evidence-20260829/<EXPERIMENT_ID>/<RUN_ID>/
  config.json environment.json commit.json samples.jsonl timeline.jsonl
  result.json failures.jsonl raw/
```

`result.json` 至少包含 `status/model/seed/mode/state_bytes/logical_bytes/physical_bytes/`
`chunk_size/pipeline_depth/slot_count/latency_mean/latency_p50/latency_p95/throughput/`
`foreground_wait/step_overhead/host_rss_peak/pinned_dram_peak/hbm_peak/cube_util/vector_util/`
`hbm_bandwidth/pcie_bytes/nvme_bytes/recovery_error/loss_deviation/fault_results`。

先从 `io_matrix.py` 抽取公共 evidence writer、统计函数、环境采集和 timeline 校验，补齐
`commit.json`、CANN 实际版本探测、失败隔离和缺失字段的 `null` 语义。历史结果只读导入，
保留原始路径、SHA-256、样本数和语义标签；只有原始数据无法计算 CI、少于 2 个成功样本、
损坏或持久化边界不可比时才重跑。

真正异步 DMA 实现使用 C 层 request ID、submit/poll/bounded-wait/metrics 接口；状态固定为
`QUEUED → DMA_INFLIGHT → DMA_DONE → NVME_INFLIGHT → DATA_DONE/FAILED`。使用
`aclrtMemcpyAsync` 和 ACL event，event、NVMe completion、flush、metadata commit、ACK
完成前均不得复用 slot；slot 满返回 `-EBUSY`，禁止静默丢弃。已有 blocking API 保持兼容，
由 submit＋wait 实现。

## 3. 执行顺序

### 批次 A：直接形成 PPT 主线

1. 整理 E0-1～E0-7：路径对照、批量提交、流水深度、分块大小、多槽背压、盘上恢复、增量
   变化观察；统一统计与历史标签。
2. E1：84.0.0 的 Buffered XFS/O_DIRECT 使用 io_uring，83.0.0 使用 SPDK 异步 qpair；
   顺序写覆盖 4 KiB/64 KiB/1 MiB/4 MiB/256 MiB、QD=1/4，随机写作为附录。使用 perf、
   tracefs、SPDK timeline 分解应用、页缓存、文件系统/块层、队列、设备和 flush，分解
   残差目标不超过端到端时间 10%。
3. E3：GPT-2 XL/13B，完整 Host staging 与 1/2/4 slot，chunk=1/4/16 MiB，正常盘和
   5 s 延迟；记录 RSS、pinned DRAM、HBM、slot wait、吞吐和 OOM。
4. E5：受控同步、request-ring、request-ring＋batch，400 KiB/4 MiB/XL/13B，生产者
   1/2/4/8；只使用单 Reactor owner。
5. E8：移除 Vector PMU 历史硬编码，真实启动 GPT-2/XL 训练，msprof 解析本次 CSV，3
   seeds、编译后至少 30 个稳态 step。

### 批次 B：异步流水与可靠性

6. E2a：GPT-2 XL 冻结 HBM 状态扫描串行、现有队列流水、真正 DMA 异步流水，覆盖
   depth=1/2/4/8、chunk=1/4/16 MiB、正常/100 ms/1 s/5 s 延迟，10＋30 样本。
7. E2b/E4：XL 真实训练覆盖 checkpoint interval=1/5/10/20/50，3 seeds、至少 30
   checkpoint；13B 仅执行规模复核和最优配置，仍要求 3 seeds。记录 checkpoint/non-
   checkpoint step、前台等待、积压、generation、loss 和恢复。
8. E6：ring 满、slot 耗尽、ACL/NVMe/metadata 错误、超时、重复 completion、错误
   generation/ACK、cleanup 在途和 producer 退出，执行 1,000/10,000 请求压力。
9. E7：单 metadata、A/B＋CRC＋generation、损坏回退，metadata=4 KiB/64 KiB/400 KiB/1 MiB，
   payload=4 MiB/256 MiB/XL FULL。

### 批次 C：Vector 与分类感知增量

10. E9：差分、norm、选择、FP16、INT8 和完整链，数据规模 1/10/100/500 MiB、1/2/4/8
    GiB 及 XL 分片，block=65K/262K/524K；必须使用 device event/msprof。
11. E10：Host D2H＋CPU、HBM 图内处理、图内处理＋后台 I/O、无 checkpoint；固定选择
    比例、编码、FULL interval、误差和持久化边界。
12. E11：只整理既有 GPT-2 三 seed/XL 500-step 轨迹，明确标注为逻辑轨迹估计。
13. E12：CPU 分类感知策略。Model 使用 Top-K 1/5/10/20%＋FP16/INT8；Adam-m 使用
    20/50/100% 或 error-budget 90/95/99%；Adam-v 使用 FP16/INT8 全量或每 4/8/16 次
    刷新；small/control/optimizer-other 每次 raw 全量；FULL=20/50/100/200，max-age=4/8/16。
    先离线筛选 Pareto 候选，再执行 3 seeds 精确 10/100-step recovery。
14. 仅 E12 同时满足写量<20%、单步 NRMSE≤5e-3、恢复 loss≤1%、无超龄块时执行 E13
    分类感知 NPU；E13 通过后执行 E14 XL 长程、慢盘、崩溃和 ring 回绕。

### 批次 D：最终正确性与扩展

15. E15：GPT-2 smoke 后，对 GPT-2 XL 执行 FULL/Delta/metadata/ACK/回绕故障矩阵；每
    项验证未完成 generation 不可见、旧 generation 可恢复、控制态一致和可续训。
16. E16：两 rank HCCL、per-rank shard、全局 PREPARED/COMMIT、单 rank 失败不发布、
    fresh-process 恢复；通过后再扩展四卡 GPT-2 13B。

## 4. 验收与提交

- 每个 E 编号先做 4 MiB/256 MiB 硬件 smoke，再做正式矩阵；所有提交前检查 `npu-smi`、
  PCI driver、NUMA、83 raw offset 和 84 授权目录。
- 83.0.0 使用 `flock` 保证同一时刻只有一个 SPDK owner；NPU 可并行，但每次启动前必须
  检查进程表。
- 只有真实 CSV、真实事件时间线、可恢复 generation 和成功样本才能进入 PPT 正式表；
  preliminary、historical、cross-disk、trajectory estimate 和负结果必须显式标注。
- 代码、证据、文档按批次独立提交并推送 `exp/ppt-evidence-20260829`；核心实验确认后再
  合并远端主分支。

## 5. 执行状态（2026-08-29）

- E0：已完成历史结果统一导入；E1：已完成 84.0.0/XFS `io_uring` 与 83.0.0/SPDK
  async-qpair 的代表性采集，结果包含端到端时间和 trace/perf 状态，但尚未形成逐层绝对
  时间闭合。E1 是跨盘校准结果，不作为严格同盘比较。
- E3：已导入 GPT-2 XL 真实 HBM snapshot 的 1/2/4 槽、正常盘和 5 s 延迟共 6 组历史
  实测；均完成 SHA-256 回读校验，但每组仅 3/4 个样本，且没有历史 RSS/VmPin 峰值，
  只能作为 partial/historical 证据。新增正式运行曾在首个训练 cell 的 Ascend 驱动同步
  阶段超时，已独立保存失败证据，不计入统计。
- E3 尚未完成：完整 Host staging 对照、RSS/pinned DRAM 实测、chunk=1/16 MiB、GPT-2
  13B 和每配置 30 个正式样本。完成这些缺口前，不宣称“内存随 slot×chunk 有界”的
  正式结论。
- E5：已完成 400 KiB、单生产者、单所有者的受控同步 vs request-ring＋batch，10 次
  预热＋30 次正式样本，读回校验通过；request-ring 写均值约 6.339 ms、读均值约
  1.765 ms。一次独立重跑曾出现同步读回不一致，已作为未通过样本保留在原始目录，
  成功重跑才进入正式证据包。
- 下一执行点：补齐 E3 slot×chunk×慢盘矩阵和 E5 GPT-2 13B 多生产者压力；随后执行 E1
  全尺寸逐层闭合，以及 E2/E4 的真实异步 DMA 和训练端到端门禁。E2/E4 异步 DMA 仍不得
  用已有队列级流水结果替代真正 `aclrtMemcpyAsync` 证据。

对应证据：`results/ppt-evidence-20260829/E3/summary.json`、
`experiments/benchmarks/e3_hbm_evidence.py` 和 `experiments/benchmarks/summarize_e3_ppt.py`。

## 6. 批次 A 增补执行计划（2026-08-29）

本节将 E3、E5、E1 和 E8 的新增要求固化为可执行门禁；未达到门禁的运行只保留在
`failures.jsonl`/失败目录，不进入 PPT 成功均值。

### E3：Host staging 与内存峰值

- GPT-2 XL：完整状态 Host staging、1/2/4 slot，chunk=1/4/16 MiB；正常盘和 5 s
  慢盘；每个配置 10 次预热＋30 次正式样本。
- GPT-2 13B：先执行单卡 checkpoint-only 的完整状态 Host staging/单槽基线；2/4 槽
  使用四卡真实分片、rank0 单 I/O owner，不把单卡无法容纳的配置伪装成完整模型结果。
- 记录进程 RSS、VmPin/VmLck、pinned DRAM、HBM snapshot slot、分配次数、slot wait、
  吞吐、OOM/退化和 checkpoint 完成率。VmPin 不可见时记录 `null`，不以 slot 大小替代。
- 通过条件：实测内存峰值随 `slot_count × chunk_size` 有界，并且没有以吞吐、回读校验或
  恢复正确性换取峰值下降。

### E5：单所有者控制压力

- 负载固定为 4 MiB 数据请求和 400 KiB 控制请求；对照为受控同步、单 owner request
  ring、单 owner ring＋batch。
- 覆盖 GPT-2 XL/13B checkpoint-only 负载及 producer=1/2/4/8；每个可运行配置至少
  10 次预热＋30 次正式样本，压力组另执行 1,000 请求。
- 记录 owner 线程、producer 数、队列等待、ring 占用、P50/P95/P99（仅 n≥30）、Reactor
  CPU、producer CPU、上下文切换、锁等待、busy、completion 和错误传播；禁止重新引入
  多线程共享 SPDK context 的不安全对照。
- 通过条件：无死锁、双释放、静默覆盖或状态错乱；失败 generation 不可见；所有请求
  都有 completion/busy/error 结果。

### E1：逐层软件栈剖析

- 84.0.0/XFS 只测 Buffered FS/O_DIRECT＋`fsync/fdatasync`；83.0.0 只测 SPDK
  async-qpair＋flush/metadata commit。两盘先用同样 256 MiB 控制负载做校准，结果标记
  为跨盘同型号校准，不写成严格同盘结论。
- 顺序写覆盖 4 KiB/64 KiB/1 MiB/4 MiB/256 MiB，QD=1/4；随机写作为附录。每组保留
  10 warmup＋30 formal，并另做低重复 profiler run。
- `perf stat/record`、tracefs/ftrace（权限允许时）、fio JSON、SPDK timeline 共同记录
  应用处理、拷贝、syscall、页缓存/回写、文件系统/块层、队列等待、设备服务、flush 和
  端到端时间；不可观测层明确写 `unavailable`，不伪造分解值。时间分解闭合残差目标 ≤10%。

### E8：真实 Vector PMU

- 移除历史硬编码作为正式结果的路径；GPT-2、GPT-2 XL，各 seed=41/42/43，各运行
  ArithmeticUtilization 和 Memory 两组；每组 10 次预热＋30 个真实稳态训练 step。
- 每次运行记录 commit、设备、采样窗口、step/loss、真实 `PROF_*` 原始目录；通过
  `msprof --export=on --type=text --summary-format=csv` 解析本次 run 的
  `op_statistic`、`op_summary`、`hbm` 数据。
- 通过条件：无真实 CSV 的 run 必须失败；Cube/Vector/HBM 指标来自本次采样，且只作为
  资源利用率证据，不直接表述为“免费算力”。

执行顺序固定为 E8 smoke→GPT-2 三 seed→GPT-2 XL 三 seed；E3/E5 正式硬件运行必须先
  通过 `npu-smi info` 检查空闲 NPU，并通过 83 lock 保证单一 SPDK owner。

## 7. 批次 A 执行记录（2026-08-29，持续更新）

### 已完成并可纳入证据

- **E3 Host staging**：GPT-2 XL 和 GPT-2 13B 的 regular Host staging 均完成 30 个
  正式样本，覆盖完整 checkpoint 状态的 D2H/H2D 往返，并记录 RSS/VmPin。XL 状态为
  3,274,208,000 bytes，13B 状态为 26,204,712,960 bytes；13B regular 样本中 RSS 约
  55.97 GB、VmPin=0。单个完整 pinned host buffer 的 XL/13B 分配均以 ACL 返回码
  107000 失败，作为边界负结果保存，不能用 slot 大小替代 pinned DRAM 峰值。
  这批结果证明了完整 Host staging 的内存代价和 pinned 单缓冲的容量限制，但尚未证明
  `slot_count × chunk_size` 有界；1/2/4 slot、1/16 MiB 的 Host staging 矩阵仍待实现。

- **E5 4 MiB 控制压力**：单 owner、SPDK Reactor、producer=1/2/4/8 均完成 30 个正式
  样本（相应样本数为 30/60/120/240），写入/读回和控制路径通过。生产者由 1 增至 8
  时，批量有效吞吐约由 62.3 MiB/s 降至 13.9 MiB/s，端到端延迟随并发上升；该结果是
  控制面和 owner 串行化压力证据，不应解释为物理盘吞吐提升。

- **E1 软件栈剖析**：84.0.0 的 Buffered FS/O_DIRECT＋fsync/fdatasync 代表性运行、
  83.0.0 的 SPDK＋flush/metadata 边界代表性运行均保留 perf stat/record、tracefs
  可用性和端到端结果。当前 tracefs 可观测 6 类 block/syscall/writeback 事件，但尚未
  形成“页缓存/文件系统/块层/设备服务”绝对时间的闭合分解；因此 PPT 只能写成
  instrumentation-ready/代表性路径结果，不能伪造逐层百分比。

- **E8 Vector PMU**：GPT-2 seed=41/42/43 的 ArithmeticUtilization 和 Memory 共
  6 个真实训练运行均通过；每个运行有 10 次预热＋30 个稳态 step，并由本次 `msprof`
  CSV 解析 Cube/Vector/HBM 指标。GPT-2 的 step mean 约 83.3–92.6 ms，HBM read
  约 13.0–14.0 GB/s、write 约 10.5–11.4 GB/s；结果只表示实际资源利用率，不等同于
  “可免费使用的算力”。

### 进行中/待完成

- E5 GPT-2 XL 4 MiB、producer=1/4/8 的完整状态单 owner 矩阵已完成；每组 30 个正式
  样本，完整 flat-HBM 读回哈希均通过。此前一次未产生首样本的运行已标记为
  per-parameter D2H 校验超时，现行版本改用 4 KiB 对齐的连续 HBM flat snapshot，失败
  和修正路径均保留。
- E8 GPT-2 XL 三 seed 已完成 ArithmeticUtilization/Memory 共 6 组真实 PMU 运行；每组
  10 次预热＋30 个稳态 step，结果由本次 `PROF_*` CSV 解析。为避免 XL profiler raw
  JSON/SQLite 填满 home 文件，已保留真实 CSV/命令/结果并清理可重生成派生 JSON/DB；
  首轮空间失败目录作为负样本保存。
- E5 GPT-2 13B 多生产者正式矩阵尚未开始；必须在 XL 运行结束并确认 NPU4、83.0.0
  释放后执行，优先 producer=1，再扩展 producer=4/8。
- E3 slot×chunk×慢盘矩阵、E1 全尺寸/随机写的逐层闭合、E2 真正
  `aclrtMemcpyAsync` 重叠、E4 真实训练端到端影响、E6/E7 故障和提交开销仍属于后续
  门禁，不能由本批次结果替代。

## 8. P1–P9 成稿前执行计划（2026-08-29）

本节是下一阶段唯一执行入口。P1–P9 使用统一的 `results/ppt-evidence-20260829/P<编号>/`
结果目录、10 次预热＋30 次正式微基准样本、3 seeds 真实训练和失败样本隔离。P5/P6/P7
分别固定为环形内存、Vector 利用率、变化规律；P8 为实际增量写量，P9 为恢复正确性。

### 8.1 固定口径和安全边界

- 设备固定 Ascend 910B3；83/84 为同型号 Huawei ES3000 V6。83.0.0 仅用于 SPDK 和
  声明安全偏移的临时 O_DIRECT 校准；84.0.0 文件系统实验只写
  `/models/npu_nvme_exp/ppt-evidence-20260829/`。
- 83 的 O_DIRECT 校准允许临时切回内核 NVMe 驱动，使用64 GiB安全偏移；运行前读取并
  校验superblock/FULL/Delta布局，若区域重叠则停止。校准后恢复SPDK绑定并重新probe。
- 慢盘延迟只在每个generation的payload完成后、metadata/ACK前注入一次，避免将XL单个
  分块延迟误报为设备服务时间。
- 大型 `msprof` 原始目录使用 `/tmp/npu-nvme-ppt-raw/<RUN_ID>/`；结束后提取必要CSV、
  命令、时间线和SHA-256清单，原始失败目录不计入成功均值。

### 8.2 实验顺序

1. P1：84 Buffered FS/O_DIRECT 与83 SPDK的4 KiB/64 KiB/1 MiB/4 MiB/256 MiB、QD1/4、
   读写公平矩阵；写入分别等待fsync/fdatasync或flush＋metadata commit，读取主结果清冷缓存。
2. P2：在P1的4 MiB/256 MiB代表组上使用perf、tracefs、block trace、fio JSON和SPDK
   timeline分解应用、复制、页缓存/writeback、文件系统/块层、队列、设备服务和flush；
   互斥时间桶闭合残差目标≤10%，否则降级为事件/CPU开销对比。
3. P3：实现C层 `submit/poll/wait/release` 请求句柄，使用 `aclrtMemcpyAsync`＋ACL event
   串联DMA和SPDK，比较串行、现有队列、真实异步三条路径；XL覆盖chunk=1/4/16 MiB、
   depth=1/2/4/8和正常/100 ms/1 s/5 s尾部延迟。
4. P4：XL seed 41/42/43比较无checkpoint、同步FULL、现有队列和真实异步；interval=1/5/10/
   20/50，至少30个正式checkpoint，记录普通step、checkpoint step、前台等待、积压、
   busy、generation、tokens/s、loss和最终drain。
5. P5：XL完整训练状态比较full staging与1/2/4槽、1/4/16 MiB、正常盘和5 s尾部延迟；
   记录RSS、VmPin/VmLck、pinned DRAM、HBM、slot wait和吞吐，13B只复核最佳配置。
6. P6：从msprof task/op时间戳构造与step对齐的10 ms Vector/Cube/HBM时间序列，并注入
   差分、norm、Top-K、FP16、INT8和完整链；仅当oracle一致且step overhead≤5%时支持
   “Vector低利用窗口可用于预处理”。
7. P7：GPT-2三seed的已有100-step轨迹加XL seed 41/43补齐三seed、500-step早/中/晚窗口，
   输出三种block size的能量覆盖、Jaccard和状态分类贡献；结果标注为轨迹估计。
8. P8：CPU分类感知Model/Adam-m/Adam-v策略扫描并选最低写量、最低误差、Pareto最多三组，
   在83执行真实frame、metadata、对齐和周期FULL，按SPDK提交字节及SMART增量统计实际写量。
9. P9：对候选执行FULL＋10/100 Delta、fresh-process恢复、控制态校验和10/100步续训；
   联合写量、NRMSE、loss、年龄和generation门禁决定Go/Pivot。

### 8.3 异步接口和验收门禁

新增不透明 `NPUNVMERequest` 及 `npu_nvme_submit_write_batch()`、
`npu_nvme_poll_request()`、`npu_nvme_wait_request()`、`npu_nvme_release_request()`；
现有阻塞API由submit＋wait实现。请求逐块状态为
`QUEUED→DMA_INFLIGHT→DMA_DONE→NVME_INFLIGHT→DATA_DONE/FAILED`，ACL event、NVMe
completion和ACK前不得复用槽位，槽满返回`-EBUSY`。

P3通过要求真实异步路径在4 MiB/depth4正常盘下重叠率中位数≥0.30且CI下界>0，端到端
耗时较串行下降≥10%；P4平均step overhead≤5%；P5内存随slot×chunk有界且实用配置吞吐
不低于full staging的90%；P8实际摊销写量低于FULL-only的20%；P9单步NRMSE≤5e-3、
恢复loss偏差≤1%、无超龄块且所有已提交generation可恢复。未达到的结果保留为正式负结果，
不得用历史或逻辑估计替代。

P3/P4 FULL异步独立于P8/P9推进；只有P8/P9同时通过，才实现分类感知NPU增量和长程Delta。
每阶段独立提交并推送 `exp/ppt-evidence-20260829`，确认核心结果后再合并master。
