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
