# Phase 5: 最终方案 & E2E 打通规划

> 日期: 2026-06-17 | **核心约束: sink=TRUE, GRAPH_MODE 必须遵守**

---

## 一、环境约束与架构原则

### 1.1 不可违背的硬约束

| 约束 | 原因 | 影响 |
|------|------|------|
| **sink=TRUE** | 训练性能必需, sink=FALSE per-step callback 开销不可接受 | per-step 逻辑无法在 callback 中执行 |
| **GRAPH_MODE** | delta detection 需 GE 编译进图 | Host 侧 numpy 操作不能在图内 |
| **AICPU kernel 不可注入** | GE `dlopen(RTLD_LOCAL)` → 符号不可达 | 只能在图内用 GE 原生 ops |
| **epoch_end callback 只在 epoch 边界触发** | sink=TRUE 性质 | FULL ckpt 只能在 epoch 间做 |

### 1.2 时序设计原则

```
训练步 (图内, GRAPH_MODE, sink=TRUE):
  ┌─────────────────────────────────────────┐
  │ forward → backward                      │
  │ → delta_detection (batched GE ops)      │  ← I3: 在 Vector Engine 上
  │ → optimizer update                      │
  │ → step_counter += 1                     │  ← FaF: C 层 listener 可轮询
  └─────────────────────────────────────────┘
                      │
                      ▼
  C 层 FaF listener:
    poll step_counter → 检测到 step 变化
    → 读 delta norms (可能需要 sync)        ← delta norms 在 device 上
    → Top-K 选择 + INT8 量化 + delta write   ← host 侧, SPDK 同步 I/O
    → 需要在 next step optimizer 更新前完成!

Epoch 边界:
  epoch_end callback:
    → FULL ckpt save (同步阻塞写入, SPDK DMA)
    → 不与训练步重叠 (epoch 间的停顿)
```

**关键原则**:
1. **FULL ckpt**: 同步阻塞在 epoch 边界——不干扰训练步
2. **Delta**: 每步写入，必须在 optimizer 更新前完成 → **不需要物理 barrier，但需要统计时间点验证**
3. **SPDK I/O 共享**: FULL 和 delta 共用同一个 qpair，不能并发

## 二、当前同步语义分析

### 2.1 FaF 的已知问题

FaF 设计: C 层 listener 每 10ms 轮询步号，触发 SPDK 全量写盘 (~705ms GPT-2 XL)，不与训练步同步阻塞。结果: 最后 ~35% 参数可能是"混合态"。

对**全量检查点**来说混合态可接受（参数变化 1e-7~1e-6），但对**增量检查点**来说需要确定步状态。

### 2.2 FaF → Delta 的适配方案

```
原 FaF:
  listener: step%10==0 → 触发 3.1GB 全量写 (~705ms)
  
改 Delta:
  listener: 检测到 step 变化 → 触发 delta write (1-5MB, ~50ms)
  
时序:
  step_counter 从 N→N+1 (optimizer 已更新)
  → listener 在 ~10ms 内检测到
  → delta write 在 ~50ms 内完成
  → 需要在 optimizer 更新 step N+1 的参数前完成
  
  GPT-2 Small per-step ~400ms, optimizer update ~50ms
  → listener 检测窗口: optimizer 更新后 ~350ms, delta write ~50ms
  → 完全在窗口内 ✅
```

### 2.3 需要的验证统计

| 指标 | 如何测 | 目标 |
|------|------|:---:|
| Delta write begin (ts_delta_start) | C 层 listener 打印时间戳 | — |
| Delta write end (ts_delta_end) | SPDK write 完成后 | — |
| Optimizer update begin (ts_opt) | 图内 step_counter 变更时刻 | — |
| Gap = ts_opt(next) - ts_delta_end | 计算 | > 0 (delta 在下一个 optimizer 前完成) |

## 三、SPDK 带宽问题排查

### 3.1 当前数据

| 测试 | pipeline_depth | BW | 配置来源 |
|------|:---:|----:|------|
| 原 `spdk_end_to_end.py` | **8** | 4380 MB/s | GPT-2 XL, 3.1GB |
| S4 当前 | **4** | 2252 MB/s | GPT-2 Small, 249MB |

**根因**: `pipeline_depth=4` (即只有 4 个 concurrent DMA 槽位) vs 原测试 `pipeline_depth=8`

### 3.2 修复

1. 恢复 `pipeline_depth=8`
2. 检查 chunk_size (原测试 4MB, S4 也是 4MB — 一致)
3. 验证 FULL ckpt BW 回到 ~4000+ MB/s
4. Delta write 路径不需要高 pipeline_depth (sync_meta_io 是同步的)

## 四、改造计划

### S5: 修复 SPDK 带宽并验证 FULL ckpt 同步语义

1. 修改 `DirectCheckpoint.__init__` 默认 `pipeline_depth=8`
2. 重新运行 S3 SPDK Smoke Test 确认 BW 恢复
3. 将 `spdk_end_to_end.py` 的 FULL ckpt 延迟数据与 S4 对比
4. 确保 `save()` + `wait_for_io_completion()` + `wait_async_io()` 的阻塞语义正确

### S6: 集成 FaF listener 到 delta 写盘路径

1. 修改 `ProbeTrainOneStepCell.construct()`: 加入 batched delta detection ops
2. C 层 listener 改为 per-step 触发 delta write（而非 per-N-step 全量）
3. Python 侧添加 delta norms 读取 + Top-K + 量化 + delta_save 逻辑
4. 统计计时点：opt 更新时刻 vs delta write begin/end

### S7: E2E 全链路测试 (GRAPH_MODE, sink=TRUE)

1. 训练 GPT-2 Small, 30 步, GRAPH_MODE, sink=TRUE
2. Epoch 0: FULL ckpt (同步阻塞)
3. 每步: delta detection (图内) + delta write (C listener 触发)
4. epoch 1 end: FULL ckpt (同步阻塞)
5. Recovery: 读 FULL + delta chain → 重建 → NRMSE + loss 验证
6. 统计: delta 同步时间线 vs optimizer 时间线

## 五、论文章节映射

| 章节 | 内容 | 依赖 |
|------|------|:---:|
| §3.3.4.1 | Per-param 分块 + 小参数保护 | v6 recovery |
| §3.3.4.2 | GE subgraph launch 瓶颈发现 | Phase 3.MICRO |
| §3.3.4.3 | Batched ops — 零 overhead | Phase 3.BATCH |
| §3.3.4.4 | Delta 同步语义（确定性写入） | **S6-S7** |
| §3.3.4.5 | 恢复流程（FULL+delta链合并） | **S7** |
| §4.2 | 步时 overhead 测量 | E1, E4, **S7** |
| §4.3 | 恢复精度实验 | E2, **S7** |
| §4.4 | SPDK I/O 性能 | **S5** |
