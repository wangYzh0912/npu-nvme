# NPU-NVMe 实现计划 (合并版)

> 2026-06-27 | 合并自: I1_SPDK_plan, I2_FaF_plan, I3_VectorEngine_plan, PHASE5_FINAL_PLAN  
> Step 1 ✅ → Step 2 ✅ → Step 2b ✅ → Step 3 🔲 → Phase C (论文实验) 🔲

---

## 一、统一基准配置 (不可变)

| 参数 | 值 |
|------|------|
| NPU | Ascend 910B, device_id=1 |
| NVMe | `0000:83:00.0`, SPDK user-space driver |
| HBM | 64 GB |
| MindSpore | 2.5, `/root/miniconda3/envs/ms_2.5/bin/python` |
| 模型 | **GPT-2 XL** (48L/1600d, 772 params, **3.12 GB FP16**) |
| 运行模式 | **GRAPH_MODE, sink=TRUE** (硬约束) |
| SPDK BW (写/读) | 4412 / ~6800 MB/s (pipeline_depth=8, chunk=4MB) |

### 增量管线配置 (不可变)

| 参数 | 值 | 说明 |
|------|:---:|------|
| BLOCK_SIZE | 524288 (512K) | 1MB FP16 = 512KB INT8 |
| top_k | 0.10 (10%) | 296/2969 blocks for GPT-2 XL |
| delta_slot_size_mb | 256 | per-slot 256MB |
| delta_slot_count | 128 | ring buffer slots |
| small_threshold | 10000 elems | small-param threshold |

---

## 二、已完成: Step 1 — 基准性能测试 ✅ (2026-06-17)

### 核心数据

| 指标 | 值 |
|------|:---:|
| GPT-2 XL 步时 | **468.3 ms** (±25.8, COV 5.5%) |
| 编译时间 | **150.9 s** |
| GE kernel instances | **132,170** (AIC=10,944, AIV=113,005) |
| Cube MAC 利用率 | **55.7%** |
| Vector ALU 利用率 / idle | **12.4%** / **87.6%** |
| HBM 占用 (训练后) | **35%** (~22.9 GB / 64 GB) |
| SPDK FULL ckpt BW | **4412 MB/s** (2.90 GB in 674ms) |
| SPDK pipeline DMA BW | **~52,000 MB/s** (8-buf overlap) |

### 结论

Cube 利用率中等 (55.7%)，Vector Engine 大量闲置 (87.6%) → I3 增量计算有充足算力空间。SPDK 写入 BW 足够快 (4412 MB/s)，delta frame (< 200MB) 可在 50ms 内写完。

---

## 三、已完成: Step 2 — 图内量化可行验证 ✅ (2026-06-18)

### 设计

```
GE 图内 (construct, optimizer 后):
  AllBlocks = Concat(all_params) → [2969, 512K] FP16
  Delta = Sub(AllBlocks_fp16, Cast(P_old_int8, fp16))
  Norms = ReduceSum(delta²)
  Indices = TopK(Norms, 296)
  Quant = INT8_quant(Gather(AllBlocks, Indices))
  → Assign(quant_buf, scales, idx_buf)  [全部在 HBM]
```

### 7 项验证结果

| # | 验证项 | 结果 |
|:---:|------|:---:|
| V2.1 | GRAPH_MODE 编译 | ✅ 编译通过, 3038 blocks, 不 OOM |
| V2.2 | Delta norms 正确性 | ⚠️ FP16 vs FP32 Top-K overlap 86.8% (边界块), 可接受 |
| V2.3 | INT8 量化精度 | ✅ per-element 误差 ±1 bin |
| V2.4 | P_old 图内更新 | ⚠️ MS 2.5 ScatterUpdate INT8 bug → host callback 降级 |
| V2.5 | HBM buffer ptr | ✅ 4 个 buffer 全部非零 |
| V2.6 | 步时 overhead | ⚠️ 单步波动大, 需 Step 3 多步稳定测量 |
| V2.7 | C 层 HBM→NVMe delta write | ✅ 159MB in 45ms → 3350 MB/s |

---

## 四、已完成: Step 2b — 断点续传验证 ✅ (2026-06-18)

| 指标 | 值 | 判定 |
|------|:---:|:---:|
| Step 1 NRMSE | **0.003** | 近乎完美 |
| Step 100 NRMSE | **0.76** | 累积偏离 (预期) |
| NRMSE drift | **+3.1e-3/step** | 线性增长 |
| 压缩比 | 10% | 296/2969 blocks |

**结论**: NRMSE drift 是 top-10% 选择的预期行为 — 每步只更新 10% block，剩余 90% 逐渐偏离。论文论证 tradeoff (top_k 越大 NRMSE 越小但压缩比越低)。

---

## 五、进行中: Step 3 — I3 全路径打通 🔲

### 5.1 核心设计决策 (修订版, 2026-06-23)

| # | 决策 | 理由 |
|:---:|------|------|
| D-I3.1 | P_old = INT8 HBM Parameter (~1.52 GB) | 模型 FP16 的 50%, HBM 占用可接受 |
| D-I3.2 | P_old 每步**全量 Assign** 更新 (不 ScatterUpdate) | 绕开 MS 2.5 ScatterUpdate bug, ±1 bin 无累积误差 |
| D-I3.3 | Delta = W_current - P_old (上一完整快照) | 每步独立 ±1 bin 误差, 不累积 |
| D-I3.4 | **双量化路径**: 输出(Top-K→SPDK) + P_old(全量→Assign) | 一次前向完成全部增量计算 |
| D-I3.5 | FaF listener 读 HBM quant_buf → write_batch → NVMe | 图内 Assign → FaF 异步持久化 |
| D-I3.6 | FULL ckpt 同步阻塞在 epoch 边界 | 不干扰训练步 |

### 5.2 架构

```
TrainCell.construct() (GE 图, sink=TRUE):
  Step N: forward → backward → optimizer → Phase A-G (delta+quant+topK+P_old)
  → Assign(quant_buf/scales/idx_buf/P_old_int8)  [全部在 HBM]
  → AssignAdd(step_counter)

C 层 FaF listener (独立线程):
  poll step_counter → detect step N done
  → HBM→NVMe DMA (quant_buf/scales/idx_buf via registered_tasks)
  → signal probe_flag → safety check in epoch_end callback

Epoch 边界 callback:
  → FULL ckpt sync write (SPDK DMA)
  → per-step: no Python involvement (FaF 全部异步)
```

### 5.3 实施步骤与验证

| # | 任务 | 验证标准 |
|:---:|------|------|
| S3.1 | I3 Cell: Phase A-G 全部在图内 | GRAPH_MODE 编译, 不 OOM |
| S3.2 | Phase F: P_old 全量 INT8 Assign | 跑 1 步后 P_old 非零, delta norms 非零 (Read-before-Write OK) |
| S3.3 | 注册 quant/scales/idx buffers 到 FaF | register_tasks + set_step_ptr 无错 |
| S3.4 | 单步 E2E | HBM buffer ptr 非零, delta norms 与 CPU 参考对比 |
| S3.5 | 50 步训练 | I3 overhead **< 5%** vs baseline, PMU 验证 Vector 增量 |
| S3.6 | 全恢复验证 | FULL + delta chain → NRMSE median **< 0.05** |

---

## 六、论文实验路线图 🔲

### E0: sink_size 扫描 (前置)

扫描 sink_size = 1, 2, 5, 10, 20, 50, 100。产出 sink_size vs 吞吐曲线 → 确定最优值 + 证明 FaF 必要性。

### E1: I1 SPDK 基准 (6 实验)

| # | 实验 | 产出 |
|:---:|------|------|
| E1.1 | Raw 读写 BW | write/read BW, profiling 时间分解 |
| E1.2 | Pipeline Depth 扩展性 | depth vs BW 曲线 |
| E1.3 | Chunk Size 扩展性 | chunk vs BW 曲线 |
| E1.4 | SPDK vs 内核 NVMe | BW + CPU 利用率对比 |
| E1.5 | 多 Rank 扩展性 | 聚合 BW, 带宽争抢 |
| E1.6 | 数据完整性 | 3.1GB 逐字节校验 0 错误 |

### E2: I2 FaF 监听器 (7 实验)

| # | 实验 | 产出 |
|:---:|------|------|
| E2.1 | 轮询开销基准 | B1(纯) vs B2(+线程) vs B3(+SPDK) 步时 |
| E2.2 | 检测延迟分布 | mean/p50/p99 + 直方图 |
| E2.3 | 写盘完成时序 | Gantt 时间线: delta write vs optimizer |
| E2.4 | 同步 vs 异步 | sync_meta_io vs write_batch vs FaF |
| E2.5 | 最优 sink_size 下 per-step 触发 | 证明解耦 |
| E2.6 | 100 步一致性 | 0 遗漏 / 0 false positive |
| E2.7 | 线程资源开销 | CPU < 0.1%, 内存 < 10MB |

### E3: I3 增量管线 (8 实验)

| # | 实验 | 产出 |
|:---:|------|------|
| E3.1 | Delta norms 正确性 | GE FP16 vs CPU FP64 |
| E3.2 | INT8 量化精度 | 双路径 per-element 误差 |
| E3.3 | I3 步时 overhead | 50 步 mean±std |
| E3.4 | PMU: Baseline vs I3 | Cube/Vector 利用率 |
| E3.5 | Top-K 灵敏度扫描 | 压缩比 vs NRMSE tradeoff |
| E3.6 | 100 步恢复验证 | FULL + delta chain → NRMSE vs T |
| E3.7 | P_old Read-before-Write | GE 图排序验证 |
| E3.8 | 跨模型 (LLaMA-160M) | 重复 E3.1+E3.3+E3.6 |

### 执行顺序

```
E0 (sink_size 扫描) ← 最先
  └→ E1 (SPDK 基准) ─────────── 可并行 ─┐
     E2 (FaF) ────── 依赖 Step 3 ───────┤
     E3 (I3)  ────── 依赖 Step 3 ───────┘
```

### 实验脚本位置 (规划中，目录待创建)

```
experiments/
  step3/                   Step 3 全路径打通 (待实现)
  e0_sink_sweep/           论文实验 E0 (待实现)
  e1_spdk/                 论文实验 E1.1-E1.6 (待实现)
  e2_faf/                  论文实验 E2.1-E2.7 (待实现)
  e3_i3/                   论文实验 E3.1-E3.8 (待实现)
```

---

## 七、实现路径总览

```
Phase A: 服务器验证 (2026-06-27) 🔲
  └→ Phase B: I3 Step 3 全路径打通 🔲
      └→ Phase C: 论文实验 E0→E1/E2/E3 🔲
```
