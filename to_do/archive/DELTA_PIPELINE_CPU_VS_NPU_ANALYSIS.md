# Delta Pipeline 计算负载与数据流分析

> 日期: 2026-06-17 | 目标: 验证当前实现是否符合"图内 Vector Engine 空闲算力 + NPU→NVMe 直写"的设计逻辑

---

## 一、设计目标回顾

根据 [PHASE2B_DESIGN.md](PHASE2B_DESIGN.md) 和 [DEVELOPMENT_PLAN.md](../DEVELOPMENT_PLAN.md)，I3 增量管线的目标执行模型是:

```
训练步 (图内, GRAPH_MODE):
  forward → backward
  → delta_detection (batched GE ops, Vector Engine)  ← 图内, 与 Cube MatMul 并行
  → quantize_top_k (Vector Engine)                     ← 图内
  → optimizer update
  → DMA: HBM → Ring Buffer → SPDK NVMe write           ← C 层触发
```

**关键设计要求**:
- Delta detection/量化都在 NPU GE 图内的 Vector Engine 上执行
- 不通过 `asnumpy()` 将权重搬移到 CPU 侧做计算
- 量化结果直接从 NPU HBM DMA 到 NVMe

---

## 二、当前所有实验脚本的实际数据流

### 2.1 S4 `phase5_s4_e2e_single_card.py` — ❌ 全 CPU

```
训练 (GRAPH_MODE):
  图内: forward → backward → optimizer

Callback (每个 epoch_end, 每次 sink_size=1 触发):
  on_train_epoch_end:
    ┌─────────────────────────────────────────────────────────────┐
    │ ① SNAP: get_all_params_np(model)                          │
    │    → p.value().asnumpy().copy()  ← 全量 249MB HBM→CPU！    │
    │                                                             │
    │ ② DELTA: build_block_delta(true_w, pold, lid, ...)         │
    │    → true_w[name] 已是 numpy, 全部在 CPU 侧:                │
    │      - fp32 = true_w[name].astype(np.float32).flatten()     │
    │      - dn = np.sum((bd - po).astype(np.float64)**2)         │
    │      - q = np.clip(np.round(bd/sc), -128, 127).astype(int8) │
    │    ← 纯 CPU numpy 计算！                                    │
    │                                                             │
    │ ③ P_old: PoldStore (CPU dict) ← 纯 CPU                     │
    │                                                             │
    │ ④ WRITE: ckpt.delta_save(step, blocks, smalls)             │
    │    → pack_delta_frame (CPU) → sync_meta_io (SPDK write)    │
    │    ← 数据从 CPU buffer 写入 NVMe                            │
    └─────────────────────────────────────────────────────────────┘
```

**违反设计**:
- ❌ Delta detection (L2 norm) 在 CPU 做 — 应有 Vector Engine 执行
- ❌ INT8 量化在 CPU 做 — 应有 Vector Engine 执行
- ❌ `get_all_params_np` 每步搬运 249MB HBM→CPU — 违反了"零 PCIe 搬运"设计
- ❌ 量化数据从 CPU buffer → NVMe — 应走 NPU HBM → DMA → NVMe 路径

**测量到的延迟**: snap=~1470ms, compute=~2630ms — 主要是 asnumpy() 的 PCIe 搬运

### 2.2 S6 `phase5_s6_ge_delta.py` — ⚠️ 部分图内

```
训练 (GRAPH_MODE):
  DeltaDetectionCell.construct():
    forward → backward
    → batched delta norms: GE ops on Vector Engine ✅
        Reshape → Sub → Mul → ReduceSum → Assign(norms_param)
    → optimizer
  ← 图内只做了 delta norms！

Callback (每个 epoch_end):
  on_train_epoch_end:
    ┌─────────────────────────────────────────────────────────────┐
    │ ① SNAP: get_all_params_np(model)                          │
    │    → p.value().asnumpy().copy()  ← 全量 249MB HBM→CPU ❌   │
    │                                                             │
    │ ② NORMS: norms_arr = norms_param.value().asnumpy().copy() │
    │    → 只搬运 norms vector (~840 floats, 可忽略)             │
    │                                                             │
    │ ③ TOP-K: sorted(block_list, key=lambda x: -x[0])           │
    │    → CPU 排序 (acceptable, ~840 elements)                   │
    │                                                             │
    │ ④ QUANT: 对选中的 top-K blocks:                            │
    │    → fp32 = true_w[name].astype(np.float32).flatten()       │
    │    → sc = np.max(np.abs(bd)) / 127.0                       │
    │    → q = np.clip(np.round(bd/sc), -128, 127).astype(int8)  │
    │    ← 纯 CPU ❌  (应放在 GE 图中)                            │
    │                                                             │
    │ ⑤ WRITE: ckpt.delta_save() → CPU buffer → NVMe ❌          │
    └─────────────────────────────────────────────────────────────┘
```

**违反设计**:
- ✅ 图内 delta norms: 正确 (batched GE ops)
- ❌ INT8 量化: 在 CPU 做，需要先 asnumpy() 全量权重
- ❌ `get_all_params_np` 每步搬运全量参数
- ❌ 量化结果走 CPU→NVMe (sync_meta_io) 而非 NPU DMA

**值得注意**: `norms_param` 是一个非可训练的 Parameter，delta norms 被 ops.Assign 写入后，可以在 callback 中通过 asnumpy 读取 — 这部分是正确的设计模式。

### 2.3 S6+S7 `phase5_s6_s7_e2e.py` — ❌ 全 CPU

```
与 S4 完全相同的模式:
  TrainCell.construct(): forward → backward → optimizer  (无 delta ops)

Callback:
  get_all_params_np → build_block_delta(纯 CPU) → delta_save(CPU→NVMe)
```

**这个脚本没有做任何 GE delta detection！** 它的 `TrainCell.construct()` 只是纯 forward+backward+optimizer，delta 全在 callback 中 CPU 侧做。

### 2.4 Phase 4a v7 `phase4a_batched_i3.py` — ⚠️ 离线分析基准

```
训练 (GRAPH_MODE):
  TrainCell: forward → backward → optimizer  (无 delta)
  SnapCB: get_all_params_np  ← 全量搬运

训练结束后 (离线分析):
  对每个 snapshot:
    → build_block_delta (纯 CPU numpy)  ← 计算 delta + quant
    → reconstruct_v6 (纯 CPU)  ← 验证恢复精度
```

**这是离线验证管线**，不是在线 checkpointer。它只验证了 "如果我们在 host 侧正确做了 delta detection + Top-K + 量化，恢复精度是好的"。结果 NRMSE median=0.017, max=0.094 — 这是最佳可信数据，但它不是在线系统。

---

## 三、关键评估: asnumpy() 的使用

### 3.1 所有 asnumpy() 调用点

| 脚本 | 调用 | 数据量 | 频率 | 是否在关键路径 |
|------|------|:---:|:---:|:---:|
| S4 | `get_all_params_np` in callback | 249MB | 每步 | ✅ 关键路径 |
| S6 | `get_all_params_np` in callback | 249MB | 每步 | ✅ 关键路径 |
| S6 | `norms_param.value().asnumpy()` | ~840 floats | 每步 | ✅ 可接受 |
| S6+S7 | `get_all_params_np` in callback | 249MB | 每步 | ✅ 关键路径 |
| Phase 4a | `get_all_params_np` in callback | 249MB | 每步 | ✅ 离线 |

### 3.2 性能影响

```
asnumpy() 的实际成本 (GPT-2 Small 249MB FP16):
  - PCIe DMA: HBM → CPU DRAM
  - PCIe BW 理论: ~64 GB/s (PCIe 4.0 x16)
  - 实测: 249MB / ~1470ms ≈ 170 MB/s
  ← 远低于理论带宽，说明 Python 遍历+copy 是瓶颈
```

---

## 四、回答: 当前是否做到了"图内 Vector 算力 + NPU→NVMe 直写"？

**明确回答: 没有。**

所有 Phase 5 脚本 (S4, S6, S6+S7) 都依赖:

1. **每步 `asnumpy()` 全量权重搬运到 CPU** (~1500ms, 占步时 70%+)
2. **在 CPU 侧 (Python numpy) 做 INT8 量化**
3. **通过 CPU buffer → `npu_nvme_write_delta` (sync_meta_io) 写入 NVMe**

唯一部分在图内的操作是 **S6 的 delta norms 计算** (batched GE ops)，但量化步骤仍然在 CPU 做。

**本质上**: 这些是 **主机端增量检查点** (host-side incremental checkpointing) 而非真正的 **NPU 端增量管线** (NPU-side pipeline)。

---

## 五、要做成"图内 Vector Engine 增量量化 + NPU DMA 直写"需要什么？

### 5.1 需要在 GE 图内完成的步骤

```
construct() 内 (GRAPH_MODE, GE 图编译):
  forward → backward
  → [Step A] Batched delta norms (✅ 已有, S6 已验证)
  → [Step B] Top-K 选择 (需要在图内实现)
  → [Step C] INT8 量化: AbsMax(block)/127 → Round → Clip → Cast → int8
  → [Step D] 将量化结果写入 output Parameter (Assign)
  → optimizer

C 层 FaF listener:
  → poll step_counter
  → 读取 output Parameter 的设备地址
  → 直接 DMA: HBM → NVMe (不经过 CPU!)
```

### 5.2 Top-K 在图内的实现挑战

Top-K (选定 norms 最大的 K 个 block) 在 GE 图内不好做，因为它需要:
- 全局排序 (需要所有 block norms)
- 动态决定哪些 block 被量化

**可行方案**:
- **方案 A**: 固定阈值 — `if norm > threshold: quantize` (图内 Select 支持)
- **方案 B**: 不选 Top-K，而是对**所有** blocks 做量化，但只对 norms 最大的 K 个执行 Assign → output
- **方案 C**: 在 epoch_end callback 中做 Top-K 决策 (norms 数组小，asnumpy 可接受)，但图内**预量化所有 blocks**，callback 只决定取走哪些

### 5.3 关键架构决策

| 组件 | 图内 (GE/Vector) | Host callback | C FaF listener |
|------|:---:|:---:|:---:|
| Delta norms (L2) | ✅ | | |
| **Top-K 排序** | 困难 | ✅ (norms 小) | |
| **INT8 量化** | **必须** | ❌ (当前错误) | |
| P_old 更新 | ✅ (Assign) | | |
| SPIKE DMA write | | | ✅ |

---

## 六、当前状态的合理性和价值

### 当前实现仍然有价值的方面:

1. **Phase 4a v7 提供了可信数据基准**: NRMSE max=0.094 说明如果计算正确 (即使在 CPU 上)，增量管线的质量是好的
2. **S6 验证了 batched GE delta norms 可以在图内运行**: 这是图内计算的第一步
3. **`direct_checkpoint.py` 的 delta I/O 层 (delta_init/save/load) 是正确的**: 二进制格式、SPDK 读写、恢复流程
4. **每个脚本都提供了真实的步时和延迟测量**: 有助于说服论文读者 "CPU 侧方案性能不可接受"

### 需要明确告诉导师/读者:

> "当前 Phase 5 脚本是 **可行性验证原型**——增量管线的逻辑正确性已经验证 (Phase 4a v7, NRMSE max=0.094)，但计算执行位置 (CPU vs Vector Engine) 尚未优化。后续工作是将 INT8 量化搬入 GE 图。"

---

## 七、建议的下一步

### P0 (本周): 将 INT8 量化搬入 GE 图

1. 扩展 S6 的 `DeltaDetectionCell.construct()`: 在 delta norms 之后加入:
   - per-block absmax scale 计算
   - INT8 量化 (round→clip→cast)
   - 写入 output Parameters

2. 修改 callback: 只读取 output Parameters 中的量化数据 (已量化好的 INT8)
3. 移除 `get_all_params_np()` 从关键路径

### P1 (下周): 实现 HBM→NVMe 直写路径

1. C 层添加 `npu_nvme_write_delta_device()` 函数 — 直接从 HBM device pointer 读
2. Python 层传入量化数据所在的 HBM device pointer

### P2 (论文): 清晰区分原型与最终系统

- 原型阶段 (Phase 5): 逻辑正确性验证 (CPU 侧计算, 可接受的论文中间结果)
- 最终系统: Vector Engine 原生管线 (图内计算 + HBM→NVMe DMA)
