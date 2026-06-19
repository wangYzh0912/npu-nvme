# Phase 5 下一步行动 (2026-06-17 更新 — 架构分析后)

> **关键发现**: 当前所有 Phase 5 脚本的增量管线在 **CPU 侧执行**（asnumpy → numpy ops），未达到"图内 Vector Engine + NPU DMA 直写"的设计目标。详见 [DELTA_PIPELINE_CPU_VS_NPU_ANALYSIS.md](DELTA_PIPELINE_CPU_VS_NPU_ANALYSIS.md)。

---

## 当前状态

| 组件 | 设计目标位置 | 当前实际位置 | 差距 |
|------|:---:|:---:|------|
| Delta norms (L2) | Vector Engine | S6: Vector Engine ✅ / S4/S7: CPU ❌ | 部分完成 |
| INT8 量化 | Vector Engine | **CPU (numpy)** ❌ | 未实现图内量化 |
| Top-K 选择 | Host (可接受) | CPU (可接受) | ✅ |
| P_old 存储 | HBM (Parameter) | CPU (PoldStore dict) ❌ | 未实现在线 P_old |
| 量化数据写入 | HBM→NVMe DMA | **CPU buffer→NVMe** ❌ | sync_meta_io 而非 DMA |
| 恢复 (recovery) | NVMe→HBM DMA | 混合 (pickle+SPDK) ⚠️ | 未从 FULL ckpt 开始 |

## 重新定义 P0: 完成图内增量管线

### S8: 图内 INT8 量化 (取代 CPU asnumpy 方式)

**目标**: 将 INT8 量化从 `on_train_epoch_end` callback 中的 CPU numpy 搬入 GE 图

**技术路线**:
```
construct() 内:
  forward → backward
  → delta norms (已有 GE ops)              ← Step A
  → absmax scale per block                ← Step B (新增)
  → INT8 quant: round→clip→cast           ← Step C (新增)
  → Assign 到 output Parameter             ← Step D (新增)
  → optimizer

callback 只做:
  - 读取 output Parameter 的量化 INT8 数据
  - 打包为 delta frame
  - 调用 delta_save()
```

**实现挑战**:
- 图内无法做"按 Top-K 选择性量化"（需要 GE Select 或图外决策）
- **方案 A**: 对所有 blocks 量化 → callback 只取 Top-K（浪费 Vector 算力但简单）
- **方案 B**: 在 GE 图内用 norms→mask→Select 实现条件量化

### S9: HBM→NVMe DMA 直写路径

**目标**: 量化数据不再经过 CPU buffer，直接 HBM DMA → NVMe

**需要在 C 层添加**: `npu_nvme_write_delta_device(ctx, slot_idx, dev_ptr, total_bytes)`

### 论文叙事

当前 CPU 侧原型的性能数据（asnumpy ~1500ms, delta compute ~2600ms）本身是有价值的论据:
> "CPU 侧方案每步需搬运全量参数经 PCIe→DRAM (249MB), 增加步时 ~2.6s (GPT-2 Small), 对 GPT-2 XL 预计增加 ~20s。这正是我们主张将计算搬入 Vector Engine 的动机。"
