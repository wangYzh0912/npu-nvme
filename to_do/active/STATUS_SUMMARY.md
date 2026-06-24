# NPU-NVMe 增量检查点 — 项目进度总结 (2026-06-24 更新)

> **sink=True, GRAPH_MODE** 是必须遵守的约束条件。
>
> **正式实现计划**: 见 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
>
> **架构路线图**: 见 `C:\Users\wyf\.claude\plans\p-old-npu-delta-top-k-cpu-delightful-minsky.md`

---

## 2026-06-24 更新：代码重构完成

### Phase 0a: Python 模块拆分 ✅

`python/direct_checkpoint.py` 从 1525 行单体拆分为 8 个子模块：

| 模块 | 职责 |
|------|------|
| `disk_layout.py` | 裸盘偏移常量 (零依赖) |
| `c_bindings.py` | libnpu_nvme.so ctypes 绑定 |
| `chunk_helpers.py` | build_chunks / build_chunks_host / build_ctypes_arrays |
| `delta_protocol.py` | pack/unpack/apply_delta + FileDeltaWriter |
| `noop_init.py` | NoOpInitializer 快速初始化 |
| `training_cell.py` | ProbeTrainOneStepCell |
| `_legacy_compat.py` | WaitProbe 遗留 (DEPRECATED) |
| `direct_checkpoint.py` | DirectCheckpoint 类 (~800 行) + 重新导出 |

**关键修复**:
- Delta I/O: `delta_save()` 改用 `build_chunks_host` + `write_batch_host` (解除 64MB 限制)
- SPDK 大页泄漏: `close()` 默认值 + `_spdk_initialized` 守卫
- export_model.py: 写路径修正为 `write_batch_host` (之前误用 `write_batch` 传 host 指针)
- 工具脚本: 磁盘常量统一从 `disk_layout` import (消除 4 文件间的重复)

### Phase 0b: to_do/ 整理 ✅

- 创建 `active/` (3 个活跃文件) + `archive/` (17 个归档文件)
- 删除过时补丁 (`C_refactor.patch`)

### Phase 0c: 实验样板提取 ✅

- 创建 `experiments/common.py`: `make_gpt2xl_training`, `setup_faf_checkpointing`, `make_ckpt`, `StepTimer`, `EpochTimer`, `init_env`

### 此前修复 (2026-06-24 上午)

- `fire_and_forget.py`: 删除重复 `on_train_epoch_begin`
- `direct_checkpoint.py`: async_thread 死代码删除, 指针获取统一到 `get_dev_ptr()`, MS API 导入时检测, SPDK 大页泄漏修复
- `profiler.py`: 新建统一 profiling 模块 (SpdkProfiler)

### 冗余审计结果

| 层 | 可消除行数 | 状态 |
|------|:---:|:---:|
| C (`src/`) | ~200 | 🔲 待执行 |
| Python (`python/`) | ~100 | ✅ 已通过模块拆分消除 |
| 实验 (`experiments/`) | ~900–1,050 | 🟡 `common.py` 已创建, 脚本迁移待执行 |

---

## 此前状态 (2026-06-19)

**刚完成**: Step 1 (三阶段基准) + Step 2 (图内量化 demo) + Step 2b (100 步断点续传验证)

**当前任务**: 🔲 **Step 3 — 全路径打通** (或 Delta → Ring Buffer C 层迁移)

---

## Step 1: 基准性能测试 ✅ (2026-06-17)

| 指标 | 值 |
|------|:---:|
| GPT-2 XL 步时 (GRAPH_MODE, sink=1, 50步) | **468.3 ms** (±25.8, COV 5.5%) |
| 编译时间 | **150.9 s** (model_init 42.4s + cell_build 58.8s) |
| GE kernel instances | **132,170** (AIC=10,944, AIV=113,005, AICPU=0) |
| HBM 占用 (训练后) | **35%** (~22.9 GB / 64 GB) |
| Model size FP16 | **3.12 GB** (772 params) |
| Cube MAC 利用率 | **55.7%** |
| Vector ALU 利用率 | **12.4%** (idle **87.6%**) |
| SPDK FULL ckpt 写入 BW | **4412 MB/s** (2.90 GB in 674ms) |

## Step 2: 图内量化可行验证 ✅ (2026-06-18)

- 跨层 batched GE: 290 params → concat → [2969, 512K] → Sub/Mul/ReduceSum → TopK → Gather → INT8 quant
- ~14 个新增 GE ops，全部落在 AI_VECTOR_CORE，Cube 无争抢
- Top-K 在图内: `ops.TopK(sorted=True)` ✅ GRAPH_MODE 编译成功
- INT8 量化 per-element 误差 ±1 bin
- Delta write: 159 MB in 45ms (3350 MB/s)

## Step 2b: 断点续传验证 ✅ (2026-06-18)

- 100 步 GRAPH_MODE 训练, 每步 delta + NRMSE 验证
- NRMSE drift: **+3.1e-3/step** (top-10% block 选择)
- 增量 NRMSE 方案避免了 380GB OOM
