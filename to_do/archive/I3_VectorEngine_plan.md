## 三、I3: 基于 Vector Engine 的图内增量管线

> **状态**：未完全实现。Step 2 demo 已验证 GE 图内 delta+quant+TopK 的可行性，但 P_old 语义和整管线方案已重构。本章描述的是目标设计，与当前代码存在差异。

### 3.1 核心问题与设计原则

增量检查点的核心计算——"参数变化了多少？哪些变化最大？"——如果在 CPU 上完成，需要每步将全部参数从 HBM 拷贝到 Host。Step 2b 实测此开销约 2.2 秒/步，是基线步时 468ms 的 4.7 倍。

**I3 核心思路**：将整个增量管线嵌入 GE 图内，利用 Vector Engine 闲置算力（87.6%）完成计算。所有数据全程驻留 HBM，无 Host 参与。

**关键设计决策（2026-06-23 确定）**：

| # | 决策 | 理由 |
|:---:|------|------|
| D-I3.1 | P_old 存为 INT8，驻留 HBM（~1.52 GB for XL） | FP16 模型的 50% 大小，±1 bin 量化误差不累积 |
| D-I3.2 | P_old 每步在 GE 图内全量更新 | 消除累积误差，绕过 MS 2.5 ScatterUpdate bug |
| D-I3.3 | P_old = quant(W_{N-1})，即上一步权重的完整 INT8 快照 | 每步 delta 基准是上一步完整参数，不是拼贴画 |
| D-I3.4 | delta = W_current - P_old（即对比上一步完整快照） | 量化误差 ±1 bin，不累积，每步独立 |
| D-I3.5 | GE 图内双量化路径：输出路径（Top-K→SPDK）+ P_old 路径（全量→Assign） | 一次前向执行，完成全部增量检查点计算 |

### 3.2 GE 图内数据流（单步完整伪代码）

```
construct(*inputs):
  ═══════════════════════════════════════════════════════════════
  Phase A: 标准训练（框架黑盒，不可修改）
  ═══════════════════════════════════════════════════════════════
  loss, grads = self.grad_fn(*inputs)         # forward + backward
  opt_res = self.optimizer(grads)             # optimizer → W_post_opt 在 HBM
  loss = ops.Depend(loss, opt_res)

  ═══════════════════════════════════════════════════════════════
  Phase B: 跨层参数聚合（将所有大参数 concat 为统一块矩阵）
  ═══════════════════════════════════════════════════════════════
  # 遍历 290 个大参数，逐个 Cast(FP16) → Reshape(1D) → Concat
  flat_parts = []
  for i in range(n_large_params):
      pv = ops.Cast(param[i], fp16) if needed
      flat_parts.append(ops.Reshape(pv, (nelem[i],)))
  all_flat = ops.Concat(tuple(flat_parts))    # [total_elems] ≈ 1.52B FP16
  all_flat = ops.Pad(all_flat, (0, pad_amt))  # 零填充到 BLOCK_SIZE 整数倍
  AllBlocks = ops.Reshape(all_flat, (nb, bs)) # [2969, 512K] FP16

  ═══════════════════════════════════════════════════════════════
  Phase C: Delta Norms（读 P_old，计算每块的 L2 偏离）
  ═══════════════════════════════════════════════════════════════
  # P_old 是上一步权重的 INT8 快照（2969 blocks），Cast 到 FP16 用于比较
  P_old_2d = ops.Reshape(self.P_old_int8, (nb, bs))
  P_old_fp16 = ops.Cast(P_old_2d, fp16)

  deltas = ops.Sub(AllBlocks, P_old_fp16)     # [nb, bs]  ← W_N - quant(W_{N-1})
  delta_sq = ops.Mul(deltas, deltas)           # 每个元素都有 ±1 bin 参考误差，不积累
  norms_fp16 = ops.ReduceSum(delta_sq, 1)      # [nb] FP16

  ═══════════════════════════════════════════════════════════════
  Phase D: Top-K 选择
  ═══════════════════════════════════════════════════════════════
  norms_fp32 = ops.Cast(norms_fp16, fp32)      # FP32 for TopK stability
  _, top_indices = ops.TopK(sorted=True)(norms_fp32, k)  # [k] I32

  ═══════════════════════════════════════════════════════════════
  Phase E: 输出路径 — Top-K 选中块的 INT8 量化
  ═══════════════════════════════════════════════════════════════
  selected = ops.Gather(AllBlocks, top_indices, 0)  # [k, bs] FP16
  # 量化: Abs → ReduceMax → scale → Div → Round → Clip → Cast(INT8)
  selected_fp32 = ops.Cast(selected, fp32)
  scales_out = ops.Div(ops.ReduceMax(ops.Abs(selected_fp32), 1), 127.0)
  quant_out = ops.Cast(ops.Clip(ops.Round(
      ops.Div(selected_fp32, ops.Reshape(scales_out, (k, 1)))),
      -128, 127), int8)                        # [k, bs] INT8

  ═══════════════════════════════════════════════════════════════
  Phase F: P_old 更新路径 — 所有块的 INT8 量化（替换旧 P_old）
  ═══════════════════════════════════════════════════════════════
  AllBlocks_fp32 = ops.Cast(AllBlocks, fp32)            # [nb, bs] FP32
  scales_all = ops.Div(ops.ReduceMax(ops.Abs(AllBlocks_fp32), 1), 127.0)
  quant_all = ops.Cast(ops.Clip(ops.Round(
      ops.Div(AllBlocks_fp32, ops.Reshape(scales_all, (nb, 1)))),
      -128, 127), int8)                                 # [nb, bs] INT8
  self.P_old_int8 = ops.Assign(self.P_old_int8,
      ops.Reshape(quant_all, (nb * bs,)))              # 整体覆盖，不是 ScatterUpdate！

  ═══════════════════════════════════════════════════════════════
  Phase G: Assign 输出缓冲区 + step_counter
  ═══════════════════════════════════════════════════════════════
  self.quant_buf = ops.Assign(self.quant_buf, ops.Reshape(quant_out, (k*bs,)))
  self.scale_buf = ops.Assign(self.scale_buf, scales_out)
  self.idx_buf   = ops.Assign(self.idx_buf, top_indices)
  self.step_counter = ops.AssignAdd(self.step_counter, one)

  # 确保 buffer writes 被 GE 编译器保留，不被 DCE 消除
  loss = ops.Depend(loss, self.quant_buf)
  loss = ops.Depend(loss, self.scale_buf)
  loss = ops.Depend(loss, self.idx_buf)
  loss = ops.Depend(loss, self.P_old_int8)
  return loss
```

### 3.3 三步连续数据流

以 GPT-2 XL（2969 blocks, k=296, BLOCK_SIZE=512K）为例：

```
═══════════════════════════════════════════════════════════════════════
STEP 1
═══════════════════════════════════════════════════════════════════════

GE 图执行前:
  HBM 状态: W_init (随机初始化, FP16, 3.12 GB)
            P_old_int8 = zeros(2969×512K) ≈ 1.52 GB (通过 Parameter.init 置零)
            quant_buf = zeros(296×512K) ≈ 148 MB
            scale_buf = zeros(296) ≈ 1.2 KB
            idx_buf = zeros(296) ≈ 1.2 KB

GE 图内:
  Phase A: forward(batch_1) → backward → optimizer → W_1 (HBM 上原地更新)
  Phase B: Concat(W_1 所有参数) → AllBlocks [2969, 512K] FP16
  Phase C: P_old_fp16 = Cast(P_old_int8, fp16)  ← 当前为全零
           deltas = AllBlocks - 0 = W_1        ← 首次，delta = 权重本身
           norms = ReduceSum(deltas²)          ← norms ≈ 权重的 L2 范数平方
  Phase D: indices = TopK(norms, 296)          ← 选中幅值最大的 296 blocks
  Phase E: quant_out = INT8_quant(AllBlocks[indices])
           → quant_buf = [296×512K] INT8
  Phase F: P_old_int8 = INT8_quant(AllBlocks)  ← W_1 的完整 INT8 快照
           → 1.52 GB, 全量覆盖（不再是拼贴画！）
  Phase G: step_counter += 1

GE 图结束后 (FaF listener):
  检测 step_counter 变化 → 读 quant_buf → SPDK write_batch → NVMe delta slot
  Delta frame: ~148 MB INT8 + header → NVMe

恢复状态:
  P_old_int8 = quant(W_1)  ← 一步完整快照, ±1 bin 精度
  可用于下一步 delta 计算

═══════════════════════════════════════════════════════════════════════
STEP 2
═══════════════════════════════════════════════════════════════════════

GE 图执行前:
  P_old_int8 = quant(W_1)  ← 上一步完整权重的 INT8 快照 (来自 Phase F)

GE 图内:
  Phase A: forward(batch_2) → backward → optimizer → W_2
  Phase B: AllBlocks = Concat(W_2) [2969, 512K] FP16
  Phase C: P_old_fp16 = Cast(P_old_int8, fp16)  ← dequant(quant(W_1))
           deltas = W_2 - P_old_fp16 ≈ W_2 - W_1  ← 差值含 ±1 bin 参考误差
           norms = ReduceSum(deltas²)
  Phase D: indices = TopK(norms, 296)
           → 选中 296 个"从上一步以来变化最大"的 block
  Phase E: quant_out = INT8_quant(AllBlocks[indices])
           → 这些 block 在 W_2 中的实际值（不是 delta 值！）
  Phase F: P_old_int8 = INT8_quant(AllBlocks)  ← W_2 的完整快照，替换旧的

  ✓ 关键: Phase C 的 P_old 读取在 Phase F 的 P_old 写入之前
          GE 编译器按图拓扑排序保证 Read-before-Write

恢复时 (从 FULL ckpt W_0 + delta chain):
  W_recovered_step2 = W_0 + dequant(delta_1) + dequant(delta_2)
  其中 delta_i 包含 top-K 选中的 block 值（覆盖写）
  ±1 bin 误差来自 P_old 的 INT8 参考，不累积

═══════════════════════════════════════════════════════════════════════
STEP 3 (及之后)
═══════════════════════════════════════════════════════════════════════

  P_old_int8 = quant(W_2)  ← 每步都是上一步的完整快照
  deltas = W_3 - quant(W_2)
  → norms 精度始终为 ±1 bin（不随步数增长）
  → Top-K 选择始终在正确的基础上进行
  → 预期 NRMSE drift 显著低于旧方案（累积拼贴画 +3.1e-3/step）
```

### 3.4 模块设计

新设计简化为 **5 个模块**（去掉旧的 M5 P_old 状态管理）：

```
┌──────────────────────────────────────────────────────────────────┐
│                    I3 系统架构 (修订版)                            │
│                                                                  │
│  M1: 参数分析与块布局       M2: I3 GE Cell（含完整 construct）    │
│  · analyze_model()          · Phase A-G 全部在图内                │
│  · BLOCK_SIZE=512K          · 双量化路径: 输出 + P_old            │
│  · 大/小参数分类            · GRAPH_MODE + sink=TRUE 适配         │
│                                                                  │
│  M3: INT8 量化 (per-block absmax)   M4: Top-K 选择               │
│  · 两个调用点: 输出路径 +          · ops.TopK(sorted=True)        │
│    P_old 路径复用同一逻辑          · k = top_k_frac × nb          │
│  · ~10 ops per call               · CPU fallback 保留             │
│                                                                  │
│  M5: HBM 缓冲区管理 (P_old + 输出 buffers)                       │
│  · P_old_int8: [nb×bs] INT8, ~1.52 GB                            │
│  · quant_buf: [k×bs] INT8,  ~148 MB                              │
│  · scale_buf: [k] FP32,     ~1.2 KB                              │
│  · idx_buf:   [k] I32,      ~1.2 KB                              │
│  · 全部 HBM Parameter → get_dev_ptr() 暴露给 C 层                │
└──────────────────────────────────────────────────────────────────┘
```

#### M1: 参数分析与块布局（沿用 Step 2 demo 设计）

**实现参考**：`experiments/baselines/step2_demo/step2_demo.py:80-158`

**核心逻辑**：
- `analyze_model(model)` → 遍历 trainable_params()
- `SMALL_THRESHOLD = 10000`：区分大参数（分块）和小参数（全量保存）
- GPT-2 XL 输出：290 large params → 2969 blocks，482 small params
- `BLOCK_SIZE = 524288`（512K elements = 1MB FP16 = 512KB INT8）

**⚠️ 待实现**：小参数的 delta 处理策略——小参数（<10000 elements）不做分块检测，直接全量保存或跳过。需要在 `step2_demo.py` 的 `analyze_model` 基础上增加小参数收集逻辑。

#### M2: I3 GE Cell 构造（需从 Step 2 demo 重构）

**实现参考**：`experiments/baselines/step2_demo/step2_demo.py:279-398`

**变化点 vs Step 2 demo**：

| 组件 | Step 2 demo (旧) | 新设计 |
|------|------|------|
| P_old 更新 | ❌ 未实现（注释掉） | Phase F: 全量 INT8 quant → Assign |
| P_old 更新方式 | ScatterUpdate（有 bug） | Assign(RESHAPE(quant_all)) — 无 bug |
| GE ops 总数 | ~16（不含 P_old） | ~30（含 P_old 全量量化） |
| Host 侧逻辑 | 需要 callback 更新 P_old | **不需要！** P_old 在图内完成 |
| 依赖 MS 版本 | 需等待 MS 2.6 fix | 不依赖，全程用 Assign |

**新增 Phase F 的 ops 分解**（~10 ops）：
```
Cast(FP16→FP32) → Abs → ReduceMax(axis=1) → Div(/127) → Reshape([nb,1])
→ Div(block/scale) → Round → Clip(-128,127) → Cast(INT8) → Reshape(flat) → Assign
```

**GE 编译器 Read-before-Write 保证**：
- Phase C 读取 P_old_int8（做 delta norms）
- Phase F 写入 P_old_int8（更新为当前权重快照）
- 因为 Phase F 不产生 Phase C 的输入依赖（AllBlocks 直接来自模型参数），图拓扑保证了旧 P_old 在 Phase C 被读取后才会被 Phase F 覆盖
- **⚠️ 需要在实现后做实验验证**：运行 1 步后检查 P_old 非全零（已更新）且 delta norms 非零（读取的是旧 P_old）

#### M3: INT8 量化引擎（复用 Step 2 demo 逻辑，增加第二个调用点）

**实现参考**：`experiments/baselines/step2_demo/step2_demo.py:358-370`

**两个调用点**：
1. **输出路径**（Phase E）：对 top-K 选中块做量化 → `quant_out [k, bs] INT8`
2. **P_old 路径**（Phase F）：对所有块做量化 → `quant_all [nb, bs] INT8`

两者使用完全相同的量化逻辑（per-block absmax），只是输入矩阵的行数不同（k vs nb）。**建议封装为独立函数 `quantize_blocks(x, bs)` 以避免代码重复。**

**量化语义**：
- 保存的是 block 的**值**（不是 delta），恢复时直接覆盖对应 block
- 恢复时：`W_recovered = W_full_ckpt + Σ dequant(selected_blocks_i)`
- 每步数据量 = k × bs × 1B = 296 × 512K × 1B ≈ 148 MB（top-10%）

#### M4: Top-K 选择（沿用 Step 2 demo）

**实现参考**：`experiments/baselines/step2_demo/step2_demo.py:350`

**主方案**：GE 图内 `ops.TopK(sorted=True)(norms_fp32, k)` — V2.1 ✅ 已编译通过

**k 值计算**：
```python
total_nb = model_info['total_nb']  # 2969 for GPT-2 XL
TOP_K_FRAC = 0.10
k = max(1, int(total_nb * TOP_K_FRAC))  # 296
```

**⚠️ 注意**：P_old 从累积拼贴画改为每步完整快照后，norms 的精确度提升（不再有历史累积量化误差），Top-K 重叠率预期改善。

#### M5: HBM 缓冲区管理（从旧 M6 简化而来）

**四个 HBM Parameter**，全部在 GE Cell `__init__` 中创建：

| 名称 | Shape | dtype | 大小 | 用途 |
|------|------|------|:---:|------|
| `P_old_int8` | `[nb × bs]` | int8 | ~1.52 GB | 上一步权重的完整 INT8 快照 |
| `quant_buf` | `[k × bs]` | int8 | ~148 MB | 输出：Top-K 选中块的量化值 |
| `scale_buf` | `[k]` | float32 | ~1.2 KB | 输出：每块的量化 scale |
| `idx_buf` | `[k]` | int32 | ~1.2 KB | 输出：选中的 block 索引 |

**HBM 占用合计**：1.52 GB + 0.148 GB ≈ **1.67 GB**（占 64 GB 总量的 2.6%）

**C 层访问**：通过 `get_dev_ptr()` 获取 device pointer → `npu_nvme_set_step_ptr()` 注册给 FaF listener → `aclrtMemcpy` 直接读 HBM

### 3.5 关键风险与验证策略

| 风险 | 概率 | 验证方法 | 缓解 |
|------|:---:|------|------|
| GE 编译器不保证 Read-before-Write（Phase C 读 P_old vs Phase F 写 P_old） | 低 | 运行 1 步后检查：P_old 非全零 AND delta norms 非零 | 如果乱序：用 `ops.Depend` 显式约束 P_old read 在 write 之前 |
| Phase F 全量量化导致步时显著增加 | 低 | E3.3: baseline vs I3 步时对比 | 全部在 Vector(87.6% idle)，~10 ops / 1.52B elem |
| P_old INT8 1.52 GB 在 GRAPH_MODE 编译时 OOM | 低 | 先编译再测试 | GPT-2 XL 训练后 HBM 35%, +1.67 GB → ~38% |
| GRAPH_MODE 下 ops.Assign 对大 Parameter 的行为异常 | 中 | 单步 Assign 验证 + 读回 | 如果 Assign 不支持 >1GB Parameter：改为原地覆盖（`ops.Assign` 的 MS 语义确认） |
| FP16 delta norms 精度影响 Top-K（边界块排序） | 低 | E3.1: 与 CPU FP64 norms 对比 Top-K 重叠率 | 历史重叠率 86.8%，预期改善 |

### 3.6 实现路径

**Step 3.1: 修改 Step2Cell**
- 基于 `step2_demo.py` 的 `Step2Cell` 类
- 添加 Phase F（P_old 全量量化 + Assign）
- 移除旧的 Phase G 注释块（ScatterUpdate 注释）
- 验证 GRAPH_MODE 编译

**Step 3.2: Read-before-Write 验证**
- 单步执行，检查 P_old 非零且 delta norms 合理
- 如果乱序：插入 `ops.Depend(P_old_write, P_old_read_result)`

**Step 3.3: 集成到 FaF 路径**
- 将 Step2Cell 的 `construct()` 中 `step_counter` 赋值接上 I2 的 listener
- C 层 listener 读取 `quant_buf` device pointer → SPDK write

**Step 3.4: 端到端恢复验证**
- I1 SPDK 写入 FULL ckpt + delta chain 到 NVMe
- 恢复路径：`DirectCheckpoint.recover()` 从 NVMe 读回 → 重建权重
- NRMSE vs Oracle 对比

**Step 3.5: 实验数据采集**
- 执行 E3.1~E3.8 全部实验

### 3.7 Motivation 论证结构（论文 §3.4 建议）

```
§3.4.1 增量检查点的计算挑战
  ① 为什么不能用 CPU？（3.12 GB × 每步 asnumpy = 2.2s → 4.7× 开销）
  ② 为什么必须嵌入图内？（即时性、零拷贝、与训练同步）
  ③ 为什么 Vector Engine？（87.6% 闲置，不争抢 Cube）

§3.4.2 跨层 Batched GE Ops 与双量化路径设计
  ① 跨层 Concat → [nb, BLOCK_SIZE]（O(1) subgraph launch）
  ② 输出路径：Top-K 选中块的 INT8 量化（供 SPDK 写盘）
  ③ P_old 路径：全量 INT8 量化（供下一轮 delta 参考）
  ④ GE ops 总数 ~30，占比 0.02%（vs baseline 132,000）

§3.4.3 P_old 语义与累积误差消除
  ① 旧方案（拼贴画）的分析：累积量化误差 +3.1e-3/step
  ② 新方案：P_old = quant(W_{N-1})，每步独立 ±1 bin 误差
  ③ 三步数据流示例（§3.4.3 核心图）
  ④ HBM 代价：1.52 GB INT8（模型 FP16 的 50%）

§3.4.4 图内 INT8 量化与 Top-K 选择
  ① Per-block absmax 量化精度（±1 bin 验证）
  ② Top-K 在 GE 图的编译通过性
  ③ FP16 vs FP32 norm 精度对 Top-K 排序的影响

§3.4.5 Vector Engine 利用论证
  ① 训练中的 Vector/Cube PMU 基线（Step 1b）
  ② I3 ops 的执行单元归属验证（E3.4: msprof）
  ③ I3 不增加 Cube 负载
```

### 3.8 I3 Evaluation 实验设计

| 编号 | 实验 | 验证什么 | 状态 |
|:---:|------|------|:---:|
| **E3.1** | Delta Norms 正确性 | GE FP16 norms vs CPU FP64（新 P_old 语义下） | 🔲 |
| **E3.2** | INT8 量化精度 | Per-element 量化误差（输出路径 + P_old 路径双验证） | 🔲 |
| **E3.3** | I3 步时 Overhead | ~30 ops 对训练步时的影响 | 🔲 |
| **E3.4** | PMU: Baseline vs I3 | Cube/Vector 利用率对比 | 🔲 |
| **E3.5** | Top-K 灵敏度扫描 | 压缩比 vs NRMSE tradeoff 曲线 | 🔲 |
| **E3.6** | 100步恢复验证 | FULL + delta chain → NRMSE vs T（预期显著优于旧方案） | 🔲 |
| **E3.7** | P_old Read-before-Write 正确性 | Phase C 读旧 P_old vs Phase F 写新 P_old 的图排序 | 🔲 |
| **E3.8** | 跨模型验证 | LLaMA-160M 重复关键实验 | 🔲 |

> 注意：旧的 E3.7（P_old 更新时序 — Host callback）已废弃。新 E3.7 验证的是 GE 图内 Read-before-Write 顺序。

---

#### E3.1: Delta Norms 正确性

**问题**：新 P_old 语义下（P_old = quant(W_{N-1}) 完整快照），GE 图内 FP16 delta norms 与 CPU FP64 参考的一致性？

**方法**：同 Step 2 V2.2 方法论，但在新的 P_old 更新语义下重跑：
```
1. Step N: GE 图执行 → norms_N (after Phase F 更新 P_old)
2. CPU: 用 W_N × dequant(P_old_int8_N) 计算 FP64 norms 作为参考
   (注意: P_old_int8_N 是 W_{N-1} 的 INT8 量化快照)
3. 比较: rel_err = |ge_norms - cpu_norms| / (cpu_norms + 1e-8)
4. Top-K 索引重叠率 (与 CPU FP64 参考对比)
```

**预期**：norms 精度与旧方案类似（FP16 vs FP64），但 Top-K 质量更高（P_old 不含累积误差）

---

#### E3.2: INT8 量化精度（双路径验证）

**问题**：输出路径和 P_old 路径的 INT8 量化是否都正确？

**方法**：
```
Path A (输出): GE 图 quant_out vs CPU numpy quant（同 Step 2 V2.3）
Path B (P_old): GE 图 P_old_int8 vs CPU numpy quant(AllBlocks)
  两者使用完全相同的量化逻辑，只需验证输入正确
```

---

#### E3.3: I3 步时 Overhead

**方法**：
```
条件 A (Baseline): 纯训练 (Step 1a OracleCell)
条件 B (I3 full):  训练 + I3 完整管线 (Phase A-G, ~30 ops)

sink_size 使用 E0 最优值, GPT-2 XL
各 50 步, 测量 mean±std 步时, 首步编译时间
```

**预期**：I3 overhead < 3%（30 ops vs 132,000 baseline ops，全部在 Vector）

---

#### E3.4: PMU: Baseline vs I3

同旧方案，msprof 对比 Cube/Vector 利用率。关键验证：I3 新增 ops 全部落在 AI_VECTOR_CORE。

---

#### E3.5: Top-K 灵敏度扫描

同旧方案。在新 P_old 语义下预期 NRMSE drift 显著降低，tradeoff 曲线整体下移。

---

#### E3.6: 100步恢复验证

同旧方案，但在新 P_old 语义下。**核心对比**：新方案的 NRMSE drift 应远低于旧方案的 +3.1e-3/step。

---

#### E3.7: P_old Read-before-Write 正确性（新实验）

**问题**：GE 编译器是否保证 Phase C 读取的 P_old 是旧值（被 Phase F Assign 更新之前的值）？

**方法**：
```
Step 1: 
  P_old 初始化为全零
  执行 GE 图 → 读 norms → asnumpy()
  读 P_old → asnumpy()

验证:
  ① P_old 非全零（Phase F 的 Assign 已执行）
  ② norms 非零（Phase C 的 delta 计算读到的是旧 P_old = 全零，而非新 P_old = W_1）
     → 如果 norms ≈ 0，说明 Phase F 在 Phase C 之前执行了（乱序！）
     → 如果 norms 合理（≈ W_1 的 block L2 范数），说明 Read-before-Write 正确

Step 2:
  再执行一步
  P_old 应与 W_1 的 INT8 量化值一致（Phase C 时读到的旧 P_old = Step 1 写入的）
  norms 应反映 W_2 - quant(W_1)
```

**论文数据需求**：单步验证通过即可，不产生图表

---

#### E3.8: 跨模型验证

同旧方案。LLaMA-160M 上重复 E3.1 + E3.3 + E3.6（简化版）

---

