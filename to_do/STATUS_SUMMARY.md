# NPU-NVMe 增量检查点 — 项目进度总结 (2026-06-19 更新)

> **sink=True, GRAPH_MODE** 是必须遵守的约束条件。所有实验需在此条件下运行。
> 
> **正式实现计划**: 见 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — 三步计划 (Step 1 ✅ → Step 2 ✅ → Step 2b ✅ → Step 3 待开始)
>
> **会话恢复**: 见 [CLAUDE_INSTRUCTIONS.md](CLAUDE_INSTRUCTIONS.md) — 每次新建对话必须首先读取

---

## 零、当前状态 (2026-06-19)

**刚完成**: Step 1 (三阶段基准) + Step 2 (图内量化 demo) + Step 2b (100 步断点续传验证)

**当前任务**: 🔲 **Step 3 — 全路径打通** (或 Step 2b 补充: top-K 灵敏度扫描)

---

## Step 1: 基准性能测试 ✅ (2026-06-17)

### 1a: 纯训练基准
| 指标 | 值 |
|------|:---:|
| GPT-2 XL 步时 (GRAPH_MODE, sink=1, 50步) | **468.3 ms** (±25.8, COV 5.5%) |
| 编译时间 | **150.9 s** (model_init 42.4s + cell_build 58.8s) |
| GE kernel instances | **132,170** (AIC=10,944, AIV=113,005, AICPU=0) |
| HBM 占用 (训练后) | **35%** (~22.9 GB / 64 GB) |
| Model size FP16 | **3.12 GB** (772 params) |

### 1b: 设备级 PMU
| 指标 | 值 | 方法 |
|------|:---:|------|
| Cube MAC 利用率 | **55.7%** | msprof sample-based ArithmeticUtilization |
| Vector ALU 利用率 | **12.4%** | msprof sample-based ArithmeticUtilization |
| Vector idle | **87.6%** | 100% - Vector ALU util |

### 1c: SPDK FULL ckpt BW
| 指标 | 值 |
|------|:---:|
| 写入 BW (wall) | **4412 MB/s** (2.90 GB in 674ms) |
| 写入 BW (pipeline DMA) | **~52,000 MB/s** (8-buf overlap) |
| 配置 | pipeline_depth=8, chunk=4MB |

---

## Step 2: 图内量化可行验证 ✅ (2026-06-18)

### 设计
- 跨层 batched GE: 290 params → concat → [2969, 512K] → Sub/Mul/ReduceSum → TopK → Gather → INT8 quant
- ~14 个新增 GE ops，全部落在 AI_VECTOR_CORE，Cube 无争抢
- Top-K 在图内: `ops.TopK(sorted=True)` ✅ GRAPH_MODE 编译成功
- P_old 1.59 GB INT8 Parameter 驻留 HBM

### 7 项验证结果

| 编号 | 验证项 | 结果 | 详情 |
|:---:|------|:---:|------|
| V2.1 | GRAPH_MODE 编译 | ✅ | build=5.5s, 3038 blocks, 不 OOM |
| V2.2 | Delta norms 正确性 | ⚠️ 可接受 | Top-K index 重叠 86.8% (263/303), FP16 vs FP32 norm 精度差异导致边界块重排 |
| V2.3 | INT8 量化正确性 | ✅ | max_abs_diff=1.0 (1 INT8 bin), 对262个共同选中块的 per-element 验证 |
| V2.4 | P_old 图内更新 | ⚠️ 降级 | MS 2.5 `tensor_scatter_update` INT8 dtype 编译失败 → 移至 host callback |
| V2.5 | HBM output buffer ptr | ✅ | 4 个 buffer 全部非零 device pointer |
| V2.6 | 步时 overhead | ⚠️ 波动大 | 单步测量 403~1696ms (vs 468ms baseline), 需 Step 3 多步稳定测量 |
| V2.7 | HBM→NVMe delta write | ✅ | 159.4 MB in 45ms → 3350 MB/s (write_batch, 绕过 64MB sync_meta_io 限制) |

### 关键技术细节

**MS 2.5 GE scatter bug**: `ops.tensor_scatter_update()` 在 GRAPH_MODE 下处理 INT8 dtype 时触发 `infer_dtype() missing 3 required positional arguments`。这是 MS 2.5 已知 bug。P_old (INT8) 的 scatter 更新改为 host callback 执行——单个 step 的 scatter < 0.2ms，无感知开销。

**FP16 delta norm 精度**: GE 图内 delta 用 FP16 计算 norms, CPU 参考用 FP64。Top-K 边界块 (norms 值相近) 的排序存在 13.2% 差异。这不影响增量检查点语义——Top-K 选择的是"变化最大"的块，边界块的重排不改变累积偏移检测。

**SPDK 写通道**: `npu_nvme_write_delta` 内部使用 `sync_meta_io`，受 64MB DMA buffer 限制。实际 delta frame ~159MB (296 blocks × 512K INT8) 超出限制。改用 `npu_nvme_write_batch` (HBM→NVMe DMA, 无大小限制) 绕过。

## Step 2b: 断点续传验证 ✅ (2026-06-18)

### 实验设计
- 100 步 GRAPH_MODE 训练 (sink_size=1)
- 每步在 epoch_end callback 中:
  1. 保存 oracle 权重 → 立即计算 NRMSE vs 增量恢复权重 → **立即丢弃** (不存储 100 份 3.1GB 快照)
  2. 计算 CPU delta pipeline (W_post_opt vs P_old → Top-K → INT8 quant)
  3. 更新 recoery flat buffer 和 P_old
- 首次尝试存储 100 份 oracle 权重导致 380GB OOM → 改为增量 NRMSE 方案

### 结果

| 指标 | 值 | 判定 |
|------|:---:|:---:|
| Step 1 NRMSE median | **0.003** | 近乎完美恢复 |
| Step 10 NRMSE median | **0.04** | 初期偏离 |
| Step 100 NRMSE median | **0.76** | 累积偏离 |
| NRMSE drift trend | **+3.1e-3/step** | 线性增长 |
| 增量 NRMSE 方案 | ✅ | 避免了 380GB OOM |
| 绘图 | ✅ | NRMSE vs T + Loss 曲线 |

### 分析

NRMSE 增长是 top-10% 块选择的预期行为:
- 每步只更新 296/2969 (10%) 的 block
- 剩余 90% 的 block 保持接近 step_0 的值，与持续训练的 oracle 权重逐渐偏离
- 100 步后 NRMSE=0.76 意味着 top-10% 选择捕获了约 24% 的参数变化量
- top-K 越大 → 恢复精度越高 → 压缩比越低。这是核心 tradeoff

**下一步建议**: Step 2b 补充 top-K 灵敏度扫描 (5%/10%/20%/50%) 建立 NRMSE vs 压缩比曲线，作为论文核心数据点。

---

## 一、环境约束（关键）

| 约束 | 说明 |
|------|------|
| **sink=TRUE 必须** | 训练性能必需，sink=FALSE 的 per-step 开销不可接受 |
| **GRAPH_MODE** | 增量管线 delta ops 需 GE 编译进图 |
| **MS 2.5 已知限制** | `tensor_scatter_update` INT8 dtype GRAPH 编译失败; PYNATIVE `value_and_grad` + GPT-2 XL → NaN at step 2 |
| **AICPU kernel 不可注入** | GE 在 sink=TRUE 下通过 `dlopen(RTLD_LOCAL)` 加载，自定义 kernel 符号不可达 |
| **sink=TRUE callback 局限** | `on_train_step_end` 只在 epoch 首尾触发，中间步不可见 |
| **per-step 时间测量** | sink=TRUE 下只有 C 层 listener 的 step_counter 轮询可测 |

## 二、当前 checkpoint I/O 同步语义（理解）

### 2.1 关键时序约束

对于 sink=TRUE, GRAPH_MODE 下的检查点：

```
Epoch boundary:
  Step N-1 (last of epoch):
    optimizer update 完成
    → [epoch_end callback] → FULL ckpt save (阻塞训练)
  Step N (first of next epoch):
    forward + backward + optimizer
    → [epoch_end callback] → (next ckpt cycle)

Per-step (图内):
  forward → backward → delta_detection → optimizer → delta_write?
  
  但 delta_write 是 Host 侧操作，不能在 GRAPH_MODE 图内执行。
  Host 侧 callback 只在 epoch 边界触发。
```

### 2.2 当前 FaF (Fire-and-Forget) 方案回顾

FaF 的设计：step_counter 在图内自增，C 层 listener 线程异步轮询并触发 SPDK 写盘。**问题：SPDK 写盘与 next step optimizer 更新有时间重叠**（~500ms），导致检查点中最后 35% 的参数可能是"混合态"。

这是 FaF 的已知特性，对恢复训练影响很小（参数变化 1e-7~1e-6），但**对精确的增量检查点**来说，delta 必须记录确定的步状态。

### 2.3 增量检查点的两种可能路径

**路径 A: 全同步**（当前 S1-S4 实现走的路）
- FULL ckpt：epoch_end callback 中，`save()` → `wait_for_io_completion()` → 阻塞到写盘完成
- Delta：每步在 host 侧 numpy-leveler 做，做完 delta_save 后写盘
- 问题：per-step delta 在 GRAPH_MODE sink=TRUE 下无法 hook 到优化器更新后的瞬间
- S4 当前做了——在 graph 训练结束后、host numpy 从 snapshot 中取权重做 delta → **这是离线的、不与训练流交织的**

**路径 B: 图内 delta + 同步 delta 写**
- Delta detection 在图内（GE batched ops → AI_VECTOR_CORE）
- Delta block 选择 + 量化 + 写盘在 host 侧（需要 per-step 触发点）
- 如果用 epoch_end callback → 只能在 epoch 边界触发 delta flush → Delta 无法做到"per step"
- 如果用 FaF listener 触发 → delta 写盘可以 per-step，但与 optimizer 有时间重叠

### 2.4 用户确认的方向

根据当前对话中的反馈：
1. **sink=TRUE, GRAPH_MODE 必须**
2. **FULL ckpt 同步写在 epoch 之间**（阻塞训练不构成问题，它不干扰训练步）
3. **Delta 写盘必须在 optimizer 更新前完成**（保证确定性）
4. **Delta detection 可以编入图内**（batched GE ops），delta 写盘在 step 边界触发
5. **不需要物理同步原语**（如 WaitProbe），但**需要添加时间点统计**来验证 delta 写盘没有和 optimizer 重叠
6. **SPDK 带宽 ~2252 MB/s 需排查** — 原测试 4380 MB/s，可能原因：pipeline_depth 做了 4 而不是 8，或 chunk_size 不同

## 三、已完成模块 ✅

### 3.1 I1: SPDK 用户态 NVMe

| 模块 | 状态 | 关键数据 |
|------|:---:|------|
| 两段 DMA + 裸盘布局 | ✅ | 4412 MB/s (Step 1c 实测) |
| FaF listener + 轮询 | ✅ | 3.1GB 写盘 ~705ms |
| `src/npu_nvme.c` | ✅ | FaF + Delta API + 大页修复 |
| `python/direct_checkpoint.py` | ✅ | DirectCheckpoint + delta_init/save/load + recover |

### 3.2 I2: Fire-and-Forget 同步

| 配置 | per-step | 相对纯 MS |
|------|----------|----------|
| B1: 纯 MindSpore | **412ms** | 基准 |
| B2: 探针闲置 | **436ms** | +5.8% |
| B3: 完整 FaF (SPDK per-10-step) | **435ms** | +5.6% |

### 3.3 I3 Phase 1-4: 增量管线核心验证

| 阶段 | 关键发现 | 状态 |
|------|---------|:---:|
| Phase 1a | Vector/Cube 隔离, 步时+1.5% | ✅ |
| Phase 1b | 67% Vector idle 是平台常数 | ✅ |
| Phase 2a | Ascend C 4 条路径全部阻塞 | ✅ |
| Phase 2b | Per-param blocks + INT8 + Top-K | ✅ |
| Phase 3 | Batched ops = 零 overhead, GE 无上限, 瓶颈在 subgraph launch | ✅ |
| Phase 4 | 去掉轮转, 全层检测 | ✅ |

### 3.4 Phase 5: S1-S4 代码实现

| 模块 | 状态 | 文件 |
|------|:---:|------|
| Meta 格式 (FULL+DELTA) | ✅ | `direct_checkpoint.py:_commit_metadata` |
| delta_init/save/load | ✅ | `direct_checkpoint.py` delta_* 方法 |
| pack/unpack delta frame | ✅ | `python/i3_delta_writer.py` |
| C 层 Delta API | ✅ | `src/npu_nvme.c`: npu_nvme_delta_* |
| 大页自动修复 | ✅ | `src/npu_nvme.c`: ensure_hugepages() |
| SPDK Smoke Test | ✅ | `phase5_s3_spdk_smoke.py` |
| S4 E2E Test | ⚠️ | `phase5_s4_e2e_single_card.py` (FULL+delta 路径通, BW 待修) |

## 四、当前待解决问题

### 4.1 主计划推进 (P0)

**Step 3: 全路径打通** 是下一步主要工作。目标: 将 Step 2 的 GE 图内 delta+quant 与 SPDK delta write 集成到完整的训练+检查点系统中。

架构:
```
TrainCell.construct() (GE 图, GRAPH_MODE, sink=TRUE):
  forward → backward → batched delta+quant+topK → optimizer
  → Assign quant_buf / scale_buf / idx_buf (HBM)

C 层 FaF listener:
  poll step_counter → 检测步变化
  → 从 HBM (quant_buf/scales/idx_buf) DMA 量化数据 → NVMe delta slot
  → 更新 metadata

Epoch 边界 callback (on_train_epoch_end):
  → FULL ckpt 同步写入 (阻塞, SPDK DMA)
  → 只在 epoch 间发生, 不干扰训练步
```

关键验证:
- V3.1: Delta write 在下一个 optimizer 前完成
- V3.2: FULL ckpt 同步阻塞不干扰训练步
- V3.3: 恢复: FULL + delta chain → device → NRMSE + Loss vs Oracle
- V3.4: 50 步完整训练: baseline vs I3 步时对比
- V3.5: 50 步完整训练: baseline vs I3 loss 对比

### 4.2 已知技术问题

| 问题 | 状态 | 详情 |
|------|:---:|------|
| MS 2.5 `tensor_scatter_update` INT8 bug | ⚠️ | P_old 更新降级到 host callback，Step 3 沿用 |
| MS 2.5 PYNATIVE `value_and_grad` + XL → NaN | ⚠️ | GRAPH_MODE 稳定 (Step 1a, 2b 均已验证) |
| `delta_save` 64MB DMA buffer 限制 | ✅ | 改用 `write_batch` 绕过 (Step 2 V2.7) |
| 大页不足 (SPDK init 失败) | ✅ | `ensure_hugepages()` 自动处理 |
| NRMSE 累积 (top-10% 覆盖不足) | ⚠️ | Step 2b 已量化，top-K tradeoff 待扫描 |

### 4.3 待补充实验

| 实验 | 优先级 | 说明 |
|------|:---:|------|
| Top-K 灵敏度扫描 (5%/10%/20%/50%) | 高 | 建立 NRMSE vs 压缩比曲线 |
| Epoch 内 delta write 时序验证 | 高 | 确认 delta write + optimizer 无重叠 |
| FULL ckpt + delta chain 恢复端到端 | 高 | 从 NVMe 恢复完整权重 |
| LLaMA-160M 跨模型验证 | 中 | 证明方法通用性 |
| GPU/CPU 离线对比 | 低 | 量化误差 vs 全精度基线的准确界限 |

## 五、下一步工作

### P0: Step 3 全路径打通

1. **集成 I3 GE cell + FaF listener**: 将 Step 2 demo cell 的 delta ops 嵌入 Step 1a 的 OracleCell
2. **FaF listener 实现 per-step delta write**: C 层监听 step_counter → 读取 HBM quant_buf → SPDK write_batch
3. **Epoch 边界 FULL ckpt**: epoch_end callback → 同步 SPDK DMA 写入完整权重
4. **恢复验证**: FULL ckpt + delta chain → NRMSE vs Oracle (Step 2b 方法论)
5. **时间线统计**: delta write begin/end vs optimizer 时间戳

### P1: 补充实验

1. **Top-K 灵敏度扫描**: 5%/10%/20%/50% → NRMSE vs 压缩比
2. **E2E 50 步训练**: baseline vs I3 步时 + loss 对比

## 六、关键命令

```bash
sudo密码: CGCL_2025_#$
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
PYTHON=/root/miniconda3/envs/ms_2.5/bin/python
REPO=/home/user7/npu-nvme

# Step 1a: 纯训练基准
echo "CGCL_2025_#$" | sudo -S bash -c "source .../setenv.bash && $PYTHON $REPO/experiments/baselines/benchmark/step1_benchmark.py --steps 50 --device-id 1"

# Step 1b: 设备级 PMU
bash $REPO/experiments/baselines/benchmark/_run_1b.sh 12 1

# Step 1c: SPDK FULL ckpt BW
bash $REPO/experiments/baselines/benchmark/_run_1c.sh 1

# Step 2: 图内量化 demo
bash $REPO/experiments/baselines/step2_demo/_run.sh 2 1

# Step 2b: 断点续传验证
bash $REPO/experiments/baselines/step2b_recovery_validation/_run.sh 100 1

# Git
cd $REPO && sudo chown -R user7:user7 .git/objects && git add -A && git commit -m "..."
```
