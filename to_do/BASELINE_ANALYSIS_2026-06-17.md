# NPU-NVMe 项目系统性回顾与基准配置分析

> 日期: 2026-06-17 | 作者: Claude Opus 4.8  
> 基于完整代码仓库阅读 (python/, src/, experiments/, to_do/)  
> 当前开发基准: PHASE2B_DESIGN.md (已完成) → Phase 5 S5-S7 (进行中)

---

## 1. 项目全景架构

### 1.1 三大创新点 (I1/I2/I3)

```
增量检查点需求 (近两年热点: LowDiff/SIC/MoEtion/GoCkpt)
  │
  ├─ I1: SPDK 用户态 NVMe 驱动 ──────→ 写入 4380MB/s, 加载 ~6800MB/s
  │   内核 FS 不可行: write()=syscall, 高频小块写开销累积
  │   SPDK: lock-free SQ, 确定性轮询延迟, per-core 无锁
  │   实现: src/npu_nvme.c (1622行) + src/npu_nvme_transfer.c + python/direct_checkpoint.py (1362行)
  │
  ├─ I2: WaitProbe/TriggerProbe AICPU 同步原语 ──────→ 0.53ms 同步延迟
  │   提供 backward→optimizer 之间的确定性插入点
  │   实现: src/aicpu_probe.cc (54行) + ProbeTrainOneStepCell (direct_checkpoint.py)
  │
  └─ I3: Vector Engine 原生增量管线 ──────→ batched GE ops, 零步时开销
     用 Vector 闲置算力 (92.6% idle) 做:变化检测→选择→量化
     实现: experiments/baselines/ (Phase 1a→Phase 5)
```

### 1.2 系统组件依赖

```
Python 层:
  python/direct_checkpoint.py     — DirectCheckpoint 主类 + delta I/O + recovery
  python/i3_delta_writer.py       — 二进制帧打包/解包 + FileDeltaWriter fallback
  python/transfer_api.py           — ACL 传输 API
  python/format_npu_disk.py        — 裸盘格式化

C 层:
  src/npu_nvme.c (1622行)          — SPDK init + DMA + 大页管理 + Delta API
  src/npu_nvme_transfer.c          — NPU→NVMe 传输核心
  src/aicpu_probe.cc (54行)         — AICPU WaitProbe kernel
  include/npu_nvme.h (157行)       — 公共头文件

实验脚本:
  experiments/baselines/           — Phase 1a→Phase 5 全部实验脚本
  experiments/output/              — JSON 实验结果
```

---

## 2. 当前测试配置基准

### 2.1 硬件环境

| 项目 | 值 |
|------|:---:|
| NPU | Ascend 910B (单卡, device_id=1) |
| NVMe SSD | PCIe 0000:83:00.0 |
| SPDK 写入理论带宽 | 4380 MB/s (pipeline_depth=8, chunk=4MB) |
| SPDK 加载理论带宽 | ~6800 MB/s |
| HBM | 64 GB |
| sudo 密码 | `CGCL_2025_#$` |

### 2.2 软件栈

| 项目 | 值 |
|------|------|
| MindSpore | 2.5 (pip install 在 /root/miniconda3/envs/ms_2.5/) |
| CANN | /usr/local/Ascend/ascend-toolkit/latest/ |
| SPDK | third_party/spdk (git submodule) |
| Python 解释器 | /root/miniconda3/envs/ms_2.5/bin/python |
| 当前运行模式 | **sink=TRUE, GRAPH_MODE** (硬约束) |

### 2.3 当前关键配置参数

```python
# DirectCheckpoint 默认配置
pipeline_depth:  8  # (原基准4380MB/s) 注意: S4用了4→BW降到2252MB/s
chunk_size:      4 * 1024 * 1024   # 4MB
keep_last_n:     100  # 最多保留N个FULL ckpt
slot_size_gb:    50   # 每slot大小 (GPT-2 XL ~3.1GB/full ckpt)

# 增量管线 (I3) 配置
BLOCK_SIZE:      524288  # 512K elements = 1MB FP16
top_k:           0.10    # 10% blocks per step
SMALL_THRESHOLD: 10000   # params <10K elements = "小参数" (每步全保存)

# Delta 盘布局
delta_slot_size_mb:  256   # 每槽256MB
delta_slot_count:    128   # 128槽环形
# Delta 区域总大小: 128×256MB = 32GB
```

---

## 3. 当前实验数据全景

### 3.1 Phase 5 S5: SPDK 裸盘带宽 ✅

| 指标 | 值 | 状态 |
|------|:---:|:---:|
| pipeline_depth | 8 | 恢复正确配置 |
| 数据大小 | 3.2 GB (GPT-2 XL scale) | |
| 写入时间 | 893ms | |
| 实测带宽 | **3674 MB/s** | 理论值83.5% ✅ |
| 目标带宽 | 4380 MB/s | 差距 ~16% |

**S5结论**: pipeline_depth=8 修复有效，3674MB/s 接近理论极限，可接受。

### 3.2 Phase 1a/2b/3/4a: 增量管线核心验证 ✅

| 阶段 | 关键结果 | 状态 |
|------|---------|:---:|
| Phase 1a | Vector/Cube 隔离验证, 步时+1.5% | ✅ |
| Phase 1b | 67% Vector idle 是平台常数 | ✅ |
| Phase 2b S1 | Block delta GE: 104 ops, GRAPH_MODE ✅ | ✅ |
| Phase 2b S234 | INT8 quant + Top-K: 134 ops, 91.6ms/step | ✅ |
| Phase 3 | Batched ops = 零 overhead | ✅ |
| Phase 4a v7 | 离线 delta: NRMSE median=0.017, **max=0.094** | ✅ 理想 |

**Phase 4a v7 是关键的参考基准**: 在离线 (host numpy) 条件下，M=10 轮转，20步恢复的 NRMSE max=0.094 (9.4%), median=0.017 (1.7%) — 这是**当前最优结果**，可作为可信数据的质量目标。

### 3.3 Phase 5 S4/S6+S7: E2E GRAPH_MODE 在线测试 ⚠️

| 指标 | S4 (pipeline_depth=4) | S6+S7 (pipeline_depth=8) | 问题 |
|------|:---:|:---:|------|
| FULL write BW | 未达目标 (BW~2252 MB/s) | 2339 MB/s | ⚠️ 低于 S5 裸盘3674 |
| Avg delta write | 434ms | 404ms | ⚠️ 过高 (应为~50ms量级) |
| NRMSE median | **0.0106** (1.06%) | **0.0108** (1.08%) | ✅ 小参数准确 |
| NRMSE max | **3.44** (344%) | **3.35** (335%) | ❌ 致命错误 |
| NRMSE p95 | **2.53** (253%) | **2.52** (251%) | ❌ 大量块错误 |
| Hash match | **false** | **false** | ❌ 不一致 |
| Compression ratio | 16.6x | 16.3x | ✅ |

---

## 4. 核心问题诊断

### 4.1 根本原因：恢复路径从随机初始值而非 FULL 检查点开始

这是当前 **最关键的 bug**。两个 E2E 脚本（S4 和 S6+S7）的恢复路径都存在同一问题：

```python
# phase5_s6_s7_e2e.py 第303-304行 — 恢复路径
ms.set_seed(42); ms.common.set_seed(42)
recover_model = AutoModel.from_config(cfg)
w = {p.name: p.value().asnumpy().copy() for p in recover_model.trainable_params()}
                                                    ↑
                                    用的是随机初始化的权重，而非 step_0 FULL ckpt！
```

虽然 seed=42 保证随机初始化与训练时相同，但**增量帧只覆盖 top-10% 变化最大的 block**：
- 每步覆盖约 17-32 blocks + 122 小参数（全保存）
- 20步后每个 block 平均被保存 ~2次
- 某些 block **从未被保存** → 保留随机初始值 → NRMSE 极高
- 有些 block 最后一次保存是十几步前 → 严重过期

**证据链**：
1. NRMSE median=0.0108 说明每次全量保存的小参数（n_small=122/196）正确
2. NRMSE max=3.35 说明大参数 block 中有从未被覆盖的
3. Phase 4a v7 的 M=10 轮转确保每层每 10 步必被覆盖 → max NRMSE=0.094

### 4.2 增量语义混淆

当前代码在 `build_block_delta()` 中：
```python
dn = float(np.sum((bd - po).astype(np.float64)**2))  # 用于 Top-K 排序
...
# 但 apply_delta_patches 中写入的是:
fp32 = i8.astype(np.float32) * scale  # 当前步的绝对量化值，不是 delta！
wv[start:end] = fp32[:end-start]       # 直接覆盖
```

命名上叫 "delta"，实际上写的是 **绝对参数值的量化近似**。这使问题更严重——如果写入真实 delta (Δ = W_cur - P_old)，即使部分 block 没被覆盖也只是保留旧值而非随机值。

### 4.3 SPDK Delta 写盘延迟异常

```
Avg delta write: 404ms (S6+S7) vs 预期 ~50ms
```

可能原因：
- `delta_save()` 中调用了 `wait_for_io_completion()` 导致阻塞等待前序 FULL ckpt 完成
- `sync_meta_io` 对 delta frame 的同步写入未优化
- delta frame 实际数据量 ~15MB/step，但 404ms 对应 BW=37MB/s — 远低于 SPDK 能力

### 4.4 GPT-2 XL 的 E1 测试严重异常

```
I3 overhead: 32,213ms (7895%) — 完全不可用
```

这是因为 XL 的 GE 图编译为 `PYNATIVE_MODE`（非 GRAPH_MODE），导致所有 delta ops 都以 Python 开销运行。需在 `GRAPH_MODE` 下重新测试。

---

## 5. 推荐的基准测试配置

### 5.1 统一基准配置 (Baseline Profile)

```python
# ═══════════════════════════════════════════════════════════════
# 后续所有实验统一使用的基准配置
# ═══════════════════════════════════════════════════════════════

BASELINE_CONFIG = {
    # ── SPDK I/O ──
    "pipeline_depth": 8,            # 恢复 4380MB/s 写入带宽
    "chunk_size": 4 * 1024 * 1024,  # 4MB DMA 块
    "nvme_addr": "0000:83:00.0",
    "npu_device_id": 1,
    "spdk_shm_id_range": [70, 99],  # 每次测试用不同 shm_id 避免冲突
    
    # ── 模型 ──
    "model": "GPT-2 Small",         # 12L/768d, 196 params, 249MB FP16
    "seq_length": 1024,
    "batch_size": 1,
    
    # ── 增量管线 ──
    "block_size": 524288,           # 512K elements = 1MB FP16
    "top_k": 0.10,                  # 10% blocks/layer/step
    "small_threshold": 10000,       # 小参数阈值 (elements)
    
    # ── Delta 盘 ──
    "delta_slot_size_mb": 256,
    "delta_slot_count": 128,
    
    # ── 训练 ──
    "mode": "GRAPH_MODE",           # 硬约束
    "sink_mode": True,             # 硬约束
    "sink_size": 1,                # 1=每步触发 callback
    "learning_rate": 1e-5,
    "optimizer": "AdamWeightDecay",
    
    # ── 验证 ──
    "test_steps": [10, 20, 50],    # 短/中/长 delta 链
    "nrmse_threshold": {"median": 0.05, "max": 0.10},
}
```

### 5.2 正确恢复流程 (Correct Recovery Path)

```python
def correct_recovery(target_step: int):
    """正确的增量恢复流程"""
    # Step 1: 从 SPDK 加载最近 FULL ckpt (必须是 step_0 或 step_N)
    base_step = find_nearest_full(target_step)
    ckpt.load(model, step=base_step)  # SPDK DMA → device
    
    # Step 2: 读增量链
    chain = ckpt.delta_load_chain(base_step, target_step)
    
    # Step 3: 从已恢复的 device 权重拉取 host numpy
    host_weights = {p.name: p.value().asnumpy().copy()
                    for p in model.trainable_params()}
    
    # Step 4: Apply delta patches 到 FULL ckpt 的基础之上
    for sid, blocks, smalls in chain:
        host_weights = apply_delta_patches(host_weights, blocks, smalls, block_size)
    
    # Step 5: 写回 device
    for p in model.trainable_params():
        p.set_data(Tensor(host_weights[p.name].astype(np.float16), ms.float16))
    
    return host_weights
```

当前代码缺失 Step 1。`direct_checkpoint.py` 中的 `recover()` 方法（第1311-1359行）已正确实现了此流程，但 S4 和 S6+S7 实验脚本**未使用该方法**，而是在实验代码中手动重写了有缺陷的恢复逻辑。

### 5.3 可信数据产出路线

要产出 "可信的基准数据"，按以下顺序修复和重测：

```
Step A: 修复 recovery 路径
  → 使用 DirectCheckpoint.recover() 或等价正确流程
  → 验证 NRMSE max < 0.10, hash match=true
  → 预计耗时: ~2h

Step B: 重新运行 S7 E2E (20步)
  → pipeline_depth=8, GRAPH_MODE, sink_size=1
  → 正确恢复路径
  → 产出可信的 E2E 数据

Step C: 扩展测试矩阵
  → 20/50/100 步 — 验证 delta 链扩展性
  → Top-K 扫描: K∈{5,10,20}% — 确定安全 K 范围
  → E9: 跨层全局 Top-K — 替代 per-layer Top-K

Step D: GPT-2 XL GRAPH_MODE 复测
  → 修复 E1 的 PYNATIVE_MODE 问题
  → 在 GRAPH_MODE 下验证 XL 的 GE 图可编译
```

---

## 6. 当前数据质量评估

### 6.1 可信数据 (可用于论文)

| 数据 | 来源 | 可信度 |
|------|------|:---:|
| SPDK 裸盘 BW 3674 MB/s (depth=8) | S5 | **高** — 独立测试, 可复现 |
| Batched GE ops 零 overhead | Phase 3 | **高** — 多次验证 |
| Vector/Cube 隔离, 67% idle | Phase 1b | **高** — PMU 数据支撑 |
| Block delta GE compile OK | Phase 2b S1/S234 | **高** — GRAPH_MODE 验证 |
| Phase 4a v7: NRMSE max=0.094 (离线) | Phase 4a | **中高** — 离线, 非在线 |
| FULL+delta I/O 链路打通 (SPDK) | S3/S4/S6 | **中** — I/O 链路通但恢复逻辑有 bug |
| INT8 量化精度 rel_err=6.5e-2 | Phase 3 | **高** — 统计充分 |

### 6.2 不可信数据 (需修复后重测)

| 数据 | 问题 |
|------|------|
| S4/S6+S7 NRMSE max=3.35 | 恢复路径从随机值开始 |
| S4/S6+S7 Hash mismatch | 同上 |
| S4 FULL BW~2252 MB/s | pipeline_depth=4, 已修复为 8 |
| E1 XL batched overhead=7895% | PYNATIVE_MODE, 需 GRAPH_MODE 重测 |
| Delta write 404ms avg | 可能含 wait_for_io 开销 |

---

## 7. 立即行动计划

### P0 (今天): 修复恢复路径 + 跑通可信 E2E

1. **修改 `phase5_s6_s7_e2e.py`** 的恢复部分，从 step_0 FULL ckpt (从 NVMe 加载) 开始恢复
2. **或直接改用** `DirectCheckpoint.recover()` 方法 (已验证可用)
3. 跑 20 步验证: NRMSE median<0.05, max<0.10, hash match=true

### P1 (1-2天): 扩展基准数据

1. 多步测试: 20/50/100 步
2. Top-K 灵敏度: 5%/10%/20%
3. E9 跨层全局 Top-K (替代当前 per-layer Top-K)

### P2 (本周): 系统集成

1. FaF listener per-step 触发 delta write (替代当前 callback 触发)
2. 时间戳验证 (delta write 在 optimizer 更新前完成)
3. GPT-2 XL GRAPH_MODE 全量测试

---

## 8. 关键命令速查

```bash
# sudo 密码
echo "CGCL_2025_#$

# 构建 C 层
cd /home/user7/npu-nvme && bash build.sh

# 环境
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export PYTHONPATH=/home/user7/npu-nvme/python:$PYTHONPATH

# S5: SPDK 裸盘 BW 测试
echo "CGCL_2025_#$" | sudo -S bash -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && \
  /root/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/baselines/phase5_s5_spdk_raw_bw.py --device-id 1'

# S6+S7: E2E 全链路 (当前有 bug 版本 — 需修改后重跑)
echo "CGCL_2025_#$" | sudo -S bash -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && \
  /root/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/baselines/phase5_s6_s7_e2e.py --steps 20 --device-id 1'

# Git 提交
cd /home/user7/npu-nvme && git add -A && git commit -m "your message"
```
