# P0-U8: 项目文档整理与下一步开发规划 (Jun 8, 2026)

## 一、FaF 名称含义

**FaF = Fire-and-Forget**

源自军事术语 "fire-and-forget"（发射后不管），指导弹发射后就不再需要跟踪指导的自主攻击模式。

在我们的系统中，FaF 指：
- **NPU "开火"（fire）**：训练步中仅递增 step_counter，不等待，不阻塞，继续下一组计算
- **CPU/SPDK "接管"（forget）**：C 层 listener 线程自行检测 step_counter 变化，独立发起 SPDK 异步写盘
- NPU 不关心、不等待、也不知道写盘何时完成。这是一个完全解耦的异步架构

这与之前的 WaitProbe 方案形成鲜明对比——WaitProbe 方案中 NPU 必须在 WaitProbe kernel 中自旋等待 CPU 写完 I/O 并更新 flag，是同步握手模式。

---

## 二、项目文档全景

### 2.1 已有文档分类

**根目录（5 个）**：

| 文档 | 类型 | 说明 |
|------|------|------|
| `README.md` | 接口文档 | 项目简介 + API 说明（npu_nvme_transfer 纯传输） |
| `DEVELOPMENT_PLAN.md` | 规划 | 完整论文路线图（~540行）：3 个创新点、时间线、风控 |
| `EXPERIMENT_REPORT.md` | 实验 | 完整实验报告：基线 A-F + SPDK + V1-V4 + Q1-Q3 |
| `BASELINE_REPORT.md` | 实验 | 基线对比报告：SPDK 715ms vs 传统 2513ms |
| `FEASIBILITY_ASSESSMENT.md` | 分析 | I1/I2/I3 可行性评估，含风险矩阵和 P0 重排 |

**to_do/ 目录（9 个）**：

| 文档 | 类型 | 说明 |
|------|------|------|
| `STATUS_SUMMARY.md` | ⭐ 最新 | **P0-U6/U7 完整总结**：FaF 架构 + SPDK overhead root cause |
| `STATUS.md` | 旧 | 早期进展记录（Jun 7 之前） |
| `P0_U5_unified_rerun.md` | 旧 | P0-U5 统一实验重跑 ✅ |
| `P0_U6_fire_and_forget.md` | 开发日志 | FaF 开发全过程：v1→v2→v3 方案演进 + Bug 修复 |
| `P0_U6_analysis_and_plan.md` | 规划 | FaF 中期分析 |
| `P0_U6_benchmark_report.md` | 实验 | 性能基准报告（含 sink_size 对比） |
| `P0_U6_phase2_plan.md` | 规划 | FaF Phase 2 规划 |
| `P0_U7_spdk_overhead.md` | 开发日志 | **SPDK overhead 调查全过程**：实验矩阵、perf、strace、root cause |
| `P0_U7_summary.md` | ⭐ 最终 | **P0-U7 最终结论**：初始化顺序 root cause + workaround |

**doc/ 目录（1 个）**：

| 文档 | 说明 |
|------|------|
| `doc/midterm_report.md` | 硕士论文中期报告 |

### 2.2 文档清理计划

需归档/标记：

| 文档 | 操作 | 理由 |
|------|------|------|
| `to_do/STATUS.md` | → 移到 `to_do/archive/` | 旧版，已被 STATUS_SUMMARY.md 取代 |
| `to_do/P0_U6_analysis_and_plan.md` | → 归档 | 中间分析，结论已并入 P0_U6_fire_and_forget.md |
| `to_do/P0_U6_phase2_plan.md` | → 归档 | Phase 2 完成，结论已并入 STATUS_SUMMARY.md |
| `to_do/P0_U6_benchmark_report.md` | 保留 | 有独立价值的性能数据 |
| `to_do/P0_U6_fire_and_forget.md` | 保留 | 完整开发日志（历史价值） |
| `to_do/P0_U7_spdk_overhead.md` | 保留 | 完整 debug 过程日志（历史价值） |
| `to_do/P0_U7_summary.md` | 保留，移至根目录 | P0-U7 final，适合在根目录查找 |

---

## 三、开发全貌回顾

### 3.1 时间线

```
Week 1-2 (5月底-6月初):  基础设施
  ├── SPDK 集成 + 编译 + NVMe attach
  ├── DirectCheckpoint 管理类（裸盘布局 + meta + superblock）
  ├── 基线对比实验 A-F（SPDK 4380MB/s vs XFS 1954MB/s）
  └── WaitProbe AICPU kernel 原型

Week 3 (6月5日):          P0-U5 统一实验
  ├── 基线、SPDK端到端、Cell开销隔离、算子微观
  └── WaitProbe overhead 分析（0.53ms 同步延迟）

Week 3-4 (6月6-7日):     P0-U6 Fire-and-Forget ← 核心密集期
  ├── v1: AICPU TriggerProbe+WaitProbe → GE RTLD_LOCAL 失败
  ├── v2: C层轮询 expected → sink=TRUE Parameter 不可见 失败
  ├── v3: C层轮询 step_counter → ✅ 端到端通过
  ├── Bug: listener 初始化竞态 → 修复
  ├── Bug: Python probe_flag_ptr → C层自分配 + getter API
  └── 性能: sink=TRUE sink_size=30 → 1550ms（过大），sink_size=10 → 398ms

Week 4 (6月8日):         P0-U7 SPDK 开销定位
  ├── 发现 SPDK init → 304% training overhead
  ├── 系统性排查: CPU affinity, perf stat, TLB miss, cache miss, strace, cross-validation
  ├── 4次结论反复（高开销→低开销→高开销→...）
  ├── Root cause: MS C++ runtime 初始化顺序
  └── Workaround: warmup-before-SPDK（overhead → +10%）
```

### 3.2 关键数字

| 指标 | 值 | 来源 |
|------|-----|------|
| SPDK 写盘带宽 | 4380 MB/s | baseline_benchmark |
| SPDK 写盘延迟 (3.13GB) | 715ms | spdk_end_to_end |
| WaitProbe 同步延迟 | 0.53ms | operator_microbenchmarks |
| 纯训练步 (sink=F) | 388ms | E0/R0 |
| 纯训练步 (sink=T, s=10) | 398ms | R5 |
| **FaF 完整栈 (sink=T)** | **425ms** | **R6** |
| FaF SPDK overhead | +27ms (+7%) | R6-R5 |
| cell(x) 纯执行 | 369ms | cell_overhead |
| step_counter 图内开销 | +68ms (+17%) | sink=F |
| sink_size=10 vs 30 | 425ms vs 1550ms | R6 vs run10 |

---

## 四、下一步开发计划（按优先级）

### P0 — 立即可做（本周）

| # | 任务 | 说明 | 预计 |
|---|------|------|:---:|
| P0-1 | **warmup-before-SPDK 集成** | 在 `DirectCheckpoint.__init__` 中自动做 MS warmup | 2h |
| P0-2 | **FaF 安全校验修复** | Python 侧 probe_flag_ptr 正确更新（已写 getter API，未集成） | 1h |
| P0-3 | **spdk_end_to_end 重跑** | 清理环境后确认 1925ms 是否真实（可能也是初始化顺序问题） | 2h |
| P0-4 | **文档整理** | 按 Section 2.2 归档旧文档 | 30min |

### P1 — 本周内

| # | 任务 | 说明 | 预计 |
|---|------|------|:---:|
| P1-1 | **listener 轮询优化** | 降低 poll 频率（100μs→10ms），减少 ACL context 绑定 | 2h |
| P1-2 | **移除诊断日志** | listener poll# 日志移入 `#ifdef DIAGNOSTIC` | 30min |
| P1-3 | **FaF 完整 100 步测试** | sink=TRUE sink_size=10, CKPT_INTERVAL=10，100 步端到端 | 1h |

### P2 — Thesis (6月中-7月)

| # | 任务 | 说明 |
|---|------|------|
| P2-1 | 删除 dead code | WaitProbe/TriggerProbe AICPU kernel（从 master 移除） |
| P2-2 | 双层检查点架构 | delta 每 10 步 (FaF) + full 每 100 步 (需 barrier) |
| P2-3 | I3 Vector Engine 原型 | P_old FP8 存储 + 变化量检测 + 轮转选择器（见 DEVELOPMENT_PLAN.md） |
| P2-4 | 收敛性证明 | bounded staleness → 收敛率一致性（见 DEVELOPMENT_PLAN.md §4） |

### P3 — Paper writing (7-8月)

| # | 任务 |
|---|------|
| P3-1 | 实验评估章节统稿 |
| P3-2 | F5/F6 算子必要性论证 |
| P3-3 | 完整检查点 barrier 设计 |
| P3-4 | 答辩准备 |

---

## 五、文档更新操作

即将执行：
1. `to_do/STATUS.md` → `to_do/archive/STATUS_old.md`
2. `to_do/P0_U6_analysis_and_plan.md` → `to_do/archive/`
3. `to_do/P0_U6_phase2_plan.md` → `to_do/archive/`
4. `to_do/P0_U7_summary.md` → 移到根目录或合并到 `to_do/STATUS_SUMMARY.md`
5. 更新 `to_do/STATUS_SUMMARY.md` 作为单一真相来源 (single source of truth)
