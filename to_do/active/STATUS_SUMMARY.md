# NPU-NVMe 项目进度 (2026-06-25)

> **下一步**: 修复 T4 ms.Model.train 形状问题 → T5 overhead → T6 恢复 → 论文实验

---

## 代码重构: 完成 ✅ (2026-06-24)

| 层 | 重构内容 | 效果 |
|------|------|------|
| Python | `direct_checkpoint.py` 拆分 1→8 模块 | 1525→1049 行, 消除重复常量/ctypes/协议 |
| Python | Delta I/O 迁移到 ring buffer pipeline | 解除 64MB 限制, 支持任意大小 delta frame |
| C | 删除 `write_delta`/`read_delta`，新增 `read_batch_host` | API 19 函数 |
| C | 内部头文件 + pipeline 统一 + profiling 统一 | 894 行 (重构后) |
| 实验 | `common.py` 4 共享函数 | 消除 ~900-1050 行样板 |
| 文档 | to_do/ 整理 | active/5 + archive/23 |

## 服务器验证: 完成 ✅ (2026-06-24)

| 测试 | 结果 | 说明 |
|------|:---:|------|
| C 编译 (with -DHAS_NPU) | ✅ | 2 warnings |
| Pure-logic tests | **10/10** | ring buffer, ALIGN_4K, constants |
| Hardware integration tests | **15/15** | init/cleanup/capacity/max_transfer/meta_io/delta |
| A.5 Python imports | ✅ | 全部模块导入成功 |
| A.6 FULL ckpt roundtrip | ✅ | save→load→cleanup 通过 |
| A.7 Delta >64MB roundtrip | ✅ | 150MB × 300 blocks 逐字节验证 |
| Step 1c SPDK BW | **3790 MB/s** | 2.90 GB / 784ms, pipeline_depth=8 |

**修复的 Bug**: verify_phaseA.sh import, chunk_size 类型(int→uint32_t), probe_cb 返回类型(int→bool), pipeline.h SPDK 前向声明, c_bindings 模块引用, delta read_batch_host(新 API)

## 研究进展

| 阶段 | 内容 | 核心结果 |
|:---:|------|------|
| 基准测试 ✅ | 训练 + PMU + SPDK BW | 步时 468ms, Cube 55.7%, Vector idle 87.6%, BW 4412→3790 MB/s |
| 图内量化 ✅ | 7 项验证 | INT8 ±1 bin, Top-K 86.8%, Delta 159MB/45ms |
| 断点续传 ✅ | 100 步 NRMSE 曲线 | NRMSE drift +3.1e-3/step, top-10% 覆盖不足 |
| 增量检查点 🔄 | DeltaTrainCell + FaF + E2E | 见下方 |

## 增量检查点模块 (新增)

| 模块 | 文件 | 状态 |
|------|------|:---:|
| DeltaTrainCell (7-phase GE pipeline) | `python/delta_cell.py` | ✅ |
| analyze_model_layers | `python/delta_cell.py` | ✅ |
| build_layout_for_delta | `python/direct_checkpoint.py` | ✅ |
| register_delta_tasks | `python/direct_checkpoint.py` | ✅ |
| setup_delta_faf | `experiments/common.py` | ✅ |
| make_delta_training | `experiments/common.py` | ✅ |
| E2E 测试 (T1-T6) | `experiments/delta_e2e/` | 🔄 |

## E2E 测试进展

| # | 测试 | 状态 | 结果 |
|:---:|------|:---:|------|
| T1 | GRAPH_MODE 编译 | ✅ | 3038 blocks, 1.59 GB delta_p_old, 无 OOM |
| T2 | 单步 E2E | ✅ | p_old_sum=3.2e10, quant_sum=3.2e9 均非零 |
| T3 | FaF 注册 | ✅ | 3 delta buffers 注册成功, step_counter + flag 有效 |
| T4 | 多步 FaF 触发 | 🔴 | 阻塞: `ms.Model.train()` 形状广播不匹配 |
| T5 | Overhead 对比 | 🔲 | 依赖 T4 |
| T6 | 恢复验证 | 🔲 | 依赖 T4 |

**T4 阻塞原因**: `ms.Model.train()` 框架包装层改变输入形状 — GPT-2 LM 模型内部将 `input_ids` 切片为 `seq_length-1`，导致 attention mask `[1,1024,1024]` 与 lower_triangle `[1,1025,1025]` 广播失败。T2 使用直接迭代 (`cell(*data)`) 正常。修复方向：T4 改直接迭代。

## 当前任务

| 优先级 | 任务 | 预计 |
|:---:|------|:---:|
| **P0** | ~~服务器验证~~ ✅ | — |
| **P0** | ~~DeltaTrainCell 模块~~ ✅ | — |
| **P1** | 修复 T4 (ms.Model.train → 直接迭代) | <1h |
| **P1** | T5 overhead (I3 ops vs baseline 步时) | 1h |
| **P1** | T6 恢复验证 (FULL + delta chain NRMSE) | 1h |
| **P2** | 论文实验 E0→E1/E2/E3 | 1-2w |

## 关键设计决策

| # | 决策 | 理由 |
|:---:|------|------|
| D1 | delta_p_old 全量 Assign (非 ScatterUpdate) | 绕开 MS 2.5 scatter bug |
| D2 | checkpoint_name_or_path="" | 避免 seq_len 与预训练权重不兼容 |
| D3 | seq_len=1025 | 匹配 gpt2_train_1025 mindrecord |
| D4 | 禁止开发标签 (I3/Step3/Phase) | STYLE_GUIDE.md §4.3 |

## 文档索引

| 文档 | 内容 |
|------|------|
| `DELTA_CHECKPOINT_DESIGN.md` | 增量检查点详细设计 |
| `ARCHITECTURE_ROADMAP.md` | 代码结构速览 |
| `IMPLEMENTATION_PLAN.md` | 实验路线图 |
| `CLAUDE_INSTRUCTIONS.md` | 服务器环境 + 会话恢复 |
| `STYLE_GUIDE.md` | 编码规范 |
