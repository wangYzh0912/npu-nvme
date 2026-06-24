# NPU-NVMe 项目进度 (2026-06-27)

> **下一步**: Phase A 服务器验证 → Phase B I3 Step 3 → Phase C 论文实验

---

## 代码重构: 完成 ✅ (2026-06-24)

| 层 | 重构内容 | 效果 |
|------|------|------|
| Python | `direct_checkpoint.py` 拆分 1→8 模块 | 1525→800 行, 消除重复常量/ctypes/协议 |
| Python | Delta I/O 迁移到 ring buffer pipeline | 解除 64MB 限制, 支持任意大小 delta frame |
| Python | 6 bug 修复 | lib=None, 悬垂指针, sync no-op 等 |
| C | 删除 `write_delta`/`read_delta` | −106 行, API 18→16 函数 |
| C | 内部头文件 + pipeline 统一 + profilng 统一 | −554 行, `npu_nvme.c` 1354→894 |
| 实验 | `common.py` 5 共享函数 | 消除 ~900-1050 行样板 |
| 文档 | to_do/ 整理 | active/4 + archive/17 |

## 研究进展

| Step | 内容 | 核心结果 |
|:---:|------|------|
| 1 ✅ | 基准测试 | 步时 468ms, Cube 55.7%, Vector idle 87.6%, BW 4412 MB/s |
| 2 ✅ | 图内量化 | INT8 ±1 bin, Top-K 86.8%, Delta 159MB/45ms |
| 2b ✅ | 断点续传 | 100 步 NRMSE drift +3.1e-3/step |

## 当前任务

| 优先级 | 任务 | 预计 |
|:---:|------|:---:|
| **P0** | Phase A: 服务器验证 (`scripts/verify_phaseA.sh`) | 1h |
| **P1** | Phase B: I3 Step 3 全路径打通 | 1-2d |
| **P2** | Phase C: 论文实验 E0→E1/E2/E3 | 1-2w |

## 文档索引

| 文档 | 内容 |
|------|------|
| `ARCHITECTURE_ROADMAP.md` | 代码结构速览 |
| `IMPLEMENTATION_PLAN.md` | Step 3 详细方案 + 论文实验矩阵 |
| `CLAUDE_INSTRUCTIONS.md` | 服务器环境 + 会话恢复指令 |
| `C:\Users\wyf\.claude\plans\...` | 完整架构演进路线图 |
