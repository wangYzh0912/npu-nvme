# E10/E11 阻塞问题：SPDK 大页耗尽 — 已解决 ✅

> 日期: 2026-06-17 (原始) | 2026-06-25 (验证确认)

---

## 根因

Ascend 910B 服务器的 NPU 驱动（`drv_devmm_host` 等内核模块）在系统配置中预留了大量大页作为 DMA 缓冲区。DPDK 的 `rte_eal_init()` 检测到 Free=0 后拒绝启动。

## 解决方案

在 `src/npu_nvme.c` 中增加 `ensure_hugepages()` 函数，在 `spdk_env_init` 之前自动检测空闲大页数量。若空闲页 < 512，则追加对应数量至 `nr_hugepages`。

`HUGEPAGE_PADDING = 512`（1 GB），系统有 2 TB 内存，额外 1 GB 可忽略。

## 当前环境状态 (2026-06-25)

| 指标 | 值 |
|------|:---:|
| 总大页数 | **1024** (2 GB) |
| 空闲大页 | **1024** (全部空闲) |
| 每 NUMA node | **128 / 128 free** (8 nodes) |
| `ensure_hugepages()` 需要扩容？ | **否** (1024 > 512) |

**8500+ 大页的历史残留不是本项目的泄漏**。当前系统配置仅有 1024 个大页（内核默认值）。之前的 8544 页是 NPU 驱动安装脚本在当时设置的值。系统重启/重配置后恢复为 1024。

当前环境 SPDK 可直接使用，无需任何干预。

## 清理流程修复确认 (2026-06-25)

delta 分支的 `npu_nvme_cleanup()` 曾因访问全局变量 `probe_flags`（可能已释放）而崩溃，导致进程异常退出 → DPDK 库析构函数 `rte_eal_cleanup()` 不执行 → 大页无法归还内核。

HEAD 已通过移除全局 `probe_flags` 根除该问题：
- 监听器停止改为 `ctx->listener.stop_listener = 1`（仅操作 context 内部字段）
- 所有状态封装在 context 子结构中，无悬空指针风险
- cleanup → 进程正常退出 → `rte_eal_cleanup()` → 大页释放 ✅

## SPDK qpair 线程安全性 (2026-06-25)

已确认 listener 线程与主线程共享同一 `ctx->qpair`，存在竞争风险。已添加 `pthread_mutex_t io_lock` 保护所有 I/O 入口点（`run_write_pipeline`、`run_read_pipeline`、`sync_meta_io`、`read_batch_host`）。

## 验证结果

```
S3 SPDK Smoke Test: SPDK init 2.4s, Write 1.1ms, Read 1648ms, Round-trip byte-perfect ✅
S4 E2E Single-Card:   Train 30 steps → FULL(SPDK) + DELTA×30(NVMe) → Recovery(pickle) PASS ✅
```

## E2E 链路总结

```
写链路:
  [Save] ckpt.save(model, step=0)       → SPDK DMA write 249MB, 141ms H/W
  [Delta] ckpt.delta_save(step, blocks)  → sync_meta_io 1-5MB, 420ms avg
  [Meta] _commit_metadata + pickle        → NVMe JSON + local pickle

恢复链路:
  [Read] pickle → meta (checkpoints + delta_chain)
  [Full] 初始模型 (seed=42 重建, step_0 全量)
  [Delta] rr_ckpt.delta_load_slot(slot) × 30 → apply_delta_patches
  [Verify] NRMSE + Loss + Hash vs oracle
```
