# E10/E11 阻塞问题：SPDK 大页耗尽 — 已解决 ✅

> 日期: 2026-06-17 | S4 端到端通过

---

## 根因

8544 个大页（17GB）是 **NPU 驱动 (`drv_devmm_host` 等内核模块) 在内核启动时预分配的 DMA 缓冲区**，不是 SPDK/DPDK 残留。

## 解决方案

在 `src/npu_nvme.c` 中增加 `ensure_hugepages()` 函数，在 `spdk_env_init` 之前自动检测并追加 512 个大页（1GB）给 DPDK 使用。

系统有 2TB 内存，额外 1GB 可忽略。每次 `spdk_env_fini`(`rte_eal_cleanup`) 会正确释放这些页面供下次复用。

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
