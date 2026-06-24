# P0-U7 Final Report

## SPDK overhead 取决于 MS Context 初始化顺序

### Root Cause

`spdk_env_init()` → `rte_eal_init()` 改变进程全局状态（hugepage 映射、CPU 调度）。如果发生在 MSC++ runtime 首次初始化**之前**，MS 的所有组件在 "被 SPDK 污染" 的状态下启动 → 性能劣化 +304%。如果发生在**之后**，MS 已在干净状态下就绪 → 仅 +10% overhead。

### 证据链

| 测试 | 初始化顺序 | Overhead |
|------|-----------|----------|
| E1/R1/X2a/X2b/Z1 | SPDK → MS train | +296-333% |
| H1 / R_test R2 | MS train → SPDK → MS train | +4-10% |

### Workaround (已实现)

```python
model.train(epoch=1, train_dataset=ds.take(1), dataset_sink_mode=False)  # warmup
ckpt = DirectCheckpoint(...)  # SPDK init after MS runtime ready
model.train(epoch=N, ...)     # only ~10% overhead
```

### Impact

- Fire-and-Forget 架构不需要改变
- 独立进程 / io_uring / torch_npu 方案都不需要
- 只需在 DirectCheckpoint.__init__ 前做一次 1-step warmup
