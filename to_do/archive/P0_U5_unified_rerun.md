# P0-U5: 统一实验重跑

- [x] 清理旧输出数据 — 自动覆盖
- [x] 基线 A-F 对比 (`baseline_benchmark.py`) — ✅ 完成
- [x] SPDK 端到端 (`spdk_end_to_end.py`) — ✅ 完成
- [x] Cell 开销隔离 (`cell_overhead_analysis.py`) — ✅ 完成（修复重名 Parameter bug）
- [x] 算子微观 E2+F1+F2+F3 (`operator_microbenchmarks.py`) — ✅ 完成（修复重名 Parameter bug）
- [x] 汇总数据一致性检查 — ✅ 完成

## 修复记录
- operator_microbenchmarks.py: `Dense+ReLU+Dense` → `SequentialCell` (避免 weight/bias 重名)
- cell_overhead_analysis.py: 同上 + CellMinimal.construct 参数类型修复 (*inputs → x)
