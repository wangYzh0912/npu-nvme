# C 与硬件验证

`test_npu_nvme` 验证主库，`v2_smoke_test` 验证 init/Host 写读/cleanup。
`test_npu_nvme` 中设备批量回环仍有 TBD 占位，不能把该程序退出成功当作四路径全通过；
实际设备数据回环与恢复须执行 `tests/hardware/full_io_roundtrip.py` 等对应门禁。
旧 reactor_v0 独立原型及构建目标已移除，现有数据面仍采用单 Reactor。
构建方式和修补 DPDK 归档要求见根 README。

硬件测试会写入裸盘区域，使用已确认的专用设备和布局；不要将历史 BDF 当作机器无关配置。
构建后按测试程序实际 CLI 指定已核对的设备；旧 Phase A 一次性脚本已移除。

当前 `tests/hardware/` 覆盖 I/O 回环、FULL 重启、元数据、Delta 协议链、多 rank 提交及
故障生命周期。完整训练恢复入口为 `c1_training_state_restart.py`、`c2_multirank_state.py`
和 `experiments/benchmarks/run_single_card_full.py`，参数用对应 `--help` 查看。
通过要求：源进程退出后重新读取、字段/控制状态校验及续训；注入失败不得发布成功代际。
Linux/CANN/SPDK 硬件测试不能由 Windows 下的 Python 子集替代。
