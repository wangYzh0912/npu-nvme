# C 与硬件验证

`test_npu_nvme` 是旧布局下的主库 smoke，恒真/TBD 占位与旧 delta 测试已经删除。
它的默认写盘布局不是当前 Qwen 注册区域，不要直接用于 Qwen 验收。
实际设备数据回环须按 `tests/hardware/full_io_roundtrip.py` 的参数指定已授权布局；
当前 Qwen 定向闭环使用 `tools/full_checkpoint_acceptance.py`，不格式化介质。
旧 reactor_v0 独立原型及构建目标已移除，现有数据面仍采用单 Reactor。
构建方式和修补 DPDK 归档要求见根 README。

硬件测试会写入裸盘区域，使用已确认的专用设备和布局；不要将历史 BDF 当作机器无关配置。
构建后按测试程序实际 CLI 指定已核对的设备；旧 Phase A 一次性脚本已移除。

当前 `tests/hardware/` 覆盖 I/O 回环、FULL 重启、元数据、Delta 协议链、多 rank 提交及
故障生命周期。当前存在的状态恢复探针为 `tests/hardware/d1_full_state.py`、
`tests/hardware/c2_multirank_state.py`；Qwen 训练入口为根目录 `train.py`。
通过要求：源进程退出后重新读取、字段/控制状态校验及续训；注入失败不得发布成功代际。
Linux/CANN/SPDK 硬件测试不能由 Windows 下的 Python 子集替代。
