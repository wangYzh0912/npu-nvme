# C 测试说明

| 目标 | 用途 | 是否需要硬件 |
|---|---|---|
| `test_npu_nvme` | 环形队列等纯逻辑测试；传入 PCI 地址和 NPU ID 后追加硬件回环 | 可选 |
| `v2_smoke_test` | 固定测试环境下的 Host→NVMe→Host 冒烟测试 | 是 |
| `reactor_v0_test` | SPDK thread/poller 可行性与生命周期诊断 | 是 |
| `reactor_v0_spdk_thread_test` | DPDK mempool 与 SPDK thread 初始化回归诊断 | 是 |

构建仍由仓库根目录的 `CMakeLists.txt` 统一管理。目标机上先完成 `build.sh`，再运行：

```bash
# 无参数时只执行 test_npu_nvme 中不依赖硬件的检查
./build/test_npu_nvme

# 指定裸盘 PCI 地址和 NPU ID 后执行破坏性的硬件回环检查
sudo -n ./build/test_npu_nvme 0000:83:00.0 1
```

硬件回环会直接读写传入 PCI 地址对应的裸 NVMe 空间，只能在确认测试盘和数据边界后运行。
