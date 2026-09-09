# NPU-NVMe Transfer

Ascend NPU 与裸 NVMe 之间的检查点传输实现。C 层通过 ACL、Host DMA 缓冲和 SPDK
执行分块 I/O；Python 层管理 FULL 训练状态、请求生命周期与多 rank 提交。
数据经过 Host DMA 缓冲，不是 NPU 到 SSD 的 PCIe peer-to-peer 通路。

## 内容与范围

- `src/`、`include/`：单 Reactor 数据面、元数据与请求接口。
- `python/`：DirectCheckpoint、FULL/多 rank 协议、增量基础实现。
- [experiments](experiments/README.md)：当前 FULL/I/O 验证、训练更新观测和统一 baseline。
- `tests/`：协议、回环、故障、新进程恢复与多 rank 测试。
- [results](results/README.md)：最新正式实验及解释当前限制所必需的原始证据。

master 不含开发计划、汇报材料和过往实验；历史可从 Git 原提交及实验分支追溯。
当前正式模型入口为 GPT-2/XL，Qwen 升级与适配尚未实现。已保存记录支持 GPT-2 FULL
和 2/4-rank 恢复；XL live 的严格续训门禁仍有失败。Delta/R0 是实验实现，协议测试和
更新统计不等于已验收的增量训练持久化通路。

## 构建

目标平台为 Linux/Ascend。最近保存的环境记录为 MindSpore 2.5.0、MindFormers 1.3.2、
CANN 7.5.0.1.129、Python 3.9、910B3；这不是跨版本兼容保证。
SPDK 使用 Git 子模块锁定版本；构建要求 CMake ≥3.16、GCC ≥7.3。

```bash
git submodule update --init --recursive
cd third_party/spdk
./configure
make -j"$(nproc)"
cd ../..
# 按 .build_config.example 创建本机 .build_config
./build.sh
```

配置实际 CANN/SPDK 路径，以及导出 `common_ring_*` 的
`DPDK_MEMPOOL_RING_FIXED_LIB`。该修补归档不是仓库自带产物，缺失时构建会失败。
产物为 `build_out/lib/libnpu_nvme.so` 和 `build_out/include/npu_nvme.h`。

## 运行

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH="$PWD:$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/build_out/lib:$LD_LIBRARY_PATH"
python experiments/benchmarks/run_single_card_full.py --dry-run \
  --model gpt2 --mode serial --checkpoint-steps 2 --total-steps 3
```

目标机运行示例（先核对实际设备和测试区域）：

```bash
python experiments/benchmarks/run_single_card_full.py \
  --model gpt2 --mode serial --checkpoint-steps 2 --total-steps 3 \
  --npu 7 --pci 0000:83:00.0 --run-dir experiments/output/full-smoke-001
```

入口执行源训练、持久化、源进程退出、新进程恢复及续训门禁，失败返回非零。
`frozen_async` 和 `live_async` 是独立机制，各自需要正确性验证。
硬件运行需核对 NPU/NUMA、hugepages、NVMe 序列号/绑定和动态库路径。裸盘写入会覆盖
测试区域，只使用专用区域；已有盘不应为了复现重新格式化。首次初始化选项见
`python python/format_npu_disk.py --help`。SPDK 需要设备与 hugepage 权限；已保存恢复
记录在 root/PA IOVA 下通过，不承诺普通用户或不同驱动组合可直接 attach。

`include/npu_nvme.h` 定义 C API。`DirectCheckpoint.save/load` 面向模型参数，
`save_state/load_state` 面向命名组件和控制状态；完整恢复还要涵盖优化器、RNG 和数据
位置，参照共享状态桥和 FULL runner。异步派发后必须检查完成屏障或句柄的失败状态。

## 验证

完整 Python 集：`python -m pytest tests/python -q`，需要 Linux/MindSpore 等依赖。
普通开发机子集见[Python 测试说明](tests/python/README.md)，目标机边界见
[C/硬件测试说明](tests/c/README.md)。本轮整理不改变原实验的 commit/环境记录，
也不将本机语法或协议测试视为 Ascend/SPDK 硬件重跑。
