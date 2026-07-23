# NPU-NVMe Transfer

NPU-NVMe Transfer 提供 Ascend NPU HBM 与 NVMe SSD 之间的数据传输接口，并在 Python 层提供面向 MindSpore 模型的 checkpoint 保存与恢复能力。

## 依赖

- Ascend CANN 8.0.RC3 或兼容版本
- MindSpore 2.5.x
- Python 3.9
- SPDK v26.01-pre，DPDK 25.07，通过 `third_party/spdk` submodule 提供
- CMake 3.16 或更高版本
- GCC 7.3 或更高版本
- 运行 `python/bench.py` 需要 MindFormers、GPT-2 XL 配置和本地 MindRecord 数据集

## 构建

先初始化 submodule 并构建 SPDK：

```bash
git submodule update --init --recursive
cd third_party/spdk
./configure
make -j"$(nproc)"
cd ../..
```

再构建并安装本项目：

```bash
export SPDK_ROOT_DIR="$(pwd)/third_party/spdk"
./build.sh
```

默认安装目录为 `install/`，主要产物包括：

- `install/lib/libnpu_nvme.so`
- `install/include/npu_nvme.h`
- `install/bin/*`

## 基本 Transfer 接口

C 接口在 `include/npu_nvme.h` 中声明。典型流程是先初始化上下文，再按批次提交传输任务，最后释放上下文：

```c
NPUNVMEContext *ctx = NULL;
int rc = npu_nvme_init(&ctx, "0000:83:00.0", 0, 4, 4 * 1024 * 1024,
                       false, ".");
if (rc != 0) {
    return rc;
}

rc = npu_nvme_write_batch(ctx, npu_ptrs, nvme_offsets, sizes, num_items);
rc = npu_nvme_read_batch(ctx, npu_ptrs, nvme_offsets, sizes, num_items);

npu_nvme_cleanup(ctx);
```

主要接口：

- `npu_nvme_write_batch`：从 NPU HBM 写入 NVMe。
- `npu_nvme_read_batch`：从 NVMe 读取到 NPU HBM。
- `npu_nvme_write_batch_host` / `npu_nvme_read_batch_host`：Host DRAM 与 NVMe 之间的批量传输。
- `npu_nvme_raw_write_batch` / `npu_nvme_raw_read_batch`：checkpoint-independent 的 raw NPU-SSD 传输，调用方显式指定 NVMe 绝对偏移。
- `npu_nvme_raw_write_batch_host` / `npu_nvme_raw_read_batch_host`：Host 侧 raw 传输。
- `npu_nvme_write_batch_async` / `npu_nvme_read_batch_async`：提交设备侧异步请求并立即返回 request handle。
- `npu_nvme_raw_write_batch_async` / `npu_nvme_raw_read_batch_async`：带 raw range 校验的设备侧异步 raw 传输。
- `npu_nvme_request_poll` / `npu_nvme_request_wait` / `npu_nvme_request_free`：异步 request 的轮询、等待和释放。
- `npu_nvme_get_completion_fd` / `npu_nvme_drain_completions`：eventfd 驱动的异步完成通知。

Python 侧 raw transfer 封装在 `python/raw_io.py`：

```python
from raw_io import CompletionDispatcher, RawIO

dispatcher = CompletionDispatcher(ctx, auto_start=True)
raw = RawIO(ctx, completion_dispatcher=dispatcher)
raw.write_host([host_ptr], [nvme_offset], [size_bytes])
raw.read_host([host_ptr], [nvme_offset], [size_bytes])

future = raw.write_async([npu_ptr], [nvme_offset], [size_bytes])
future.add_done_callback(lambda fut: fut.result())
future.result(timeout=30.0)
dispatcher.close()
```

## Checkpoint 基本操作

`DirectCheckpoint` 面向 MindSpore 模型提供全量 checkpoint 的保存和恢复。模型需要先完成一次前向或训练步骤，确保参数已分配 NPU device address。

```python
from direct_checkpoint import DirectCheckpoint

ckpt = DirectCheckpoint(
    nvme_addr="0000:83:00.0",
    npu_device_id=0,
    pipeline_depth=4,
    requested_chunk_size=4 * 1024 * 1024,
)

ckpt.save(model, step=100)
ckpt.wait_for_io_completion()

ckpt.load(model, step=100)
ckpt.cleanup()
```

说明：

- `save(model, step)` 将设备侧参数作为 C async request 提交到 reactor，提交后返回；Host fallback 参数仍走同步 Host I/O。
- `wait_for_io_completion()` 等待未完成的 C async request，并在 I/O 完成后提交 checkpoint metadata。
- `load(model, step)` 会先等待当前进程内未完成的 save，再按 metadata 从 NVMe 恢复对应 step 的模型参数。
- `cleanup()` 会等待未完成 I/O，并释放 SPDK、ACL 资源。

## Bench 示例

`python/bench.py` 是公开的 GPT-2 XL 使用示例，用于对比纯训练、delta pipeline 和全量 checkpoint-only 路径。示例会调用 `DirectCheckpoint.save()`，并在需要同步统计时调用 `wait_for_io_completion()`。

```bash
export LD_LIBRARY_PATH="$(pwd)/install/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(pwd)/python:${PYTHONPATH:-}"

sudo env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" PYTHONPATH="$PYTHONPATH" \
  python python/bench.py --device-id 0 --steps 50 --ckpt-every 10
```

输出文件默认写入 `output/bench_full.json`。

## 运行测试

运行 API 测试：

```bash
LD_LIBRARY_PATH="$(pwd)/install/lib:${LD_LIBRARY_PATH:-}" \
  install/bin/npu_nvme_api_test
```

指定 NVMe PCI 地址和 NPU device id：

```bash
sudo env LD_LIBRARY_PATH="$(pwd)/install/lib:${LD_LIBRARY_PATH:-}" \
  install/bin/npu_nvme_api_test 0000:83:00.0 0
```

其他构建目标：

```bash
cmake --build build --target \
  reactor_host_io_roundtrip_test \
  spdk_thread_runtime_test \
  spdk_reactor_poller_test
```
