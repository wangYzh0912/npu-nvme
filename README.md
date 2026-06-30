# NPU-NVMe Transfer

Ascend NPU 与 NVMe SSD 之间的零拷贝数据传输引擎。C 层提供 HBM ↔ NVMe 批量读写；
Python 层 (`DirectCheckpoint`) 封装裸盘布局与检查点语义。

## 依赖

| 组件 | 版本 |
|------|:---:|
| Ascend CANN | 8.0.RC3 |
| MindSpore | 2.5.0 |
| Python | 3.9 |
| SPDK | v26.01-pre (DPDK 25.07) |
| CMake | ≥ 3.16 |
| GCC | ≥ 7.3 |

## 构建

```bash
git submodule update --init --recursive
cd third_party/spdk && ./configure && make -j$(nproc) && cd ../..
./build.sh
```

产物：`build_out/lib/libnpu_nvme.so`、`build_out/include/npu_nvme.h`。

## 运行

SPDK 需 root 权限访问 NVMe 设备。`/dev/davinci*` 已对所有用户可读写。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH="$(pwd)/build_out/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64"
export PYTHONPATH="$(pwd)/python:$PYTHONPATH"

# 首次使用初始化磁盘
sudo python python/format_npu_disk.py --yes
```

## API

### C 层

```c
// 初始化
int npu_nvme_init(NPUNVMEContext **ctx, const char *pci_addr, int npu_id,
                  int pipe_depth, uint32_t chunk_size,
                  bool enable_profiling, const char *prof_dir);
void npu_nvme_cleanup(NPUNVMEContext *ctx);

// HBM ↔ NVMe 批量读写 (阻塞)
int npu_nvme_write_batch(ctx, void **npu_ptrs, uint64_t *offsets, size_t *sizes, int n);
int npu_nvme_read_batch (ctx, void **npu_ptrs, uint64_t *offsets, size_t *sizes, int n);

// Host ↔ NVMe 批量读写 (memcpy, 无需 NPU)
int npu_nvme_write_batch_host(ctx, void **ptrs, uint64_t *offsets, size_t *sizes, int n);
int npu_nvme_read_batch_host (ctx, void **ptrs, uint64_t *offsets, size_t *sizes, int n);
```

完整 API 见 `include/npu_nvme.h`。

### Python 层

```python
from direct_checkpoint import DirectCheckpoint

ckpt = DirectCheckpoint(nvme_addr="0000:83:00.0", npu_device_id=1,
                         pipeline_depth=8, requested_chunk_size=4194304)

ckpt.save(model, step=100)       # 全量保存
ckpt.load(model, step=100)       # 全量加载
ckpt.cleanup()
```

