# NPU-NVMe Transfer

Ascend NPU 与 NVMe SSD 之间的高性能零拷贝数据传输引擎，面向大模型训练检查点
（checkpoint）场景。提供两层 API：**C 层传输 API**（HBM ↔ NVMe 批量读写）和
**Python 层检查点 API**（全量保存/加载、增量持久化）。

---

## 版本依赖

| 组件 | 版本 | 说明 |
|------|:---:|------|
| 操作系统 | openEuler 22.03 LTS | Linux 5.10, aarch64 |
| Ascend CANN | 8.0.RC3 | `/usr/local/Ascend/ascend-toolkit/latest` |
| MindSpore | 2.5.0 | Python 3.9, conda 环境 |
| MindFormers | 1.3.2 | GPT-2 模型定义 |
| Python | 3.9 | `/home/user7/miniconda3/envs/ms_2.5` |
| SPDK | v26.01-pre | 用户态 NVMe 驱动，捆绑的 DPDK 25.07 |
| ISA-L | 2.30+ | Intel 存储加速库，SPDK 依赖 |
| GCC | 7.3+ | C11 标准编译 |
| CMake | 3.16+ | 构建系统 |

---

## 初始化

### 1. 克隆与子模块

```bash
git clone <repo-url>
cd npu-nvme
git submodule update --init --recursive
```

### 2. 编译 SPDK

```bash
cd third_party/spdk
./configure
make -j$(nproc)
cd ../..
```

### 3. 编译 libnpu_nvme.so

```bash
./build.sh
```

产物在 `build_out/`：
| 路径 | 说明 |
|------|------|
| `lib/libnpu_nvme.so` | C 传输库 |
| `include/npu_nvme.h` | C 公共头文件 |
| `bin/test_npu_nvme` | 冒烟测试 |
| `bin/run_test.sh` | 带环境变量的运行脚本 |

### 4. 运行时环境

```bash
# 激活 CANN 工具链
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置 Python 搜索路径
export PYTHONPATH="$(pwd)/python:$PYTHONPATH"

# 设置库搜索路径（SPDK 需 root 权限访问 NVMe 设备）
export LD_LIBRARY_PATH="$(pwd)/build_out/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH"
```

> **注意**: SPDK 通过用户态驱动直接访问 NVMe 设备，需 root 权限。NPU 设备
> (`/dev/davinci*`) 已对所有用户可读写，无需额外权限。

### 5. 格式化 NVMe 磁盘

首次使用前需初始化超级块和元数据区：

```bash
sudo python python/format_npu_disk.py --yes
# 默认使用 PCIe 地址 0000:83:00.0，NPU 设备 0
```

### 6. 验证安装

```bash
# C 层冒烟测试
sudo LD_LIBRARY_PATH=build_out/lib:... build/bin/test_npu_nvme
```

---

## C 层传输 API

C 层提供纯数据搬运能力，不包含检查点逻辑。所有函数通过 `NPUNVMEContext *` 不透明
句柄操作。头文件：`include/npu_nvme.h`

### 初始化与清理

```c
int  npu_nvme_init(NPUNVMEContext **ctx, const char *pci_addr, int npu_id,
                   int pipe_depth, uint32_t chunk_size,
                   bool enable_profiling, const char *prof_dir);
void npu_nvme_cleanup(NPUNVMEContext *ctx);
uint64_t npu_nvme_get_total_blocks(NPUNVMEContext *ctx);  // 磁盘容量 (字节)
int      npu_nvme_get_max_transfer(NPUNVMEContext *ctx);   // 配置的块大小
```

| 参数 | 推荐值 | 说明 |
|------|:---:|------|
| `pci_addr` | `"0000:83:00.0"` | NVMe 设备 PCIe BDF 地址 |
| `npu_id` | `0` 或 `1` | 昇腾 NPU 设备 ID |
| `pipe_depth` | `4`–`16` | DMA 管线深度，影响并发度 |
| `chunk_size` | `4194304` (4 MiB) | 单次 DMA 块大小 |

### 批量读写

```c
// HBM 路径 — 通过 aclrtMemcpy 进行 NPU ↔ DMA 缓冲区拷贝
int npu_nvme_write_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                         uint64_t *nvme_offsets, size_t *sizes, int num_items);
int npu_nvme_read_batch(NPUNVMEContext *ctx, void **npu_ptrs,
                        uint64_t *nvme_offsets, size_t *sizes, int num_items);

// 主机路径 — 通过 memcpy 在主机内存与 DMA 缓冲区之间拷贝，无需 NPU 参与
int npu_nvme_write_batch_host(NPUNVMEContext *ctx, void **host_ptrs,
                              uint64_t *nvme_offsets, size_t *sizes, int num_items);
int npu_nvme_read_batch_host(NPUNVMEContext *ctx, void **host_ptrs,
                             uint64_t *nvme_offsets, size_t *sizes, int num_items);
```

所有批量 I/O 为**同步阻塞调用**：函数返回时数据已到达存储介质（写）或已复制到
用户缓冲区（读）。内部通过异步有限状态机驱动，但对外暴露为阻塞语义。

### 元数据 I/O

```c
int npu_nvme_sync_meta_io(NPUNVMEContext *ctx, uint64_t byte_offset,
                          uint32_t total_bytes, int is_read, void *meta_buffer);
```

使用专用 qpair，不与数据路径竞争。适用于超级块和 JSON 账本的读写，单次 ≤ 1 MiB。

### FaF 监听器控制

FaF（Fire-and-Forget）模式在训练步边界自动触发后台写入，Python 训练循环不阻塞。

```c
int  npu_nvme_register_tasks(NPUNVMEContext *ctx, void **npu_ptrs,
                             uint64_t *nvme_offsets, size_t *sizes, int num_items);
int  npu_nvme_set_probe_flag_ptr(NPUNVMEContext *ctx, void *dev_ptr);
int  npu_nvme_set_probe_flag_value(NPUNVMEContext *ctx, uint32_t value);
int  npu_nvme_set_step_ptr(NPUNVMEContext *ctx, void *dev_ptr, int ckpt_interval);
void *npu_nvme_get_probe_flag_dev_ptr(NPUNVMEContext *ctx);
```

### 性能剖析

```c
// 返回最近一次批量 I/O 的 C 层延迟 (微秒)，排除 Python 编组开销
uint64_t npu_nvme_get_last_io_us(NPUNVMEContext *ctx, int is_read);
// is_read: 0 = 写, 1 = 读
```

### 增量检查点磁盘布局

```c
int      npu_nvme_delta_init(NPUNVMEContext *ctx, uint64_t slot_size, uint32_t count);
uint64_t npu_nvme_delta_get_area_offset(NPUNVMEContext *ctx);
uint64_t npu_nvme_delta_get_slot_size(NPUNVMEContext *ctx);
uint32_t npu_nvme_delta_get_slot_count(NPUNVMEContext *ctx);
```

---

## Python 检查点 API

`DirectCheckpoint` 类封装裸盘布局、超级块管理、并发控制和 FaF 异步持久化，
提供面向训练循环的检查点接口。

### 全量保存与加载

```python
from direct_checkpoint import DirectCheckpoint

# 初始化
ckpt = DirectCheckpoint(
    nvme_addr="0000:83:00.0",   # NVMe PCIe 地址
    npu_device_id=1,             # NPU 设备 ID
    pipeline_depth=8,            # DMA 管线深度
    requested_chunk_size=4194304,# 块大小 (4 MiB)
)

# 保存（默认异步: 后台线程执行 SPDK 写入）
ckpt.save(model, step=100)

# 等待后台 I/O 完成
ckpt.wait_for_io_completion()

# 同步保存（save + wait 一步完成）
ckpt.save(model, step=100, sync=True)

# 加载
ckpt.load(model, step=100)

# 释放资源
ckpt.cleanup()
```

### 增量检查点（FaF 异步持久化）

```python
from direct_checkpoint import DirectCheckpoint
from delta_cell import DeltaTrainCell

ckpt = DirectCheckpoint(nvme_addr="0000:83:00.0", npu_device_id=1)

# DeltaTrainCell: 图编译后注册输出缓冲区
cell = DeltaTrainCell(model, optimizer, block_size=524288, top_k_frac=0.10)
_ = cell(*next(dataset.create_tuple_iterator()))  # 触发图编译
ckpt.register_delta_tasks(cell, ckpt_interval=1)

# FaF 模式下训练: 步边界自动触发后台写入
for step, data in enumerate(dataset):
    loss = cell(*data)

# 恢复到指定步数（合并 FULL + delta 链）
ckpt.recover(model, target_step=50)
ckpt.cleanup()
```

### C 层性能剖析

```python
c_latency_us = ckpt.get_last_io_us(is_read=False)  # 最近一次写 I/O 的纯 C 层延迟
c_bw = total_bytes / (c_latency_us / 1e6) / 1e6     # 排除 Python 开销的真实带宽
```

### DirectCheckpoint 核心方法

| 方法 | 说明 |
|------|------|
| `__init__(nvme_addr, npu_device_id, ...)` | 初始化 SPDK，挂载文件系统 |
| `save(model, step, sync=False)` | 全量保存；`sync=True` 阻塞直到写入完成 |
| `load(model, step)` | 全量加载 |
| `wait_for_io_completion()` | 等待后台 I/O 完成 |
| `register_delta_tasks(cell, ckpt_interval)` | 注册 delta 输出，启动 FaF 监听 |
| `recover(model, target_step)` | FULL + delta 链合并恢复 |
| `get_last_io_us(is_read=False)` | C 层 I/O 延迟（微秒） |
| `cleanup()` | 释放所有 SPDK 和 ACL 资源 |

---

## 工具

以下工具脚本位于 `python/` 目录下，用于磁盘格式化、设备探查和性能基准测试：

| 工具 | 说明 |
|------|------|
| `format_npu_disk.py` | 格式化 NVMe 磁盘，初始化超级块 |
| `inspect_npu_disk.py` | 查看磁盘元数据信息 |
| `bench.py` | 全量/增量检查点综合基准测试 |
| `export_model.py` | 模型导出工具 |

工具用法请查看各脚本的 `--help` 输出。

---

## 性能

测试环境：昇腾 910B (64 GB HBM) + 三星 PM9A3 3.84 TB NVMe SSD +
GPT-2 XL (3.28 GB FP16 参数)，4 MiB 块，管线深度 8。

| 指标 | 数值 |
|------|:---:|
| SPDK 顺序写带宽 | **4,432 MB/s** |
| C 层纯 I/O 延迟 (1 GB 写) | 259 ms (4,143 MB/s) |
| Python 开销 | 0.5% |
| Reactor CPU 占用 (稳态) | < 1% |

---

## 目录结构

```
npu-nvme/
├── include/
│   ├── npu_nvme.h              # C 公共 API 头文件
│   └── internal/                # 内部头文件 (io_task, context, ring_buffer)
├── src/
│   └── npu_nvme.c              # C 传输引擎实现 (SPDK + ACL)
├── python/
│   ├── direct_checkpoint.py    # DirectCheckpoint 检查点管理器
│   ├── delta_cell.py           # DeltaTrainCell 增量训练封装
│   ├── delta_protocol.py       # 增量帧打包/解包协议
│   ├── chunk_helpers.py        # 参数 → 块数组转换
│   ├── disk_layout.py          # 磁盘布局常量
│   ├── c_bindings.py           # Python→C ctypes 绑定
│   ├── format_npu_disk.py      # 磁盘格式化工具
│   ├── inspect_npu_disk.py     # 磁盘探查工具
│   └── bench.py                # 综合基准测试
├── experiments/
│   ├── i1/                     # 原始带宽 + 数据完整性
│   ├── i2/                     # FaF 延迟 + 一致性
│   └── i3/                     # 分阶段分解 + TopK 敏感性
├── docs/
│   └── REACTOR_CONTROL_PLANE.md # SPDK Reactor 控制平面论文
├── third_party/spdk/           # SPDK 子模块
├── CMakeLists.txt              # CMake 构建配置
└── build.sh                    # 一键构建脚本
```

## 许可

Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
