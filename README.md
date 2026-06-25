# NPU-NVMe Transfer

Ascend NPU 与 NVMe SSD 之间的高性能零拷贝数据传输引擎，面向大模型训练增量检查点（delta checkpoint）场景。

---

## 一、架构概览

```
┌─────────────────────────────────────────────────┐
│  DirectCheckpoint (Python)                       │
│  save / load / delta_save / recover / bench      │
├─────────────────────────────────────────────────┤
│  libnpu_nvme.so (C)                              │
│  SPDK 用户态 NVMe 驱动 + FaF 设备侧轮询           │
├─────────────────────────────────────────────────┤
│  Ascend 910B NPU  │  NVMe SSD (PCIe 直通)        │
└─────────────────────────────────────────────────┘
```

两层 API：
- **C 层** (`npu_nvme_*`): 纯传输 API，HBM ↔ NVMe 批量读写
- **Python 层** (`DirectCheckpoint`): 检查点管理，封装裸盘布局、增量管线、FaF 异步持久化

---

## 二、环境

### 2.1 依赖

- Ascend CANN Toolkit 8.0.RC3 (`/usr/local/Ascend/ascend-toolkit/latest`)
- MindSpore 2.5.0 + MindFormers 1.3.2
- Python 3.9 (conda)

### 2.2 Python 环境

```bash
# conda 环境路径
/home/user7/miniconda3/envs/ms_2.5/bin/python

# 激活
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export PYTHONPATH=/home/user7/npu-nvme/python:$PYTHONPATH
```

> **历史记录**: 环境已从 `/root/miniconda3/envs/ms_2.5` 迁移至 `/home/user7/miniconda3/envs/ms_2.5`。root 下的原环境保留作为备份。

### 2.3 运行权限

当前部分操作需 root 权限（NVMe 设备 `crw------- root:root`）：

| 资源 | 当前状态 | 迁移到普通用户 |
|------|:---:|:---:|
| NVMe 设备 (`/dev/nvme1`) | root only | udev 规则 或 `chmod` (~5 min) |
| NPU 设备 (`/dev/davinci*`) | 所有用户可读写 ✅ | 无需改动 |
| 大页 (`/proc/sys/vm/nr_hugepages`) | root 写 | 启动时预分配 (~5 min) |
| SPDK VFIO | root only | `vfio-noiommu` (~30 min) |
| **迁移总工作量** | — | **~0.5 天** |

---

## 三、快速开始

> 以下命令中 `sudo` 仅在 NVMe/SPDK 操作时需要。NPU 训练可在普通用户下运行。

### 3.1 初始化子模块

```bash
git submodule update --init --recursive
```

### 2.3 编译 SPDK

```bash
cd third_party/spdk
./configure
make -j$(nproc)
```

### 2.4 探测 NVMe 设备（获取 PCIe 地址）

```bash
sudo scripts/setup.sh    # SPDK 自带脚本，列出可用 NVMe 设备和 PCIe 地址
# 输出示例: 0000:83:00.0 → NVMe SSD
```

### 2.5 编译 libnpu_nvme.so

```bash
./build.sh
```

产物在 `build_out/`:
- `lib/libnpu_nvme.so` — C 传输库
- `include/npu_nvme.h` — C 头文件
- `bin/test_npu_nvme` — 冒烟测试

### 2.6 格式化 NVMe 磁盘（首次使用）

```bash
sudo python python/format_npu_disk.py --pci_addr 0000:83:00.0 --npu_id 1 --yes
```

---

## 四、C 层传输 API (`npu_nvme_*`)

纯数据搬运，不包含检查点逻辑。头文件: `include/npu_nvme.h`

### 初始化与清理

```c
int  npu_nvme_init(NPUNVMEContext **ctx, const char *pci_addr, int npu_id,
                   int pipe_depth, uint32_t chunk_size,
                   bool profiling, const char *prof_dir);
void npu_nvme_cleanup(NPUNVMEContext *ctx);
```

### 批量读写

```c
int npu_nvme_write_batch(ctx, void **npu_ptrs, uint64_t *offsets, size_t *sizes, int n);
int npu_nvme_read_batch (ctx, void **npu_ptrs, uint64_t *offsets, size_t *sizes, int n);
int npu_nvme_write_batch_host(ctx, void **ptrs, uint64_t *offsets, size_t *sizes, int n);
int npu_nvme_read_batch_host (ctx, void **ptrs, uint64_t *offsets, size_t *sizes, int n);
```

### 元数据同步

```c
int npu_nvme_sync_meta_io(ctx, uint64_t offset, uint32_t bytes, int is_read, void *buf);
```

### FaF 监听器（设备侧轮询）

```c
int  npu_nvme_register_tasks(ctx, void **ptrs, uint64_t *offsets, size_t *sizes, int n);
int  npu_nvme_set_probe_flag_ptr(ctx, void *dev_ptr);
int  npu_nvme_set_step_ptr(ctx, void *dev_ptr, int interval);
void *npu_nvme_get_probe_flag_dev_ptr(ctx);
```

### Delta 环形区布局

```c
int      npu_nvme_delta_init(ctx, uint64_t slot_size, uint32_t slot_count);
uint64_t npu_nvme_delta_get_area_offset(ctx);
uint64_t npu_nvme_delta_get_slot_size(ctx);
uint32_t npu_nvme_delta_get_slot_count(ctx);
```

---

## 五、Python 检查点 API (`DirectCheckpoint`)

### 4.1 FULL 检查点（全量保存/加载）

```python
from direct_checkpoint import DirectCheckpoint

# 初始化
ckpt = DirectCheckpoint(pci_addr="0000:83:00.0", npu_device_id=1,
                         pipeline_depth=8, slot_size_gb=50)

# 保存（同步阻塞模式）
ckpt.save(model, step=100, sync=True)
# 或异步模式（默认）：
ckpt.save(model, step=100)
ckpt.wait_for_io_completion()

# 加载
ckpt.load(model, step=100)

# 清理
ckpt.cleanup()
```

### 4.2 Delta 增量检查点（FaF 异步）

```python
from direct_checkpoint import DirectCheckpoint
from delta_cell import DeltaTrainCell

ckpt = DirectCheckpoint(pci_addr="0000:83:00.0", npu_device_id=1)

# 编译后注册 DeltaTrainCell 输出缓冲区
cell = DeltaTrainCell(model, optimizer, block_size=524288, top_k_frac=0.10)
_ = cell(*next(dataset.create_tuple_iterator()))  # 触发图编译
ckpt.register_delta_tasks(cell, ckpt_interval=1)

# 训练循环 — FaF 监听器在后台异步持久化增量帧
for step, data in enumerate(dataset):
    loss = cell(*data)      # GE 图内完成 delta 检测 + 量化 + Assign
    # FaF 自动写入，无需 Python 介入

# 恢复
ckpt.recover(model, target_step=50)

ckpt.cleanup()
```

### 5.3 基准测试

```bash
# 激活环境
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export PYTHONPATH=/home/user7/npu-nvme/python:$PYTHONPATH

# 全量基准（所有阶段）
sudo /home/user7/miniconda3/envs/ms_2.5/bin/python python/bench.py --device-id 1 --steps 50

# 仅 delta 管线
sudo /home/user7/miniconda3/envs/ms_2.5/bin/python python/bench.py --device-id 1 --skip-baseline --skip-full

# 仅 FULL 检查点吞吐
sudo /home/user7/miniconda3/envs/ms_2.5/bin/python python/bench.py --device-id 1 --skip-baseline --skip-delta --steps 30
```

### 5.4 DirectCheckpoint 核心方法

| 方法 | 说明 |
|------|------|
| `__init__(pci_addr, npu_device_id, ...)` | 初始化 SPDK + NVMe, 挂载文件系统 |
| `save(model, step, sync=False)` | 全量保存; sync=True 阻塞直到 SPDK 完成 |
| `load(model, step)` | 全量加载 |
| `register_delta_tasks(cell, interval)` | 注册 delta 输出缓冲区, 启动 FaF 监听 |
| `recover(model, target_step)` | FULL + delta 链合并恢复 |
| `delta_save(step, blocks, smalls)` | CPU 侧 delta 帧写入（同步路径） |
| `delta_load_chain(from, to)` | 加载一段 delta 帧链 |
| `cleanup()` | 释放所有资源 |

---

## 六、关键性能数据 (Ascend 910B, MS 2.5.0)

| 指标 | 数值 |
|------|:---:|
| GPT-2 XL 步时 (基线) | **392ms** ± 14ms |
| Delta 管线步时 (overhead) | **574ms** (+182ms, +46.6%) |
| SPDK 写入带宽 | **3661–3926 MB/s** |
| FULL 检查点写入 (3.12 GB) | **~800ms** 同步延迟 |
| Delta 帧大小 (top 10%) | ~159 MB/步 |
| FaF 异步写延迟 | ~45ms (不阻塞训练) |
| Delta overhead 构成 | 参数聚合 ~35ms + INT8 全量量化 ~100ms + TopK/Gather/Assign ~15ms + 其它 ~30ms |

---

## 七、运行环境

### 6.1 当前（root 用户）

| 资源 | 为何需要 root |
|------|--------------|
| `/dev/nvme1` | `crw------- root:root` — 仅 root 可访问 |
| `/proc/sys/vm/nr_hugepages` | 大页池扩容需写权限 |
| SPDK VFIO/PCI | DPDK 用户态驱动需 PCI 设备访问权限 |

### 6.2 迁移到普通用户的改动量评估

**总工作量: ~0.5 天**

| 步骤 | 操作 | 难度 |
|:---:|------|:---:|
| 1 | NVMe 设备权限: `chmod 666 /dev/nvme1` 或 udev 规则 | 低 (~5 min) |
| 2 | 大页预分配: 启动时 `echo 1024 > /proc/sys/vm/nr_hugepages` | 低 (~5 min) |
| 3 | SPDK 非 root 模式: `options vfio-pci enable_unsafe_noiommu_mode=Y` | 中 (~30 min) |
| 4 | NPU 设备: 当前已 `crw-rw-rw-` → 无需改动 | 无 |
| 5 | 测试验证: 重跑 bench.py 确认 BW 不退化 | 低 (~30 min) |

**关键**: NPU 设备和 HBM 访问**已经**对所有用户开放。唯一的阻塞点是 NVMe 设备和 SPDK/DPDK 初始化。步骤 1-3 是一次性系统配置，不需要代码改动。

