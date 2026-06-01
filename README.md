# NPU NVMe Transfer

本仓库对外提供的主要接口是 `npu_nvme_transfer` 纯传输 API，面向 NVMe <-> NPU/Host 的高性能数据搬运，不包含 ckpt/probe 逻辑。

## 快速开始

### 1) 准备依赖

- Ascend CANN Toolkit
- SPDK (建议使用仓库内的 third_party/spdk)

初始化 submodule：
```bash
git submodule update --init --recursive
```

### 2) 编译 SPDK (build 前必须完成)

进入 SPDK 目录并编译：
```bash
cd third_party/spdk
./configure
make
```

可选：用 SPDK 的脚本探测 NVMe 设备并获取 PCIe 地址：
```bash
sudo scripts/setup.sh
```

执行后会列出可用 NVMe 设备和对应 PCIe 地址。

### 3) 编译

```bash
./build.sh
```

构建产物默认在 `build_out/`：

- `build_out/lib/libnpu_nvme.so`
- `build_out/include/npu_nvme_transfer.h`
- `build_out/bin/test_npu_nvme`
- `build_out/bin/npu_nvme_transfer_example`

## npu_nvme_transfer API

头文件：`include/npu_nvme_transfer.h`

### 初始化与资源释放

- `npu_nvme_transfer_init`：创建上下文并初始化 NVMe/NPU 通道
	- `pci_addr`: NVMe PCI 地址，如 `0000:01:00.0`
	- `npu_id`: NPU 设备 ID
	- `pipe_depth`: 并行深度
	- `chunk_size`: 单次传输分块大小
	- `enable_profiling`, `prof_dir`: 可选性能记录
- `npu_nvme_transfer_cleanup`：释放上下文

### 能力查询

- `npu_nvme_transfer_get_total_blocks`：返回 NVMe 总块数
- `npu_nvme_transfer_get_max_transfer`：返回单次最大传输字节数（仅供参考，当前实现可能返回 0）

### 元数据同步

- `npu_nvme_transfer_sync_meta_io`：同步小块元数据
	- `byte_offset`: NVMe 偏移（字节）
	- `total_bytes`: 元数据大小（字节）
	- `is_read`: 1 读 / 0 写
	- `meta_buffer`: Host 侧缓冲区

### 批量读写

- `npu_nvme_transfer_write_batch`：NPU 侧指针批量写入 NVMe
- `npu_nvme_transfer_read_batch`：NVMe 批量读取到 NPU 侧指针
- `npu_nvme_transfer_write_batch_host`：Host 侧指针批量写入 NVMe

> 说明：当前实现不会对用户请求的 `chunk_size`/`sizes[i]` 做强制限制。

## 示例

示例调用见：[examples/npu_nvme_transfer_example.c](examples/npu_nvme_transfer_example.c)

编译示例：
```bash
./build.sh
```

示例可执行文件默认输出：
```bash
build_out/bin/npu_nvme_transfer_example <PCI_ADDR> [NPU_ID]
```

运行前需要设置运行时库路径：
```bash
export LD_LIBRARY_PATH=$PWD/build_out/lib:$LD_LIBRARY_PATH
```

