# Python 层代码审查报告 — direct_checkpoint.py

> 基线: `6f41658` | 1333 行 | 审查日期: 2026-06-23

---

## 一、调用图分析

### 模块级符号（模块加载时执行）

| 符号 | 行号 | 外部调用者 | 判定 |
|------|:---:|------|:---:|
| `bind_depend_op` | 204-207 | `cell_overhead_analysis.py`, `operator_microbenchmarks.py` | **保留**（2 个实验文件用） |
| `wait_op_info` | 225-231 | `cell_overhead_analysis.py`, `operator_microbenchmarks.py`, `test_op_compile.py` | **保留**（3 个实验文件用） |
| `trigger_op_info` | 233-243 | `test_op_compile.py` | **保留**（1 个实验文件用） |
| `_PROBE_LIB_PATH` | 43 | 0（仅定义） | **删除** |
| `ProbeTrainOneStepCell` | 245-303 | 16+ 个实验文件导入 | **保留** |

### DirectCheckpoint 方法

| 方法 | 行号 | 外部调用者 | 判定 |
|------|:---:|------|:---:|
| `__init__` | 311 | 所有实验文件 | 保留 |
| `_mount_filesystem` | 382 | `__init__`, `_find_nearest_full` | 保留 |
| `_get_current_slot_base_offset` | 418 | `build_layout`, `save`, `_background_write_worker` | 保留 |
| `_commit_metadata` | 424 | `save`, `_background_write_worker` | 保留 |
| `set_probe_flag_ptr` | 507 | 多个实验文件 | 保留 |
| `read/write_probe_flag_dev` | 532,550 | `probe_flag_selftest` | 保留 |
| `trigger_probe` | 578 | 3 个旧实验文件 | 保留 |
| `_build_local_param_registry` | 604 | `_prepare_params` | 保留 |
| `_prepare_params` | 649 | `save`, `build_layout` | 保留 |
| `save` | 818 | 5 个实验文件 | 保留 |
| `save_async` | 973 | 0（仅 `save` 内部 `async_save=True` 调用，但无人传该参数） | **删除** |
| `_background_write_worker` | 1009 | 仅 `save_async` | **删除** |
| `load` | 1048 | `recover` + 实验文件 | 保留 |
| `delta_*` 系列 | 1134-1331 | Step 2b/Step 3 实验 | 保留 |

---

## 二、问题清单

### 🔴 严重

**P1: `close()` 默认值反直觉**

```python
# line 1125
def close(self):
    if not getattr(self, '_closed', True):  # default=True → skip cleanup!
        ...
        self._closed = True
```

`getattr(self, '_closed', True)` 的默认值是 `True`，意味着如果 `_closed` 属性不存在（例如 `__init__` 抛异常后），`close()` 会静默跳过清理，可能泄漏 SPDK 资源。正确默认值应该是 `False`。

### 🟡 中等

**P2: `save()` 无条件写 debug 文件**

```python
# line 868-871 — 每次 save() 都写，未受 enable_profiling 控制
with open(f"task_mapping_rank_{self.rank_id}.txt", "w") as f:
    for i, chunk in enumerate(dev_chunks):
        f.write(f"TaskIdx: {i} | Name: {chunk[3]} | Size: {chunk[2].value}\n")
```

**P3: 未使用的 `__init__` 属性**

```python
self.async_lock = threading.Lock()  # line 373 — 从未使用
self.shard_span_bytes = shard_span_bytes  # line 326 — 接收后从未读取
```

**P4: `build_chunks` 返回元组的第 4 个元素是 name 字符串**

```python
chunks.append((ptr, offset, size, name))  # line 141-146
```

调用方 `save()` 中解包为 `(p, o, s, name)`，但 `_background_write_worker` 中解包为 `(p, o, s)`（无 name）。如果 chunk 来自同一个 `build_chunks` 路径则 OK，但耦合脆弱。

**P5: `recover()` 硬编码 `block_size=524288` 和 `np.float16`**

```python
# line 1312, 1326
block_size = 524288  # should be self.chunk_size // 2 or similar
p.set_data(Tensor(host_weights[name].astype(np.float16), ms.float16))
```

如果模型不是 FP16（例如 FP32），恢复会静默截断精度。

### 🟢 轻微

**P6: 情绪化/开发阶段注释（~20 处）**

| 行号 | 内容 |
|:---:|------|
| 137 | `# 【修改点1】：获取名字` |
| 145 | `# 【修改点2】：把名字塞进元组里` |
| 397 | `# 【核心防爆修复】...分布式阵列！` |
| 608 | `利用底层同步拷贝的安全拦截特性...摸清本卡的真实张量` |
| 689-692 | `强制阻塞屏障...防止...破坏 NPU 显存` |
| 844 | `# 【物理保险丝】：严格检查...` |
| 854 | `# 客货分流：走主机内存的...` |
| 868 | `# 【新增 Debug 代码】...看看死锁的 Task 到底叫什么名字！` |
| 941 | `# 3. 后台计算耗时并打印铁证` |
| 966 | `# 主线程极速返回，让 MindSpore 跑 Step 16`（数字 16 不对） |
| 970 | `# 因为真实写入在后台，此处返回空壳数据...` |
| 1132 | `# I3 Delta (增量) I/O — S2/S3: 端到端打通` |

**P7: `save()` 方法过长（154 行）**

`save()` 包含了布局计算、ctypes 数组组装、线程创建、计时打点四种职责。建议拆分为 `_build_ctypes_arrays()` 和 `_dispatch_io()`。

**P8: `_mount_filesystem()` 使用 magic number `28`**

```python
header = struct.unpack("<8s I Q Q", sb_buf.raw[:28])  # 28 = 8+4+8+8
```

应定义为常量 `SUPERBLOCK_HEADER_BYTES = 28`。

**P9: 模块级重复 import**

```python
# line 22
import mindspore as ms
from mindspore import ops, nn, Tensor

# line 202 (inside rebuild_chunks_from_meta, at module level after the function)
    from mindspore import nn, Tensor, ops  # dead code — already imported
```

**P10: `delta_save` 中的 `from i3_delta_writer import pack_delta_frame` 是延迟 import**

如果 `i3_delta_writer.py` 不存在或被移动，`delta_save()` 会在运行时崩溃。建议在 `__init__` 中验证导入。

---

## 三、量化修改方案

| 步骤 | 内容 | 预计变化 |
|:---:|------|:---:|
| **Step 1** | 删除死代码: `save_async`, `_background_write_worker`, `_PROBE_LIB_PATH`, 重复 import (行 202) | -55 行 |
| **Step 2** | 修复 bug: `close()` 默认值, debug 文件加 profiling gate, `recover()` 硬编码 dtype | +10/-5 |
| **Step 3** | 清理注释: 删除 `【】` 标记, 情绪化语言 → 标准英文 | ±30 |
| **Step 4** | 删除未使用属性: `async_lock`, `shard_span_bytes` | -3 |
| **Step 5** | 提取常量: `SUPERBLOCK_HEADER_BYTES`, `BLOCK_SIZE` | +5 |

**最终**: ~1300 行, 无功能回归, Python import 正常。

---

## 四、保留但需注意的模块

| 模块 | 原因 |
|------|------|
| `ProbeTrainOneStepCell` | 16+ 个实验文件导入，是 FaF 训练的唯一入口 |
| `wait_op_info` / `trigger_op_info` / `bind_depend_op` | 旧实验文件用（非 baselines），后续可随实验清理 |
| `trigger_probe()` 方法 | 3 个旧实验调用，与 C 层 probe_flags 配套 |
| `delta_*` 系列方法 | I3 核心路径，Step 2b/Step 3 实验依赖 |
