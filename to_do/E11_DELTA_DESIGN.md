# E11: SPDK 增量写盘集成 — 技术设计

> 日期: 2026-06-16

## 1. 裸盘布局扩展

```
现有布局:
[Superblock 4KB] [Meta A 400KB] [Meta B 400KB]
[Full Ckpt Stack: N slots × 50GB/slot]
                ↑ stack_start_bytes

扩展后:
[Superblock 4KB] [Meta A 400KB] [Meta B 400KB]
[Full Ckpt Stack: N slots × 50GB/slot]
[Delta Ring: D slots × delta_slot_size]
                ↑ delta_area_offset = stack_start_bytes - D × delta_slot_size
```

Superblock 新增字段（保持向后兼容，用 reserved 区域）：
- `delta_area_offset` (uint64): Delta ring 起始字节偏移
- `delta_slot_size` (uint64): 每槽位字节数 (推荐 256MB)
- `delta_slot_count` (uint32): 槽位数量 (推荐 128 = 覆盖100+步)
- `delta_magic` (uint32): 0x4E4E 标识 delta 区域已初始化

Delta 区域总大小 = 128 × 256MB = 32GB（放盘尾，全量 Stack 之前）

## 2. Delta Frame 格式

每步一个 frame，紧凑二进制格式：

```
DeltaFrame = [Header 4KB] + [Block Records] + [Small Records]

Header (28 bytes, padded to 4KB):
  magic:     uint32  0x414C5444 ("DLTA")
  step_id:   uint32  训练步号
  n_blocks:  uint32  大参数 block 数量
  n_small:   uint32  小参数数量
  total_sz:  uint32  帧总字节数 (含 header)
  checksum:  uint32  CRC32 of payload

Block Record (per-block, variable):
  layer_id:  int16
  name_len:  uint16
  name:      char[name_len]
  block_idx: int32
  data_len:  int32    (= name_len 隐含 nbytes)
  scale:     float32
  data:      int8[data_len]

Small Record (per-small-param):
  layer_id:  int16
  name_len:  uint16
  name:      char[name_len]
  data_len:  int32
  scale:     float32
  data:      int8[data_len]
```

## 3. C 层新增函数

```c
// Delta 写入: host buffer → NVMe delta slot (使用 sync_meta_io 即可, 数据小)
int npu_nvme_write_delta(NPUNVMEContext *ctx, int slot_idx,
                         const void *data, uint32_t total_bytes);

// Delta 读取: NVMe delta slot → host buffer
int npu_nvme_read_delta(NPUNVMEContext *ctx, int slot_idx,
                        void *out_buf, uint32_t max_bytes);

// 元数据: 标记 delta 槽位使用状态
int npu_nvme_delta_slot_commit(NPUNVMEContext *ctx, int slot_idx, int step_id);
```

## 4. Python 层新增类

```python
class I3DeltaWriter:
    """Host-side delta serialization + SPDK write."""
    write_frame(step, block_patches, small_patches) → slot_idx
    read_frame(slot_idx) → (step_id, block_patches, small_patches)
    
class I3DeltaRecovery:
    """Recover from latest full ckpt + delta chain."""
    recover(target_step) → model_weights
```
