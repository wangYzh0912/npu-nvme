"""Inspect NVMe metadata slots and list saved checkpoints.

Usage:
- python python/inspect_npu_disk.py --pci_addr <PCI> --npu_id <ID>

Inputs:
- PCI address and NPU device id.
Outputs:
- Prints parsed metadata and checkpoint summary.
"""
import ctypes
import json
import struct
import argparse
import sys
import os

# -- Disk layout constants (byte-addressed, must match direct_checkpoint.py) --
SUPERBLOCK_OFFSET  = 0
META_SLOT_A_OFFSET = 4096
META_SLOT_B_OFFSET = 4096 + 400 * 1024
META_SLOT_BYTES    = 400 * 1024
MAGIC_NUMBER       = b"NPUNVME1"

# -- C interface (reuse direct_checkpoint bindings) --
from direct_checkpoint import lib

def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0

def parse_metadata_slot(ctx, slot_name, offset_bytes):
    meta_buf = ctypes.create_string_buffer(META_SLOT_BYTES)
    # 传入绝对字节偏移和容量
    rc = lib.npu_nvme_sync_meta_io(ctx, offset_bytes, META_SLOT_BYTES, 1, ctypes.c_void_p(ctypes.addressof(meta_buf)))
    
    if rc != 0:
        return {"status": "I/O Error", "checkpoints": {}}
        
    meta_str = meta_buf.value.decode('utf-8', errors='ignore').rstrip('\x00')
    if not meta_str:
        return {"status": "Empty", "checkpoints": {}}
        
    try:
        data = json.loads(meta_str)
        data["status"] = "Valid JSON"
        return data
    except json.JSONDecodeError:
        return {"status": "Corrupted JSON / Dirty Data", "checkpoints": {}}

def inspect_disk(pci_addr, npu_id=0):
    print(f"\n{'='*70}")
    print(f"NPUNVME Disk Inspector")
    print(f"{'='*70}")
    print(f"Target Device : {pci_addr}")
    
    ctx = ctypes.c_void_p()
    ret = lib.npu_nvme_init(ctypes.byref(ctx), pci_addr.encode('utf-8'), npu_id, 1, 1, False, b".")
    if ret != 0:
        print("[Error] SPDK initialization failed.")
        sys.exit(1)

    try:
        # 1. 探测 Superblock
        sb_buf = ctypes.create_string_buffer(4096)
        rc = lib.npu_nvme_sync_meta_io(ctx, SUPERBLOCK_OFFSET, 4096, 1, ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if rc != 0: raise RuntimeError("Failed to read Superblock.")

        header = struct.unpack("<8s I Q Q", sb_buf.raw[:28])
        magic = header[0]
        active_slot = header[1]
        total_bytes = header[2]         # 现在这里是绝对的物理字节数
        stack_start_bytes = header[3]   # 盲写栈区起始物理字节

        is_formatted = (magic == MAGIC_NUMBER)
        
        print("\n[1] SUPERBLOCK (Offset 0)")
        print("-" * 50)
        print(f"  Magic Number    : {magic} {'(OK)' if is_formatted else '(INVALID)'}")
        if not is_formatted: return

        print(f"  Active Meta Slot: {'Slot A' if active_slot == 0 else 'Slot B'} (Pointer: {active_slot})")
        print(f"  Total Capacity  : {format_size(total_bytes)} ({total_bytes} Bytes)")
        print(f"  Stack Start Off : {format_size(stack_start_bytes)}")

        # 2. 物理空间布局
        meta_end_bytes = META_SLOT_B_OFFSET + META_SLOT_BYTES
        heap_size = stack_start_bytes - meta_end_bytes
        stack_size = total_bytes - stack_start_bytes

        print("\n[2] DISK LAYOUT (Macro View)")
        print("-" * 50)
        print(f"  [ 0        ~ 804 KB   ] Metadata Area ({format_size(meta_end_bytes)})")
        print(f"  [ 804 KB   ~ {format_size(stack_start_bytes):<8} ] Heap Area     ({format_size(heap_size)})  <- For complete models")
        print(f"  [ {format_size(stack_start_bytes):<8} ~ END      ] Stack Area    ({format_size(stack_size)})  <- For blind writes")

        # 3. 探测元数据账本内容
        print("\n[3] METADATA SLOTS")
        print("-" * 50)
        
        slot_a_data = parse_metadata_slot(ctx, "A", META_SLOT_A_OFFSET)
        slot_b_data = parse_metadata_slot(ctx, "B", META_SLOT_B_OFFSET)

        # Robust step number extractor: take the last segment after '_'
        def extract_step_num(key_str):
            try:
                return int(key_str.split('_')[-1])
            except ValueError:
                return 0

        for name, data, is_active in [
            ("Slot A", slot_a_data, active_slot == 0),
            ("Slot B", slot_b_data, active_slot == 1)
        ]:
            active_marker = "<-- [ACTIVE]" if is_active else ""
            print(f"\n* {name} {active_marker}")
            print(f"  Status: {data['status']}")
            
            ckpts = data.get("checkpoints", {})
            if not ckpts:
                print("  Checkpoints: None")
            else:
                print(f"  Checkpoints Found: {len(ckpts)}")
                
                # Use the new sort logic
                sorted_steps = sorted(ckpts.items(), key=lambda x: extract_step_num(x[0]))
                
                for ckpt_name, info in sorted_steps:
                    c_type = info.get("type", "UNKNOWN")
                    ranks = info.get("world_size", 1)
                    tensors_count = len(info.get("params", {}))
                    ckpt_bytes = sum(p.get("size", 0) for p in info.get("params", {}).values())
                    
                    # 打印时做个小小的美化，让 COMPLETE 模型更醒目
                    if c_type == "COMPLETE":
                        print(f"    * [{ckpt_name}] Type: {c_type}, Tensors: {tensors_count}, Size: {format_size(ckpt_bytes)}")
                    else:
                        print(f"    - [{ckpt_name}] Type: {c_type}, Ranks: {ranks}, Tensors: {tensors_count}, Size: {format_size(ckpt_bytes)}")

        print(f"\n{'='*70}\n")

    except Exception as e:
        print(f"\n[Fatal Error] Inspection failed: {e}")
    finally:
        lib.npu_nvme_cleanup(ctx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPUNVME Disk Inspector Tool")
    parser.add_argument("--pci_addr", type=str, default="0000:83:00.0")
    parser.add_argument("--npu_id", type=int, default=0)
    args = parser.parse_args()
    inspect_disk(args.pci_addr, args.npu_id)