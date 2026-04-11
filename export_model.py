import ctypes
import json
import struct
import argparse
import sys
import os
import math
import numpy as np
import time

# ============================================================
# 裸盘物理布局常量
# ============================================================
SUPERBLOCK_OFFSET = 0
META_SLOT_A_OFFSET = 4096                 
META_SLOT_B_OFFSET = 4096 + 400 * 1024    
META_SLOT_BYTES = 400 * 1024              
HEAP_START_OFFSET = META_SLOT_B_OFFSET + META_SLOT_BYTES 
MAGIC_NUMBER = b"NPUNVME1"
CHUNK_SIZE = 4 * 1024 * 1024  

# 【关键点1】保证 SPDK 环境与训练时不冲突
os.environ.setdefault("SPDK_SHM_ID", "1")

# ============================================================
# 绑定 C 接口 (严格对齐训练脚本)
# ============================================================
LIB_PATH = os.environ.get("NPU_NVME_LIB", "./out/lib/libnpu_nvme.so")

try:
    lib = ctypes.CDLL(LIB_PATH)
    class NPUNVMEContext(ctypes.Structure): pass

    lib.npu_nvme_init.argtypes = [
        ctypes.POINTER(ctypes.POINTER(NPUNVMEContext)), ctypes.c_char_p, ctypes.c_int, 
        ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_char_p
    ]
    lib.npu_nvme_init.restype = ctypes.c_int
    lib.npu_nvme_cleanup.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    
    lib.npu_nvme_sync_meta_io.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p
    ]
    lib.npu_nvme_sync_meta_io.restype = ctypes.c_int

    lib.npu_nvme_read_batch.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
    ]
    lib.npu_nvme_read_batch.restype = ctypes.c_int

    lib.npu_nvme_write_batch.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
    ]
    lib.npu_nvme_write_batch.restype = ctypes.c_int
except OSError as e:
    print(f"[Fatal] Failed to load C library: {e}")
    sys.exit(1)

def chunk_tensor(ptr_base, start_nvme_offset, total_size):
    """将巨型 Tensor 切割成 4MB 的安全块"""
    chunks = []
    remaining = total_size
    inner_off = 0
    nvme_off = start_nvme_offset
    
    while remaining > 0:
        take = min(remaining, CHUNK_SIZE)
        chunks.append((
            ctypes.c_void_p(ptr_base + inner_off),
            ctypes.c_uint64(nvme_off),
            ctypes.c_size_t(take)
        ))
        remaining -= take
        inner_off += take
        nvme_off += int(math.ceil(take / 4096.0) * 4096)
    return chunks

def export_to_heap(pci_addr, target_step, npu_id=0):
    print(f"\n{'='*70}")
    print(f"🚀 NPUNVME Model Exporter: Aggregating SHARDs to HEAP")
    print(f"{'='*70}")
    
    ctx = ctypes.POINTER(NPUNVMEContext)()
    # 【关键点2】pipeline_depth 严格改回 4，确保大页内存不爆
    ret = lib.npu_nvme_init(ctypes.byref(ctx), pci_addr.encode('utf-8'), npu_id, 4, CHUNK_SIZE, False, b".")
    if ret != 0:
        print("[Fatal] SPDK init failed.")
        sys.exit(1)

    try:
        # ---------------------------------------------------------
        # Phase 1: 挂载文件系统与账本解析
        # ---------------------------------------------------------
        sb_buf = ctypes.create_string_buffer(4096)
        if lib.npu_nvme_sync_meta_io(ctx, SUPERBLOCK_OFFSET, 4096, 1, ctypes.c_void_p(ctypes.addressof(sb_buf))) != 0:
            raise RuntimeError("Failed to read Superblock.")

        header = struct.unpack("<8s I Q Q", sb_buf.raw[:28])
        if header[0] != MAGIC_NUMBER: raise RuntimeError("Disk not formatted!")
            
        active_slot, total_bytes, stack_start_bytes = header[1], header[2], header[3]

        target_meta_offset = META_SLOT_A_OFFSET if active_slot == 0 else META_SLOT_B_OFFSET
        meta_buf = ctypes.create_string_buffer(META_SLOT_BYTES)
        lib.npu_nvme_sync_meta_io(ctx, target_meta_offset, META_SLOT_BYTES, 1, ctypes.c_void_p(ctypes.addressof(meta_buf)))
        
        meta_dict = json.loads(meta_buf.value.decode('utf-8', errors='ignore').rstrip('\x00'))
        shard_key = f"step_{target_step}"
        if shard_key not in meta_dict.get("checkpoints", {}):
            raise ValueError(f"Target step '{shard_key}' not found!")
            
        shard_info = meta_dict["checkpoints"][shard_key]
        print(f"[1/4] Found '{shard_key}'. Tensors: {len(shard_info['params'])}")

        # ---------------------------------------------------------
        # Phase 2: 一键直通读取 (严格复刻 direct_checkpoint)
        # ---------------------------------------------------------
        print("[2/4] Pulling SHARDs from Stack Area into RAM...")
        t_read_start = time.time()
        
        tensors_in_ram = {}
        read_chunks = []
        total_size_bytes = 0
        
        for name, info in shard_info["params"].items():
            np_arr = np.empty(info["shape"], dtype=np.dtype(info["dtype"]))
            tensors_in_ram[name] = {"arr": np_arr, "info": info}
            total_size_bytes += info["size"]
            read_chunks.extend(chunk_tensor(np_arr.ctypes.data, info["offset"], info["size"]))

        num = len(read_chunks)
        c_ptrs = (ctypes.c_void_p * num)()
        c_offs = (ctypes.c_uint64 * num)()
        c_sizes = (ctypes.c_size_t * num)()
        for i, (p, o, s) in enumerate(read_chunks):
            c_ptrs[i] = p
            c_offs[i] = ctypes.c_uint64(o.value)
            c_sizes[i] = s

        print(f"      -> Generated {num} chunks. Handing over to DMA...", flush=True)
        rc = lib.npu_nvme_read_batch(ctx, c_ptrs, c_offs, c_sizes, num)
        if rc != 0: raise RuntimeError("Read Batch failed!")
        
        print(f"      -> SUCCESS! Loaded {total_size_bytes / 1024**3:.2f} GB in {time.time()-t_read_start:.2f}s.")

        # ---------------------------------------------------------
        # Phase 3: 连续内存模型写入 Heap
        # ---------------------------------------------------------
        print("\n[3/4] Writing continuous COMPLETE model to Heap Area...")
        t_write_start = time.time()
        complete_key = f"complete_step_{target_step}"
        complete_layout = {}
        current_heap_offset = HEAP_START_OFFSET 
        write_chunks = []

        sorted_tensors = sorted(tensors_in_ram.items(), key=lambda x: x[0])
        
        for name, data in sorted_tensors:
            np_arr, info = data["arr"], data["info"]
            complete_layout[name] = {
                "offset": current_heap_offset, "size": info["size"],
                "shape": info["shape"], "dtype": info["dtype"]
            }
            write_chunks.extend(chunk_tensor(np_arr.ctypes.data, current_heap_offset, info["size"]))
            current_heap_offset += int(math.ceil(info["size"] / 4096.0)) * 4096
            
        if current_heap_offset >= stack_start_bytes: raise MemoryError("Heap Area Full!")

        w_num = len(write_chunks)
        w_c_ptrs = (ctypes.c_void_p * w_num)()
        w_c_offs = (ctypes.c_uint64 * w_num)()
        w_c_sizes = (ctypes.c_size_t * w_num)()
        for i, (p, o, s) in enumerate(write_chunks):
            w_c_ptrs[i] = p
            w_c_offs[i] = ctypes.c_uint64(o.value)
            w_c_sizes[i] = s

        rc = lib.npu_nvme_write_batch(ctx, w_c_ptrs, w_c_offs, w_c_sizes, w_num)
        if rc != 0: raise RuntimeError("Write Batch failed!")
        
        print(f"      -> SUCCESS! Written in {time.time()-t_write_start:.2f}s. End offset: {current_heap_offset / 1024**3:.2f} GB.")

        # ---------------------------------------------------------
        # Phase 4: Ping-Pong 提交新的元数据
        # ---------------------------------------------------------
        print("\n[4/4] Committing COMPLETE model metadata to Ledger...")
        meta_dict["checkpoints"][complete_key] = {
            "type": "COMPLETE", "chunk_size": CHUNK_SIZE,
            "rank_id": 0, "world_size": 1, "params": complete_layout
        }

        next_slot = 1 if active_slot == 0 else 0
        target_offset = META_SLOT_B_OFFSET if next_slot == 1 else META_SLOT_A_OFFSET
        meta_json = json.dumps(meta_dict).encode('utf-8')
        if len(meta_json) > META_SLOT_BYTES: raise RuntimeError("Metadata JSON too large!")
            
        meta_buf = ctypes.create_string_buffer(meta_json, META_SLOT_BYTES)
        if lib.npu_nvme_sync_meta_io(ctx, target_offset, META_SLOT_BYTES, 0, ctypes.c_void_p(ctypes.addressof(meta_buf))) != 0:
            raise RuntimeError("Meta write failed!")
        
        sb_write_buf = ctypes.create_string_buffer(4096)
        struct.pack_into("<8s I Q Q", sb_write_buf, 0, MAGIC_NUMBER, next_slot, total_bytes, stack_start_bytes)
        if lib.npu_nvme_sync_meta_io(ctx, SUPERBLOCK_OFFSET, 4096, 0, ctypes.c_void_p(ctypes.addressof(sb_write_buf))) != 0:
            raise RuntimeError("Superblock update failed!")

        print(f"\n[SUCCESS] Model '{shard_key}' promoted to '{complete_key}' in HEAP!")

    except Exception as e:
        print(f"\n[Fatal Error] Export failed: {e}")
    finally:
        lib.npu_nvme_cleanup(ctx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci_addr", type=str, default="0000:83:00.0")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--npu_id", type=int, default=0)
    args = parser.parse_args()
    export_to_heap(args.pci_addr, args.step, args.npu_id)