"""Export model and optional NVMe data for analysis.

Usage:
- python python/export_model.py

Inputs:
- Model configuration and NVMe parameters in script.
Outputs:
- Exported artifacts under output/ or specified paths.
"""
import ctypes
import json
import struct
import argparse
import sys
import os
import math
import numpy as np
import time
import pickle
import gc

# -- Disk layout constants (byte-addressed) --
SUPERBLOCK_OFFSET  = 0
META_SLOT_A_OFFSET = 4096
META_SLOT_B_OFFSET = 4096 + 400 * 1024
META_SLOT_BYTES    = 400 * 1024
HEAP_START_OFFSET  = META_SLOT_B_OFFSET + META_SLOT_BYTES
MAGIC_NUMBER       = b"NPUNVME1"
CHUNK_SIZE         = 4 * 1024 * 1024

os.environ.setdefault("SPDK_SHM_ID", "1")

# -- C interface (reuse direct_checkpoint bindings) --
from direct_checkpoint import lib, NPUNVMEContext, build_chunks

def export_to_heap(pci_addr, target_step, world_size, meta_dir, npu_id=0):
    print(f"\n{'='*70}")
    print(f"NPUNVME Global Exporter: Aggregating {world_size}-Rank SHARDs to HEAP")
    print(f"{'='*70}")
    
    ctx = ctypes.POINTER(NPUNVMEContext)()
    ret = lib.npu_nvme_init(ctypes.byref(ctx), pci_addr.encode('utf-8'), npu_id, 4, CHUNK_SIZE, False, b".")
    if ret != 0:
        print("[Fatal] SPDK init failed.")
        sys.exit(1)

    try:
        # ---------------------------------------------------------
        # Collect shard maps from all ranks, build global tensor view
        # ---------------------------------------------------------
        print(f"[1/4] Scanning local meta directory: {meta_dir} ...")
        global_tensor_map = {}
        
        for r in range(world_size):
            meta_file = os.path.join(meta_dir, f"checkpoint_meta_rank{r}.pkl")
            if not os.path.exists(meta_file):
                raise FileNotFoundError(f"Missing meta file for Rank {r}: {meta_file}")
                
            with open(meta_file, "rb") as f:
                r_meta = pickle.load(f)
                
            shard_key = f"step_{target_step}"
            if shard_key not in r_meta.get("checkpoints", {}):
                raise ValueError(f"Target step '{shard_key}' not found in Rank {r} ledger!")
                
            params = r_meta["checkpoints"][shard_key]["params"]
            for name, info in params.items():
                if name not in global_tensor_map:
                    global_tensor_map[name] = []
        # Record the originating rank and physical info for this shard
                global_tensor_map[name].append({"rank": r, **info})

        print(f"      -> Found {len(global_tensor_map)} unique tensors across {world_size} ranks.")

        # ---------------------------------------------------------
        # Read superblock to verify disk format and get current layout
        # ---------------------------------------------------------
        sb_buf = ctypes.create_string_buffer(4096)
        if lib.npu_nvme_sync_meta_io(ctx, SUPERBLOCK_OFFSET, 4096, 1, ctypes.c_void_p(ctypes.addressof(sb_buf))) != 0:
            raise RuntimeError("Failed to read Superblock.")

        header = struct.unpack("<8s I Q Q", sb_buf.raw[:28])
        if header[0] != MAGIC_NUMBER: raise RuntimeError("Disk not formatted!")
        active_slot, total_bytes, stack_start_bytes = header[1], header[2], header[3]
        
        # 读取当前盘头 JSON（为了后续追加 Complete 记录）
        target_meta_offset = META_SLOT_A_OFFSET if active_slot == 0 else META_SLOT_B_OFFSET
        meta_buf = ctypes.create_string_buffer(META_SLOT_BYTES)
        lib.npu_nvme_sync_meta_io(ctx, target_meta_offset, META_SLOT_BYTES, 1, ctypes.c_void_p(ctypes.addressof(meta_buf)))
        nvme_meta_dict = json.loads(meta_buf.value.decode('utf-8', errors='ignore').rstrip('\x00'))

        # ---------------------------------------------------------
        # Stream tensors from STACK to HEAP, process one at a time to avoid OOM
        # ---------------------------------------------------------
        print("\n[2/4] & [3/4] Streaming Tensors from STACK -> HEAP (OOM Safe)...")
        t_stream_start = time.time()
        
        complete_layout = {}
        current_heap_offset = HEAP_START_OFFSET 
        
        # 按参数名排序，保证每次导出的模型布局一致
        sorted_names = sorted(global_tensor_map.keys())
        
        for idx, name in enumerate(sorted_names):
            slices = global_tensor_map[name]
            
            # Dedup + concat strategy:
            # 1. Different NVMe offsets → TP (tensor parallel) concat needed
            # 2. Same offset + size → DP (data parallel) duplicate, keep only first
            unique_slices = []
            seen_offsets = set()
            for s in slices:
                if s["offset"] not in seen_offsets:
                    unique_slices.append(s)
                    seen_offsets.add(s["offset"])
            
            # 准备读取所有独特分片
            np_parts = []
            for s in unique_slices:
                np_arr = np.empty(s["shape"], dtype=np.dtype(s["dtype"]))
                chunks, _ = build_chunks_host(np_arr.ctypes.data, s["offset"], s["size"], CHUNK_SIZE)

                num = len(chunks)
                c_ptrs, c_offs, c_sizes = (ctypes.c_void_p * num)(), (ctypes.c_uint64 * num)(), (ctypes.c_size_t * num)()
                for i, (p, o, sz, _name) in enumerate(chunks):
                    c_ptrs[i], c_offs[i], c_sizes[i] = p, ctypes.c_uint64(o.value), sz
                
                if lib.npu_nvme_read_batch(ctx, c_ptrs, c_offs, c_sizes, num) != 0:
                    raise RuntimeError(f"Read failed for {name} from Rank {s['rank']}")
                np_parts.append(np_arr)
            
            # Core concat logic
            if len(np_parts) == 1:
                final_arr = np_parts[0]
            else:
                # 如果有多个 unique 碎片，默认按 axis=0 进行拼接 (针对 MP并行的简单启发式)
                # (如果是高级场景，可根据 name 中的 q_proj/k_proj 等字眼动态决定 axis=0 或 1)
                final_arr = np.concatenate(np_parts, axis=0)
            
            final_size = final_arr.nbytes
            final_shape = list(final_arr.shape)
            
            # 写入 HEAP 区
            if current_heap_offset + final_size >= stack_start_bytes:
                raise MemoryError("CRITICAL: Heap Area is Full! Cannot export model.")
                
            write_chunks, _ = build_chunks_host(final_arr.ctypes.data, current_heap_offset, final_size, CHUNK_SIZE)
            w_num = len(write_chunks)
            w_c_ptrs, w_c_offs, w_c_sizes = (ctypes.c_void_p * w_num)(), (ctypes.c_uint64 * w_num)(), (ctypes.c_size_t * w_num)()
            for i, (p, o, sz, _name) in enumerate(write_chunks):
                w_c_ptrs[i], w_c_offs[i], w_c_sizes[i] = p, ctypes.c_uint64(o.value), sz
                
            if lib.npu_nvme_write_batch(ctx, w_c_ptrs, w_c_offs, w_c_sizes, w_num) != 0:
                raise RuntimeError(f"Write failed for {name} to HEAP")
            
            # 记录到最终完整的 Layout
            complete_layout[name] = {
                "offset": current_heap_offset,
                "size": final_size,
                "shape": final_shape,
                "dtype": str(final_arr.dtype.name)
            }
            current_heap_offset += int(math.ceil(final_size / 4096.0)) * 4096
            
            # 释放内存，防止 OOM
            del np_parts
            del final_arr
            
            if idx % 50 == 0:
                print(f"      ... Processed {idx}/{len(sorted_names)} tensors. Current Heap: {current_heap_offset/1024**3:.2f}GB")

        # 强制垃圾回收
        gc.collect()
        print(f"      -> SUCCESS! Exported {len(sorted_names)} tensors in {time.time()-t_stream_start:.2f}s. End offset: {current_heap_offset / 1024**3:.2f} GB.")

        # ---------------------------------------------------------
        # Update superblock ledger with COMPLETE model metadata
        # ---------------------------------------------------------
        print("\n[4/4] Committing COMPLETE model metadata to Ledger...")
        complete_key = f"complete_step_{target_step}"
        nvme_meta_dict["checkpoints"][complete_key] = {
            "type": "COMPLETE", "chunk_size": CHUNK_SIZE,
            "rank_id": 0, "world_size": 1, "params": complete_layout
        }

        next_slot = 1 if active_slot == 0 else 0
        target_offset = META_SLOT_B_OFFSET if next_slot == 1 else META_SLOT_A_OFFSET
        meta_json = json.dumps(nvme_meta_dict).encode('utf-8')
        if len(meta_json) > META_SLOT_BYTES: raise RuntimeError("Metadata JSON too large!")
            
        meta_buf = ctypes.create_string_buffer(meta_json, META_SLOT_BYTES)
        if lib.npu_nvme_sync_meta_io(ctx, target_offset, META_SLOT_BYTES, 0, ctypes.c_void_p(ctypes.addressof(meta_buf))) != 0:
            raise RuntimeError("Meta write failed!")
        
        sb_write_buf = ctypes.create_string_buffer(4096)
        struct.pack_into("<8s I Q Q", sb_write_buf, 0, MAGIC_NUMBER, next_slot, total_bytes, stack_start_bytes)
        if lib.npu_nvme_sync_meta_io(ctx, SUPERBLOCK_OFFSET, 4096, 0, ctypes.c_void_p(ctypes.addressof(sb_write_buf))) != 0:
            raise RuntimeError("Superblock update failed!")

        print(f"\n[SUCCESS] Distributed Model '{shard_key}' seamlessly promoted to '{complete_key}' in HEAP!")

    except Exception as e:
        print(f"\n[Fatal Error] Export failed: {e}")
    finally:
        lib.npu_nvme_cleanup(ctx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci_addr", type=str, default="0000:83:00.0")
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--world_size", type=int, default=8, help="Number of ranks used during training")
    parser.add_argument("--meta_dir", type=str, default="./checkpoint_meta", help="Directory containing the local .pkl files")
    parser.add_argument("--npu_id", type=int, default=0)
    args = parser.parse_args()
    
    export_to_heap(args.pci_addr, args.step, args.world_size, args.meta_dir, args.npu_id)