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

os.environ.setdefault("SPDK_SHM_ID", "1")

# ============================================================
# 绑定 C 接口
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
        nvme_off += int(math.ceil(take / 4096.0)) * 4096
    return chunks

def export_to_heap(pci_addr, target_step, world_size, meta_dir, npu_id=0):
    print(f"\n{'='*70}")
    print(f"🚀 NPUNVME Global Exporter: Aggregating {world_size}-Rank SHARDs to HEAP")
    print(f"{'='*70}")
    
    ctx = ctypes.POINTER(NPUNVMEContext)()
    ret = lib.npu_nvme_init(ctypes.byref(ctx), pci_addr.encode('utf-8'), npu_id, 4, CHUNK_SIZE, False, b".")
    if ret != 0:
        print("[Fatal] SPDK init failed.")
        sys.exit(1)

    try:
        # ---------------------------------------------------------
        # Phase 1: 离线收集 8 张“藏宝图”，构建全局张量视图
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
                # 记录该张量碎片的来源 rank 和物理信息
                global_tensor_map[name].append({"rank": r, **info})

        print(f"      -> Found {len(global_tensor_map)} unique tensors across {world_size} ranks.")

        # ---------------------------------------------------------
        # Phase 2: 读取底层超级块，确认边界
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
        # Phase 3: 流式合并 (防 OOM)，逐个 Tensor 搬运
        # ---------------------------------------------------------
        print("\n[2/4] & [3/4] Streaming Tensors from STACK -> HEAP (OOM Safe)...")
        t_stream_start = time.time()
        
        complete_layout = {}
        current_heap_offset = HEAP_START_OFFSET 
        
        # 按参数名排序，保证每次导出的模型布局一致
        sorted_names = sorted(global_tensor_map.keys())
        
        for idx, name in enumerate(sorted_names):
            slices = global_tensor_map[name]
            
            # 【策略推断】去重与拼接逻辑
            # 1. 如果有多个 Slice，但它们的 NVMe Offset 是不同的，说明是 TP (张量并行) 需要拼接
            # 2. 如果 Offset 和 Size 完全一样，说明是 DP (数据并行) 冗余，只取第一个即可
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
                chunks = chunk_tensor(np_arr.ctypes.data, s["offset"], s["size"])
                
                num = len(chunks)
                c_ptrs, c_offs, c_sizes = (ctypes.c_void_p * num)(), (ctypes.c_uint64 * num)(), (ctypes.c_size_t * num)()
                for i, (p, o, sz) in enumerate(chunks):
                    c_ptrs[i], c_offs[i], c_sizes[i] = p, ctypes.c_uint64(o.value), sz
                
                if lib.npu_nvme_read_batch(ctx, c_ptrs, c_offs, c_sizes, num) != 0:
                    raise RuntimeError(f"Read failed for {name} from Rank {s['rank']}")
                np_parts.append(np_arr)
            
            # 【核心拼接】
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
                
            write_chunks = chunk_tensor(final_arr.ctypes.data, current_heap_offset, final_size)
            w_num = len(write_chunks)
            w_c_ptrs, w_c_offs, w_c_sizes = (ctypes.c_void_p * w_num)(), (ctypes.c_uint64 * w_num)(), (ctypes.c_size_t * w_num)()
            for i, (p, o, sz) in enumerate(write_chunks):
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
        # Phase 4: 更新盘头超级块的 Complete 账本
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