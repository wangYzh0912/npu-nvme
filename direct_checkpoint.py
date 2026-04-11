import ctypes
import math
import os
import pickle
import json
import struct
import time
import threading
from typing import List, Dict

import mindspore as ms
from mindspore import ops
import numpy as np

# ============================================================
# 裸盘物理布局常量 (全字节寻址 Byte Addressing)
# ============================================================
SUPERBLOCK_OFFSET = 0
META_SLOT_A_OFFSET = 4096
META_SLOT_B_OFFSET = 4096 + 400 * 1024
META_SLOT_BYTES = 400 * 1024
MAGIC_NUMBER = b"NPUNVME1"

# ============================================================
# 绑定 C 接口
# ============================================================
_LIB_PATH = os.path.join(os.path.dirname(__file__), "out", "lib", "libnpu_nvme.so")

try:
    lib = ctypes.CDLL(_LIB_PATH)
    class NPUNVMEContext(ctypes.Structure): pass

    lib.npu_nvme_init.argtypes = [
        ctypes.POINTER(ctypes.POINTER(NPUNVMEContext)), ctypes.c_char_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_char_p,
    ]
    lib.npu_nvme_init.restype = ctypes.c_int

    lib.npu_nvme_cleanup.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_cleanup.restype = None

    lib.npu_nvme_get_max_transfer.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_get_max_transfer.restype = ctypes.c_int

    lib.npu_nvme_get_total_blocks.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_get_total_blocks.restype = ctypes.c_uint64

    lib.npu_nvme_sync_meta_io.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p
    ]
    lib.npu_nvme_sync_meta_io.restype = ctypes.c_int

    lib.npu_nvme_write_batch.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
    ]
    lib.npu_nvme_write_batch.restype = ctypes.c_int

    lib.npu_nvme_read_batch.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
    ]
    lib.npu_nvme_read_batch.restype = ctypes.c_int

    if hasattr(lib, "npu_nvme_write_batch_host"):
        lib.npu_nvme_write_batch_host.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
        ]
        lib.npu_nvme_write_batch_host.restype = ctypes.c_int

    if hasattr(lib, "npu_nvme_read_batch_host"):
        lib.npu_nvme_read_batch_host.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
        ]
        lib.npu_nvme_read_batch_host.restype = ctypes.c_int

    if hasattr(lib, "npu_nvme_bind_thread"):
        lib.npu_nvme_bind_thread.argtypes = [ctypes.POINTER(NPUNVMEContext)]
        lib.npu_nvme_bind_thread.restype = ctypes.c_int

except OSError as e:
    print(f"[Warning] Failed to load {_LIB_PATH}. Error: {e}")

# ============================================================
# 工具：分块与合包 
# ============================================================
def build_chunks(params: List[Dict], chunk_size: int):
    chunks = []
    total_size = 0
    for p in params:
        ptr = p["ptr"]
        remaining = p["size"]
        inner_off = 0
        nvme_offset_bytes = p["offset"] 
        
        while remaining > 0:
            take = min(remaining, chunk_size)
            chunks.append((
                ctypes.c_void_p(ptr + inner_off),
                ctypes.c_uint64(nvme_offset_bytes),
                ctypes.c_size_t(take)  # 严格使用实际内存大小，绝不越界
            ))
            remaining -= take
            inner_off += take
            # 物理写入地址仍按 4KB 对齐步进，防止覆盖
            nvme_offset_bytes += int(math.ceil(take / 4096.0)) * 4096
            total_size += take
            
    return chunks, total_size

def rebuild_chunks_from_meta(models, params_meta: Dict, chunk_size: int):
    if not isinstance(models, (list, tuple)): models = [models]
    buffers = []

    def get_dev_ptr(p):
        ptr = 0
        try:
            data_obj = p.data if hasattr(p, "data") else p
            if hasattr(data_obj, "_data_ptr"):
                 if isinstance(p, ms.Parameter) and hasattr(p, "is_inited") and not p.is_inited:
                     p.init_data()
                 ptr = int(data_obj._data_ptr())
        except Exception:
            ptr = 0
        return ptr

    for model in models:
        if model is None or not hasattr(model, "parameters_and_names"): continue
        for name, param in model.parameters_and_names():
            if name not in params_meta: continue
            info = params_meta[name]

            dev_ptr = get_dev_ptr(param)
            use_dev = (dev_ptr != 0)
            np_arr = None
            ptr_val = dev_ptr

            if not use_dev:
                # 恢复真实精确大小，不加 4KB 垫片
                np_arr = np.empty(info["shape"], dtype=np.dtype(info["dtype"]))
                ptr_val = np_arr.ctypes.data

            buffers.append({
                "name": name, "ptr": ptr_val, "size": info["size"],
                "offset": info["offset"], "np_arr": np_arr,
                "param_ref": param, "use_dev": use_dev
            })

    buffers.sort(key=lambda x: x["offset"])

    dev_buffers = [b for b in buffers if b["use_dev"]]
    host_buffers = [b for b in buffers if not b["use_dev"]]

    dev_chunks, _ = build_chunks(dev_buffers, chunk_size)
    host_chunks, _ = build_chunks(host_buffers, chunk_size)

    return dev_chunks, host_chunks, buffers
    
# ============================================================
# DirectCheckpoint 主类
# ============================================================
class DirectCheckpoint:
    def __init__(
        self, nvme_addr: str = "0000:83:00.0", npu_device_id: int = 0, pipeline_depth: int = 4,
        requested_chunk_size: int = 4 * 1024 * 1024, enable_profiling: bool = False,
        profiling_dir: str = "./output/profiling", rank_id: int = 0, world_size: int = 1,
        base_offset_bytes: int = 0, shard_span_bytes: int = None, spdk_shm_id: int = 1,
        keep_last_n: int = 3, slot_size_gb: int = 50      
    ):
        self.ctx = ctypes.POINTER(NPUNVMEContext)()
        self.enable_profiling = enable_profiling
        self.profiling_dir = profiling_dir
        self.rank_id = rank_id
        self.world_size = world_size
        self.base_offset_bytes = base_offset_bytes
        self.shard_span_bytes = shard_span_bytes
        os.environ.setdefault("SPDK_SHM_ID", str(spdk_shm_id))

        self.keep_last_n = keep_last_n
        self.slot_bytes = slot_size_gb * 1024**3
        self.active_meta_slot = 0
        self.stack_start_bytes = 0
        self.meta_dict = {"checkpoints": {}}
        self.last_layout = []
        
        if self.enable_profiling:
            os.makedirs(self.profiling_dir, exist_ok=True)

        print(f"[DirectCheckpoint] loading so from {_LIB_PATH}")

        rc = lib.npu_nvme_init(
            ctypes.byref(self.ctx),
            nvme_addr.encode(),
            npu_device_id,
            pipeline_depth,
            requested_chunk_size,
            enable_profiling,
            self.profiling_dir.encode("utf-8")
        )
        if rc != 0:
            raise RuntimeError("npu_nvme_init failed")

        self.total_bytes = lib.npu_nvme_get_total_blocks(self.ctx)
        if self.total_bytes == 0:
            raise RuntimeError("Failed to get NVMe total bytes from hardware.")

        eff = lib.npu_nvme_get_max_transfer(self.ctx)
        self.chunk_size = requested_chunk_size
        print(f"[DirectCheckpoint] init ok. chunk={self.chunk_size/1024/1024:.2f}MB, rank={self.rank_id}/{self.world_size}")
        
        self.async_thread = None
        self.async_lock = threading.Lock()

        self._mount_filesystem()

    def _mount_filesystem(self):
        sb_buf = ctypes.create_string_buffer(4096)
        rc = lib.npu_nvme_sync_meta_io(self.ctx, SUPERBLOCK_OFFSET, 4096, 1, ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if rc != 0:
            raise RuntimeError("Failed to read Superblock.")

        header = struct.unpack("<8s I Q Q", sb_buf.raw[:28])
        magic = header[0]
        
        if magic != MAGIC_NUMBER:
            raise RuntimeError(f"Disk is NOT formatted! Found magic: {magic}. Please run format script.")
            
        self.active_meta_slot = header[1]
        self.stack_start_bytes = header[3]
        
        # 【核心防爆修复】：检查 Superblock 中的历史堆栈起点是否还能容纳当前的分布式阵列！
        total_stack_bytes = self.world_size * self.keep_last_n * self.slot_bytes
        if self.stack_start_bytes == 0 or (self.stack_start_bytes + total_stack_bytes > self.total_bytes):
            print(f"[Warning] Rank {self.rank_id}: Stale Superblock! Current config needs {total_stack_bytes/1024**3:.2f} GB, but disk bounds exceeded. Recalculating...", flush=True)
            self.stack_start_bytes = self.total_bytes - total_stack_bytes
            print(f"[Info] Stack dynamically re-allocated at disk end. Start Offset: {self.stack_start_bytes / 1024**3:.2f} GB")

        target_offset = META_SLOT_A_OFFSET if self.active_meta_slot == 0 else META_SLOT_B_OFFSET
        meta_buf = ctypes.create_string_buffer(META_SLOT_BYTES)
        lib.npu_nvme_sync_meta_io(self.ctx, target_offset, META_SLOT_BYTES, 1, ctypes.c_void_p(ctypes.addressof(meta_buf)))
        
        meta_str = meta_buf.value.decode('utf-8', errors='ignore').rstrip('\x00')
        if meta_str:
            try:
                self.meta_dict = json.loads(meta_str)
            except json.JSONDecodeError:
                self.meta_dict = {"checkpoints": {}}
                print("[Warning] Parsed metadata JSON was invalid. Starting fresh.")
            
        print(f"[Rank {self.rank_id}] FileSystem Mounted. Active Slot: {'A' if self.active_meta_slot==0 else 'B'}")

    def _get_current_slot_base_offset(self, step: int):
        slot_idx = step % self.keep_last_n
        rank_offset = self.rank_id * self.keep_last_n * self.slot_bytes
        slot_offset = slot_idx * self.slot_bytes
        return self.stack_start_bytes + rank_offset + slot_offset

    def _commit_metadata(self, step: int, layout: List[Dict]):
        if self.rank_id != 0:
            return

        ckpt_key = f"step_{step}"
        self.meta_dict["checkpoints"][ckpt_key] = {
            "type": "SHARD",
            "chunk_size": self.chunk_size,
            "rank_id": self.rank_id,
            "world_size": self.world_size,
            "params": {p["name"]: {
                "offset": p["offset"],
                "size": p["size"],
                "shape": p["shape"],
                "dtype": p["dtype"],
            } for p in layout}
        }
        
        saved_steps = []
        for k in self.meta_dict["checkpoints"].keys():
            if k.startswith("step_"):
                try:
                    saved_steps.append(int(k.split('_')[1]))
                except ValueError:
                    pass
        saved_steps = sorted(saved_steps)

        while len(saved_steps) > self.keep_last_n:
            oldest_step = saved_steps.pop(0)
            del self.meta_dict["checkpoints"][f"step_{oldest_step}"]

        next_slot = 1 if self.active_meta_slot == 0 else 0
        target_offset = META_SLOT_B_OFFSET if next_slot == 1 else META_SLOT_A_OFFSET
        
        meta_json = json.dumps(self.meta_dict).encode('utf-8')
        if len(meta_json) > META_SLOT_BYTES:
            raise RuntimeError(f"Metadata JSON exceeds allocated {META_SLOT_BYTES} bytes!")
            
        meta_buf = ctypes.create_string_buffer(meta_json, META_SLOT_BYTES)
        
        rc1 = lib.npu_nvme_sync_meta_io(self.ctx, target_offset, META_SLOT_BYTES, 0, ctypes.c_void_p(ctypes.addressof(meta_buf)))
        if rc1 != 0:
            raise RuntimeError(f"Fatal: Meta JSON write failed! C layer returned {rc1}.")
        
        sb_buf = ctypes.create_string_buffer(4096)
        struct.pack_into("<8s I Q Q", sb_buf, 0, MAGIC_NUMBER, next_slot, self.total_bytes, self.stack_start_bytes)
        rc2 = lib.npu_nvme_sync_meta_io(self.ctx, SUPERBLOCK_OFFSET, 4096, 0, ctypes.c_void_p(ctypes.addressof(sb_buf)))
        if rc2 != 0:
            raise RuntimeError("Fatal: Superblock write failed!")
        
        self.active_meta_slot = next_slot
        print(f"[DirectCkpt] Rank 0 Meta committed safely to Slot {'B' if next_slot == 1 else 'A'} (Superblock updated).", flush=True)

    def bind_thread(self):
        if hasattr(lib, "npu_nvme_bind_thread"):
            rc = lib.npu_nvme_bind_thread(self.ctx)
            return rc == 0
        return False

    def cleanup(self):
        if self.async_thread and self.async_thread.is_alive():
             self.async_thread.join()
        if self.ctx:
            lib.npu_nvme_cleanup(self.ctx)
            self.ctx = None

    def _prepare_params(self, models):
        if not isinstance(models, (list, tuple)): models = [models]
        params = []
        for model in models:
            if model is None or not hasattr(model, "parameters_and_names"): continue
            for name, p in model.parameters_and_names():
                ptr = 0
                np_arr = None
                try:
                    data_obj = p.data if hasattr(p, "data") else p
                    if hasattr(data_obj, "_data_ptr"):
                        if isinstance(p, ms.Parameter) and hasattr(p, "is_inited") and not p.is_inited:
                            ptr = 0
                        else:
                            ptr = int(data_obj._data_ptr())
                    if ptr == 0 and hasattr(data_obj, "device_address"):
                        dev_addr = getattr(data_obj, "device_address", None)
                        if dev_addr is not None and hasattr(dev_addr, "ptr"):
                            ptr = int(dev_addr.ptr)
                except Exception:
                    ptr = 0

                dtype_np = np.dtype(ms.dtype_to_nptype(p.dtype))
                if np.prod(p.shape) == 0: continue
                size = int(np.prod(p.shape)) * dtype_np.itemsize
                shape = list(p.shape)
                dtype = dtype_np.name

                if ptr == 0:
                    # 仅对标量(step/scale)等极小参数强制拷贝到 CPU
                    if size <= 8 or "step" in name.lower() or "scale" in name.lower():
                        np_arr = p.asnumpy()
                        ptr = np_arr.ctypes.data
                    else:
                        continue # 真正的幽灵大张量，直接安全忽略

                if ptr == 0: continue

                params.append({
                    "name": name, "ptr": ptr, "size": size,
                    "shape": shape, "dtype": dtype, "np_arr": np_arr, "param_ref": p
                })
        return params

    def save(self, model: ms.nn.Cell, step: int, meta_path: str = "checkpoint_meta.pkl", async_save: bool = False, commit_meta: bool = True):
        if async_save:
            return self.save_async(model, step, meta_path, commit_meta)

        t_start = time.time()
        params = self._prepare_params(model)
        t_prep = time.time()

        base_offset_bytes = self._get_current_slot_base_offset(step)
        print(f"[DirectCkpt] Rank {self.rank_id} saving step {step} to offset {base_offset_bytes / 1024**3:.2f} GB ...", flush=True)

        current_offset = base_offset_bytes
        layout, dev_params, host_params = [], [], []

        for p in params:
            aligned_bytes = int(math.ceil(p["size"] / 4096.0)) * 4096
            
            # 【物理保险丝】：严格检查是否越出了这张 NVMe 盘的绝对物理容量
            if current_offset + aligned_bytes > self.total_bytes:
                raise MemoryError(f"Rank {self.rank_id} CRITICAL: DMA Write will exceed disk physical capacity! Offset: {(current_offset + aligned_bytes)/1024**3:.2f}GB > Total: {self.total_bytes/1024**3:.2f}GB")

            if (current_offset - base_offset_bytes) + aligned_bytes > self.slot_bytes:
                raise MemoryError(f"Rank {self.rank_id} OOM! Tensor {p['name']} exceeds slot size.")

            p_record = {**p, "offset": current_offset}
            layout.append(p_record)

            # 客货分流：走主机内存的单独放入 host_params
            if p.get("np_arr") is not None:
                host_params.append(p_record)
            else:
                dev_params.append(p_record)

            current_offset += aligned_bytes

        t0 = time.time()
        total_written = 0

        # 1. 显存直通数据 (GPUDirect)
        dev_chunks, dev_sz = build_chunks(dev_params, self.chunk_size)
        if dev_chunks:
            num = len(dev_chunks)
            c_ptrs = (ctypes.c_void_p * num)()
            c_offs = (ctypes.c_uint64 * num)()
            c_sizes = (ctypes.c_size_t * num)()
            for i, (p, o, s) in enumerate(dev_chunks):
                c_ptrs[i], c_offs[i], c_sizes[i] = p, ctypes.c_uint64(o.value), s
            
            rc = lib.npu_nvme_write_batch(self.ctx, c_ptrs, c_offs, c_sizes, num)
            if rc != 0: raise RuntimeError(f"write_batch failed (rc={rc})")
            total_written += dev_sz

        # 2. 主机内存数据 (CPU Memory)
        host_chunks, host_sz = build_chunks(host_params, self.chunk_size)
        if host_chunks:
            if hasattr(lib, "npu_nvme_write_batch_host"):
                num = len(host_chunks)
                c_ptrs = (ctypes.c_void_p * num)()
                c_offs = (ctypes.c_uint64 * num)()
                c_sizes = (ctypes.c_size_t * num)()
                for i, (p, o, s) in enumerate(host_chunks):
                    c_ptrs[i], c_offs[i], c_sizes[i] = p, ctypes.c_uint64(o.value), s
                
                rc = lib.npu_nvme_write_batch_host(self.ctx, c_ptrs, c_offs, c_sizes, num)
                if rc != 0: raise RuntimeError(f"write_batch_host failed (rc={rc})")
                total_written += host_sz

        t1 = time.time()
        real_time = t1 - t_start
        bw = total_written / 1024 / 1024 / real_time if real_time > 0 else 0

        self.last_layout = layout

        # 分布式受控盖章
        if commit_meta:
            self._commit_metadata(step, layout)

        with open(meta_path, "wb") as f:
            pickle.dump(self.meta_dict, f)

        stats = {
            "prep_time": t_prep - t_start,
            "write_time": t1 - t0,
            "total_time": real_time,
            "bw_pure": total_written/1024/1024/(t1-t0) if (t1-t0) > 0 else 0,
            "bw_e2e": bw
        }
        return total_written, len(dev_chunks) + len(host_chunks), real_time, bw, stats

    def save_async(self, models, step: int, meta_path: str, commit_meta: bool = True):
        t_start = time.time()
        if self.async_thread and self.async_thread.is_alive():
            self.async_thread.join()
            
        t_wait = time.time()
        models_list = [models] if not isinstance(models, (list, tuple)) else models
        host_snapshot = []
        total_bytes_copied = 0
        
        for model in models_list:
            if model is None or not hasattr(model, "parameters_and_names"): continue
            for name, p in model.parameters_and_names():
                if not hasattr(p, "asnumpy"): continue
                np_arr = p.asnumpy()
                total_bytes_copied += np_arr.nbytes
                host_snapshot.append({
                    "name": name, "np_arr": np_arr, "ptr": np_arr.ctypes.data,
                    "size": np_arr.nbytes, "shape": list(p.shape), "dtype": str(np_arr.dtype.name)
                })
        
        t_snapshot = time.time()
        snapshot_time = max(t_snapshot - t_wait, 0.0001)
        snapshot_bw = total_bytes_copied / 1024 / 1024 / snapshot_time
        
        self.async_thread = threading.Thread(
            target=self._background_write_worker,
            args=(host_snapshot, meta_path, total_bytes_copied, step, commit_meta)
        )
        self.async_thread.start()
        
        return total_bytes_copied, len(host_snapshot), snapshot_time, snapshot_bw, {
            "prep_time": t_wait - t_start, "write_time": 0.0,
            "total_time": snapshot_time, "bw_pure": snapshot_bw, "bw_e2e": snapshot_bw
        }

    def _background_write_worker(self, snapshot_params, meta_path, total_size, step, commit_meta):
        t0 = time.time()
        try:
            base_offset_bytes = self._get_current_slot_base_offset(step)
            current_offset = base_offset_bytes
            layout = []
            
            for p in snapshot_params:
                aligned_bytes = int(math.ceil(p["size"] / 4096.0)) * 4096
                p["offset"] = current_offset
                layout.append(p)
                current_offset += aligned_bytes
                
            chunks, total = build_chunks(layout, self.chunk_size)
            
            num = len(chunks)
            c_ptrs = (ctypes.c_void_p * num)()
            c_offs = (ctypes.c_uint64 * num)()
            c_sizes = (ctypes.c_size_t * num)()
            for i, (p, o, s) in enumerate(chunks):
                c_ptrs[i] = p; c_offs[i] = ctypes.c_uint64(o.value); c_sizes[i] = s
                
            if hasattr(lib, "npu_nvme_write_batch_host"):
                rc = lib.npu_nvme_write_batch_host(self.ctx, c_ptrs, c_offs, c_sizes, num)
            else:
                rc = -1
            if rc != 0: return

            self.last_layout = layout

            if commit_meta:
                self._commit_metadata(step, layout)
            
            with open(meta_path, "wb") as f:
                pickle.dump(self.meta_dict, f)
                
        except Exception as e:
            print(f"[AsyncWorker] Exception: {e}", flush=True)

    def load(self, model: ms.nn.Cell, step: int = None, meta_path: str = "checkpoint_meta.pkl"):
        t_start = time.time()

        if step is not None:
            ckpt_key = f"step_{step}"
        else:
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self.meta_dict = pickle.load(f)
            valid_keys = [k for k in self.meta_dict.get("checkpoints", {}).keys() if "complete" not in k]
            if not valid_keys:
                raise FileNotFoundError("No checkpoints found in Meta Dictionary!")
            ckpt_key = sorted(valid_keys, key=lambda x: int(x.split('_')[1]))[-1]

        if ckpt_key not in self.meta_dict.get("checkpoints", {}):
            raise FileNotFoundError(f"Checkpoint for {ckpt_key} not found!")

        meta_info = self.meta_dict["checkpoints"][ckpt_key]
        chunk_size = min(meta_info.get("chunk_size", self.chunk_size), self.chunk_size)

        t_rebuild = time.time()
        dev_chunks, host_chunks, buffers = rebuild_chunks_from_meta(model, meta_info["params"], chunk_size)
        t_rebuild_end = time.time()

        t0 = time.time()
        total_read = 0

        # Load NPU Data
        if dev_chunks:
            num = len(dev_chunks)
            c_ptrs = (ctypes.c_void_p * num)()
            c_offs = (ctypes.c_uint64 * num)()
            c_sizes = (ctypes.c_size_t * num)()
            for i, (p, o, s) in enumerate(dev_chunks):
                c_ptrs[i], c_offs[i], c_sizes[i] = p, ctypes.c_uint64(o.value), s

            rc = lib.npu_nvme_read_batch(self.ctx, c_ptrs, c_offs, c_sizes, num)
            if rc != 0: raise RuntimeError("read_batch failed")
            total_read += sum(c[2].value for c in dev_chunks)

        # Load Host Data
        if host_chunks:
            if hasattr(lib, "npu_nvme_read_batch_host"):
                num = len(host_chunks)
                c_ptrs = (ctypes.c_void_p * num)()
                c_offs = (ctypes.c_uint64 * num)()
                c_sizes = (ctypes.c_size_t * num)()
                for i, (p, o, s) in enumerate(host_chunks):
                    c_ptrs[i], c_offs[i], c_sizes[i] = p, ctypes.c_uint64(o.value), s

                rc = lib.npu_nvme_read_batch_host(self.ctx, c_ptrs, c_offs, c_sizes, num)
                if rc != 0: raise RuntimeError("read_batch_host failed")
                total_read += sum(c[2].value for c in host_chunks)

        t1 = time.time()
        t_update = time.time()

        for buf in buffers:
            if buf.get("use_dev", False): continue
            param = buf["param_ref"]
            if buf["np_arr"] is not None:
                tensor = ms.Tensor(buf["np_arr"], dtype=param.dtype)
                ops.assign(param, tensor)
        t_end = time.time()

        pure_read_time = t1 - t0
        total_time = t_end - t_start
        bw_e2e = total_read / 1024 / 1024 / total_time if total_time > 0 else 0

        stats = {
            "prepare_time": t_rebuild_end - t_rebuild,
            "read_time": pure_read_time,
            "set_data_time": t_end - t_update,
            "total_time": total_time,
            "bw_pure": total_read / 1024 / 1024 / pure_read_time if pure_read_time > 0 else 0,
            "bw_e2e": bw_e2e
        }
        return total_read, len(dev_chunks) + len(host_chunks), total_time, bw_e2e, stats

    def wait_async_io(self):
        if hasattr(self, 'async_thread') and self.async_thread and self.async_thread.is_alive():
            self.async_thread.join()

    def commit_last_layout(self, step: int):
        if self.rank_id == 0 and hasattr(self, 'last_layout'):
            self._commit_metadata(step, self.last_layout)