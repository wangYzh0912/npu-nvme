import ctypes
import math
import os
import pickle
import json
import struct
import time
import threading
from typing import List, Dict
import copy

import mindspore as ms
from mindspore import ops
import numpy as np


# ============================================================
# 裸盘物理布局常量
# ============================================================
BLOCK_SIZE = 4096
SUPERBLOCK_LBA = 0
META_SLOT_A_LBA = 1
META_SLOT_B_LBA = 101
META_SLOT_BLOCKS = 100       # 每个 Meta Slot 约 400KB，足够存巨型 JSON
MAGIC_NUMBER = b"NPUNVME1"   # 防呆校验魔数

# ============================================================
# 绑定 C 接口 (统一部署于文件头部)
# ============================================================
_LIB_PATH = os.path.join(os.path.dirname(__file__), "out", "lib", "libnpu_nvme.so")

try:
    lib = ctypes.CDLL(_LIB_PATH)
    
    class NPUNVMEContext(ctypes.Structure):
        pass

    # init
    lib.npu_nvme_init.argtypes = [
        ctypes.POINTER(ctypes.POINTER(NPUNVMEContext)),
        ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_char_p,
    ]
    lib.npu_nvme_init.restype = ctypes.c_int

    # cleanup
    lib.npu_nvme_cleanup.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_cleanup.restype = None

    # get_max_transfer
    lib.npu_nvme_get_max_transfer.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_get_max_transfer.restype = ctypes.c_int

    # get_total_blocks (New)
    lib.npu_nvme_get_total_blocks.argtypes = [ctypes.POINTER(NPUNVMEContext)]
    lib.npu_nvme_get_total_blocks.restype = ctypes.c_uint64

    # sync_meta_io (New: Metadata Control Plane)
    lib.npu_nvme_sync_meta_io.argtypes = [
        ctypes.POINTER(NPUNVMEContext), ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p
    ]
    lib.npu_nvme_sync_meta_io.restype = ctypes.c_int

    # write_batch / read_batch
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

    # write_batch_host
    if hasattr(lib, "npu_nvme_write_batch_host"):
        lib.npu_nvme_write_batch_host.argtypes = [
            ctypes.POINTER(NPUNVMEContext), ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int
        ]
        lib.npu_nvme_write_batch_host.restype = ctypes.c_int

    # bind_thread
    if hasattr(lib, "npu_nvme_bind_thread"):
        lib.npu_nvme_bind_thread.argtypes = [ctypes.POINTER(NPUNVMEContext)]
        lib.npu_nvme_bind_thread.restype = ctypes.c_int

except OSError as e:
    print(f"[Warning] Failed to load {_LIB_PATH}. Error: {e}")


# ============================================================
# 工具：分块与合包 (适配新 LBA 逻辑)
# ============================================================
def build_chunks(params: List[Dict], chunk_size: int):
    """
    基于预先计算好的 param offset 进行切块。
    """
    chunks = []
    total_size = 0
    for p in params:
        ptr = p["ptr"]
        remaining = p["size"]
        inner_off = 0
        nvme_offset = p["offset"]  # 现在由调用方直接计算好绝对物理偏移
        
        while remaining > 0:
            take = min(remaining, chunk_size)
            chunks.append((
                ctypes.c_void_p(ptr + inner_off),
                ctypes.c_uint64(nvme_offset),
                ctypes.c_size_t(take)
            ))
            remaining -= take
            inner_off += take
            nvme_offset += int(math.ceil(take / 4096.0) * 4096)
            total_size += take
            
    return chunks, total_size

def rebuild_chunks_from_meta(models, params_meta: Dict, chunk_size: int):
    """从元数据恢复读取块布局（保持原有纯粹逻辑）"""
    if not isinstance(models, (list, tuple)):
        models = [models]

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
        if model is None or not hasattr(model, "parameters_and_names"):
            continue

        for name, param in model.parameters_and_names():
            if name not in params_meta:
                continue
            info = params_meta[name]
            
            dev_ptr = get_dev_ptr(param)
            use_dev = (dev_ptr != 0)
            np_arr = None
            ptr_val = dev_ptr
            
            if not use_dev:
                np_arr = np.empty(info["shape"], dtype=np.dtype(info["dtype"]))
                ptr_val = np_arr.ctypes.data
                
            buffers.append({
                "name": name,
                "ptr": ptr_val,
                "size": info["size"],
                "offset": info["offset"],
                "np_arr": np_arr,
                "param_ref": param,
                "use_dev": use_dev
            })

    buffers.sort(key=lambda x: x["offset"])

    chunks = []
    for p in buffers:
        ptr = p["ptr"]
        remaining = p["size"]
        nvme_off = p["offset"]
        inner_off = 0
        while remaining > 0:
            take = min(remaining, chunk_size)
            chunks.append((
                ctypes.c_void_p(ptr + inner_off),
                ctypes.c_uint64(nvme_off),
                ctypes.c_size_t(take)
            ))
            remaining -= take
            inner_off += take
            nvme_off += int(math.ceil(take / 4096.0) * 4096)
    return chunks, buffers


# ============================================================
# DirectCheckpoint
# ============================================================
class DirectCheckpoint:
    def __init__(
        self,
        nvme_addr: str = "0000:83:00.0",
        npu_device_id: int = 0,
        pipeline_depth: int = 4,
        requested_chunk_size: int = 4 * 1024 * 1024,
        enable_profiling: bool = False,
        profiling_dir: str = "./output/profiling",
        rank_id: int = 0,
        world_size: int = 1,
        base_offset_bytes: int = 0,
        shard_span_bytes: int = None,
        spdk_shm_id: int = 1,
        keep_last_n: int = 3,       # 新增：保留最新 N 份
        slot_size_gb: int = 50      # 新增：单个槽位容量上限
    ):
        self.ctx = ctypes.POINTER(NPUNVMEContext)()
        self.enable_profiling = enable_profiling
        self.profiling_dir = profiling_dir
        self.rank_id = rank_id
        self.world_size = world_size
        self.base_offset_bytes = base_offset_bytes
        self.shard_span_bytes = shard_span_bytes
        os.environ.setdefault("SPDK_SHM_ID", str(spdk_shm_id))

        # 堆栈与元数据参数
        self.keep_last_n = keep_last_n
        self.slot_blocks = (slot_size_gb * 1024**3) // BLOCK_SIZE
        self.active_meta_slot = 0
        self.stack_start_lba = 0
        self.meta_dict = {"checkpoints": {}}
        
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

        self.total_blocks = lib.npu_nvme_get_total_blocks(self.ctx)
        if self.total_blocks == 0:
            raise RuntimeError("Failed to get NVMe total blocks from hardware.")

        eff = lib.npu_nvme_get_max_transfer(self.ctx)
        self.chunk_size = requested_chunk_size
        print(f"[DirectCheckpoint] init ok. chunk={self.chunk_size/1024/1024:.2f}MB, rank={self.rank_id}/{self.world_size}")
        
        self.async_thread = None
        self.async_lock = threading.Lock()

        # 初始化并挂载裸盘文件系统
        self._mount_filesystem()

    def _mount_filesystem(self):
        """挂载裸盘：读取超级块和元数据字典，防呆校验"""
        sb_buf = ctypes.create_string_buffer(BLOCK_SIZE)
        rc = lib.npu_nvme_sync_meta_io(self.ctx, SUPERBLOCK_LBA, 1, 1, ctypes.byref(sb_buf))
        if rc != 0:
            raise RuntimeError("Failed to read Superblock.")

        header = struct.unpack("<8s I Q Q", sb_buf.raw[:28])
        magic = header[0]
        
        if magic != MAGIC_NUMBER:
            raise RuntimeError(f"Disk is NOT formatted! Found magic: {magic}. Please run format script.")
            
        self.active_meta_slot = header[1]
        self.stack_start_lba = header[3]
        
        # 兜底：如果格式化时未指定栈顶，这里自动从盘尾分配
        if self.stack_start_lba == 0:
            total_stack_blocks = self.world_size * self.keep_last_n * self.slot_blocks
            self.stack_start_lba = self.total_blocks - total_stack_blocks
            print(f"[Warning] Auto-calculating stack_start_lba to {self.stack_start_lba}")

        slot_lba = META_SLOT_A_LBA if self.active_meta_slot == 0 else META_SLOT_B_LBA
        meta_buf = ctypes.create_string_buffer(META_SLOT_BLOCKS * BLOCK_SIZE)
        lib.npu_nvme_sync_meta_io(self.ctx, slot_lba, META_SLOT_BLOCKS, 1, ctypes.byref(meta_buf))
        
        meta_str = meta_buf.value.decode('utf-8').rstrip('\x00')
        if meta_str:
            self.meta_dict = json.loads(meta_str)
            
        print(f"[Rank {self.rank_id}] FileSystem Mounted. Stack LBA: {self.stack_start_lba}, Active Slot: {'A' if self.active_meta_slot==0 else 'B'}")

    def _get_current_slot_base_lba(self, step: int):
        """零通信偏移量计算：本地盲算专属 LBA 槽位起始地址"""
        slot_idx = step % self.keep_last_n
        rank_offset = self.rank_id * self.keep_last_n * self.slot_blocks
        slot_offset = slot_idx * self.slot_blocks
        return self.stack_start_lba + rank_offset + slot_offset

    def _commit_metadata(self, step: int, layout: List[Dict]):
        """Rank 0 专属：同步 Ping-Pong 强一致性落盘"""
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
        
        # 清理超出 keep_last_n 的旧版本元数据
        old_step = step - self.keep_last_n
        if f"step_{old_step}" in self.meta_dict["checkpoints"]:
            del self.meta_dict["checkpoints"][f"step_{old_step}"]

        # 算出下一个备用槽位
        next_slot = 1 if self.active_meta_slot == 0 else 0
        target_lba = META_SLOT_B_LBA if next_slot == 1 else META_SLOT_A_LBA
        
        meta_json = json.dumps(self.meta_dict).encode('utf-8')
        if len(meta_json) > META_SLOT_BLOCKS * BLOCK_SIZE:
            raise RuntimeError("Metadata JSON exceeds allocated capacity!")
            
        meta_buf = ctypes.create_string_buffer(meta_json, META_SLOT_BLOCKS * BLOCK_SIZE)
        
        # 1. 覆写备用 Meta Slot
        lib.npu_nvme_sync_meta_io(self.ctx, target_lba, META_SLOT_BLOCKS, 0, ctypes.byref(meta_buf))
        
        # 2. 原子更新 Superblock
        sb_buf = ctypes.create_string_buffer(BLOCK_SIZE)
        struct.pack_into("<8s I Q Q", sb_buf, 0, MAGIC_NUMBER, next_slot, self.total_blocks, self.stack_start_lba)
        lib.npu_nvme_sync_meta_io(self.ctx, SUPERBLOCK_LBA, 1, 0, ctypes.byref(sb_buf))
        
        self.active_meta_slot = next_slot

    def bind_thread(self):
        if hasattr(lib, "npu_nvme_bind_thread"):
            rc = lib.npu_nvme_bind_thread(self.ctx)
            if rc != 0:
                print(f"[DirectCheckpoint] bind_thread failed rc={rc}", flush=True)
                return False
            return True
        return False

    def cleanup(self):
        if self.async_thread and self.async_thread.is_alive():
             self.async_thread.join()
        if self.ctx:
            lib.npu_nvme_cleanup(self.ctx)
            self.ctx = None

    def _prepare_params(self, models):
        if not isinstance(models, (list, tuple)):
            models = [models]
        params = []
        for model in models:
            if model is None or not hasattr(model, "parameters_and_names"):
                continue

            for name, p in model.parameters_and_names():
                ptr = 0
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
                    np_arr = p.asnumpy()
                    ptr = np_arr.ctypes.data
                    params.append({
                        "name": name, "ptr": ptr, "size": size,
                        "shape": shape, "dtype": dtype, "np_arr": np_arr, "param_ref": p
                    })
                else:
                    params.append({
                        "name": name, "ptr": ptr, "size": size,
                        "shape": shape, "dtype": dtype, "np_arr": None, "param_ref": p
                    })
        return params

    def save(self, model: ms.nn.Cell, step: int, meta_path: str = "checkpoint_meta.pkl", async_save: bool = False):
        if async_save:
            return self.save_async(model, step, meta_path)
            
        t_start = time.time()
        params = self._prepare_params(model)
        t_prep = time.time()
        
        # 1. 本地盲算绝对起始 LBA
        base_lba = self._get_current_slot_base_lba(step)
        current_lba = base_lba
        layout = []
        
        for p in params:
            aligned_blocks = int(math.ceil(p["size"] / 4096.0))
            if (current_lba - base_lba) + aligned_blocks > self.slot_blocks:
                raise MemoryError(f"Rank {self.rank_id} OOM! Tensor {p['name']} exceeds slot size.")
            
            layout.append({**p, "offset": current_lba * BLOCK_SIZE})
            current_lba += aligned_blocks

        chunks, total = build_chunks(layout, self.chunk_size)
        
        num = len(chunks)
        c_ptrs = (ctypes.c_void_p * num)()
        c_offs = (ctypes.c_uint64 * num)()
        c_sizes = (ctypes.c_size_t * num)()
        for i, (p, o, s) in enumerate(chunks):
            c_ptrs[i] = p
            c_offs[i] = ctypes.c_uint64(o.value) # 不再加 base_offset_bytes，绝对 LBA 直写
            c_sizes[i] = s

        t0 = time.time()
        rc = lib.npu_nvme_write_batch(self.ctx, c_ptrs, c_offs, c_sizes, num)
        if rc != 0:
            raise RuntimeError("write_batch failed")
        t1 = time.time()
        
        real_time = t1 - t_start
        bw = total / 1024 / 1024 / real_time
        
        # 2. 完全同步的元数据落盘 (替代掉以前只写本地 ext4 的逻辑)
        self._commit_metadata(step, layout)
        
        # 兼顾备份的本地 pkl 写入 (供你的外围调试脚本使用)
        with open(meta_path, "wb") as f:
            pickle.dump(self.meta_dict, f)
            
        stats = {
            "prep_time": t_prep - t_start,
            "write_time": t1 - t0,
            "total_time": real_time,
            "bw_pure": total/1024/1024/(t1-t0) if (t1-t0) > 0 else 0,
            "bw_e2e": bw
        }
        return total, len(chunks), real_time, bw, stats

    def save_async(self, models, step: int, meta_path: str):
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
            args=(host_snapshot, meta_path, total_bytes_copied, step)
        )
        self.async_thread.start()
        
        return total_bytes_copied, len(host_snapshot), snapshot_time, snapshot_bw, {
            "prep_time": t_wait - t_start, "write_time": 0.0,
            "total_time": snapshot_time, "bw_pure": snapshot_bw, "bw_e2e": snapshot_bw
        }

    def _background_write_worker(self, snapshot_params, meta_path, total_size, step):
        t0 = time.time()
        try:
            base_lba = self._get_current_slot_base_lba(step)
            current_lba = base_lba
            layout = []
            
            for p in snapshot_params:
                aligned_blocks = int(math.ceil(p["size"] / 4096.0))
                p["offset"] = current_lba * BLOCK_SIZE
                layout.append(p)
                current_lba += aligned_blocks
                
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

            # 同步落盘元数据
            self._commit_metadata(step, layout)
            
            with open(meta_path, "wb") as f:
                pickle.dump(self.meta_dict, f)
                
        except Exception as e:
            print(f"[AsyncWorker] Exception: {e}", flush=True)

    def load(self, model: ms.nn.Cell, step: int, meta_path: str = "checkpoint_meta.pkl"):
        t_start = time.time()
        
        ckpt_key = f"step_{step}"
        if ckpt_key not in self.meta_dict.get("checkpoints", {}):
            # 若内存无，尝试从 pkl 读取兜底
            if os.path.exists(meta_path):
                with open(meta_path, "rb") as f:
                    self.meta_dict = pickle.load(f)
            if ckpt_key not in self.meta_dict.get("checkpoints", {}):
                raise FileNotFoundError(f"Checkpoint for step {step} not found in Meta Dictionary!")
                
        meta_info = self.meta_dict["checkpoints"][ckpt_key]
        chunk_size = min(meta_info.get("chunk_size", self.chunk_size), self.chunk_size)

        t_rebuild = time.time()
        chunks, buffers = rebuild_chunks_from_meta(model, meta_info["params"], chunk_size)
        t_rebuild_end = time.time()
        
        num = len(chunks)
        c_ptrs = (ctypes.c_void_p * num)()
        c_offs = (ctypes.c_uint64 * num)()
        c_sizes = (ctypes.c_size_t * num)()
        for i, (p, o, s) in enumerate(chunks):
            c_ptrs[i] = p
            c_offs[i] = ctypes.c_uint64(o.value)
            c_sizes[i] = s

        t0 = time.time()
        rc = lib.npu_nvme_read_batch(self.ctx, c_ptrs, c_offs, c_sizes, num)
        if rc != 0:
            raise RuntimeError("read_batch failed")
        t1 = time.time()
        
        total = sum(c[2].value for c in chunks)
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
        bw_e2e = total / 1024 / 1024 / total_time if total_time > 0 else 0

        stats = {
            "prepare_time": t_rebuild_end - t_rebuild,
            "read_time": pure_read_time,
            "set_data_time": t_end - t_update,
            "total_time": total_time,
            "bw_pure": total / 1024 / 1024 / pure_read_time if pure_read_time > 0 else 0,
            "bw_e2e": bw_e2e
        }
        return total, len(chunks), total_time, bw_e2e, stats