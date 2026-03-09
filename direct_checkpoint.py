import ctypes
import math
import os
import pickle
import time
from typing import List, Dict

import mindspore as ms
from mindspore import ops
import numpy as np


# ============================================================
# 绑定 C 接口
# ============================================================
_LIB_PATH = os.path.join(os.path.dirname(__file__), "out", "lib", "libnpu_nvme.so")
lib = ctypes.CDLL(_LIB_PATH)

class NPUNVMEContext(ctypes.Structure):
    pass

# init(ctx**, addr, pipeline_depth, requested_chunk_size)
lib.npu_nvme_init.argtypes = [
    ctypes.POINTER(ctypes.POINTER(NPUNVMEContext)),
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.c_char_p,
]

# (Removed unused helper bindings)
lib.npu_nvme_init.restype = ctypes.c_int

# cleanup
lib.npu_nvme_cleanup.argtypes = [ctypes.POINTER(NPUNVMEContext)]
lib.npu_nvme_cleanup.restype = None

# get_max_transfer
lib.npu_nvme_get_max_transfer.argtypes = [ctypes.POINTER(NPUNVMEContext)]
lib.npu_nvme_get_max_transfer.restype = ctypes.c_int

# write_batch / read_batch
lib.npu_nvme_write_batch.argtypes = [
    ctypes.POINTER(NPUNVMEContext),
    ctypes.POINTER(ctypes.c_void_p),   # void** npu_ptrs
    ctypes.POINTER(ctypes.c_uint64),   # uint64_t* offsets
    ctypes.POINTER(ctypes.c_size_t),   # size_t* sizes
    ctypes.c_int                       # num_items
]
lib.npu_nvme_write_batch.restype = ctypes.c_int

lib.npu_nvme_read_batch.argtypes = [
    ctypes.POINTER(NPUNVMEContext),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_int
]
lib.npu_nvme_read_batch.restype = ctypes.c_int


# ============================================================
# 工具：分块与合包
# ============================================================
def build_chunks(params: List[Dict], chunk_size: int):
    """
    将一组参数（含 ptr/size/offset_on_nvme）切成 <= chunk_size 的块。
    返回 (chunks, total_size)
    chunks: List[ (ptr, nvme_offset, size) ]
    """
    chunks = []
    nvme_offset = 0
    for p in params:
        ptr = p["ptr"]
        remaining = p["size"]
        inner_off = 0
        while remaining > 0:
            take = min(remaining, chunk_size)
            chunks.append((
                ctypes.c_void_p(ptr + inner_off),   # NPU 地址 + 偏移
                ctypes.c_uint64(nvme_offset),       # NVMe 偏移（无空洞）
                ctypes.c_size_t(take)                # 本块大小
            ))
            remaining -= take
            inner_off += take
            nvme_offset += int(math.ceil(take / 4096.0) * 4096)  # NVMe 按 4K 对齐推进偏移
    return chunks, nvme_offset


def rebuild_chunks_from_meta(models, meta: Dict, chunk_size: int):
    """
    根据元数据和 chunk_size，重建读取所需的块列表，并为每个参数分配缓冲区（优先 NPU，失败则 CPU）。
    支持单个 ms.nn.Cell 或 [model, optimizer] 列表。
    返回 (chunks, buffers)。
    """
    if not isinstance(models, (list, tuple)):
        models = [models]

    buffers = []
    
    # 辅助：获取设备指针
    def get_dev_ptr(p):
        ptr = 0
        try:
            data_obj = p.data if hasattr(p, "data") else p
            
            # [CRITICAL UPDATE]
            # Based on user's environment:
            # 1. 'device_address' attribute DOES NOT EXIST on the Tensor object directly.
            # 2. '_data_ptr()' works and returns a valid integer address.
            
            # Therefore, relying on 'device_address' as a gatekeeper causes 100% false negatives (zero_copy=0).
            # We must try calling '_data_ptr()' directly.
            
            if hasattr(data_obj, "_data_ptr"):
                 # To be safe for uninitialized params (though load usually implies existing structure)
                 if isinstance(p, ms.Parameter) and hasattr(p, "is_inited") and not p.is_inited:
                     ptr = 0
                 else:
                     ptr = int(data_obj._data_ptr())
        except Exception:
            # If _data_ptr() fails (e.g. Tensor not on device), we catch it here and fall back to 0
            ptr = 0
        return ptr

    for model in models:
        # Skip if None or no parameters_and_names (e.g. empty list entry)
        if model is None or not hasattr(model, "parameters_and_names"):
            continue

        for name, param in model.parameters_and_names():
            if name not in meta:
                continue
            info = meta[name]
            
            # 尝试获取设备指针
            dev_ptr = get_dev_ptr(param)
            use_dev = (dev_ptr != 0)
            
            np_arr = None
            ptr_val = dev_ptr
            
            # 若无法获取设备指针，则回退 CPU
            if not use_dev:
                np_arr = np.empty(info["shape"], dtype=np.dtype(info["dtype"]))
                ptr_val = np_arr.ctypes.data
                
            buffers.append({
                "name": name,
                "ptr": ptr_val,
                "size": info["size"],
                "offset": info["offset"],
                "np_arr": np_arr,      # None if zero-copy
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
        spdk_shm_id: int = 1,                # shared mem id for SPDK multiprocess
    ):
        self.ctx = ctypes.POINTER(NPUNVMEContext)()
        self.enable_profiling = enable_profiling
        self.profiling_dir = profiling_dir
        self.rank_id = rank_id
        self.world_size = world_size
        self.base_offset_bytes = base_offset_bytes
        self.shard_span_bytes = shard_span_bytes
        # set env for SPDK multiprocess (primary/secondary auto-decided by SPDK via shm_id)
        os.environ.setdefault("SPDK_SHM_ID", str(spdk_shm_id))

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

        # 生效的 chunk_size：完全采用用户请求值，设备返回值仅作参考打印
        eff = lib.npu_nvme_get_max_transfer(self.ctx)
        self.chunk_size = requested_chunk_size
        print(
            f"[DirectCheckpoint] init ok. "
            f"pipeline_depth={pipeline_depth}, "
            f"requested_chunk={requested_chunk_size/1024/1024:.2f}MB, "
            f"effective_chunk={self.chunk_size/1024/1024:.2f}MB (raw_device={eff/1024/1024:.2f}MB) "
            f"rank={self.rank_id}/{self.world_size}, base_offset={self.base_offset_bytes}, shm_id={os.environ.get('SPDK_SHM_ID')}"
        )
        self.meta = {}
        self.total_size = 0

    def cleanup(self):
        if self.ctx:
            lib.npu_nvme_cleanup(self.ctx)
            self.ctx = None

    def _prepare_params(self, models):
        # Support single model or list of models
        if not isinstance(models, (list, tuple)):
            models = [models]
            
        params = []
        for model in models:
            if model is None: 
                continue
            # Some objects (like Optimizer) might not have parameters_and_names directly exposed the same way
            # But MindSpore Optimizers inherit from Cell, so they usually do.
            if not hasattr(model, "parameters_and_names"):
                continue

            for name, p in model.parameters_and_names():
                # 优先使用实验性设备指针实现零拷贝；若获取失败则回退到 CPU 拷贝
                ptr = 0
                try:
                    data_obj = p.data if hasattr(p, "data") else p
                    
                    # [CRITICAL UPDATE]
                    # Based on user's test: 'device_address' attribute is MISSING on Tensors.
                    # However, '_data_ptr()' returns a valid address.
                    # So we must call '_data_ptr()' directly.
                    
                    # 1. 尝试 _data_ptr (唯一有效的途径)
                    if hasattr(data_obj, "_data_ptr"):
                        if isinstance(p, ms.Parameter) and hasattr(p, "is_inited") and not p.is_inited:
                            ptr = 0
                        else:
                            ptr = int(data_obj._data_ptr())

                    # 2. 备选：如果未来版本有了 device_address
                    if ptr == 0 and hasattr(data_obj, "device_address"):
                        dev_addr = getattr(data_obj, "device_address", None)
                        if dev_addr is not None and hasattr(dev_addr, "ptr"):
                            ptr = int(dev_addr.ptr)

                except Exception:
                    ptr = 0  # 某些张量尚未物化到 device，回退到 CPU 拷贝
                
                dtype_np = np.dtype(ms.dtype_to_nptype(p.dtype))
                # Skip empty parameters
                if np.prod(p.shape) == 0:
                    continue

                size = int(np.prod(p.shape)) * dtype_np.itemsize

                shape = list(p.shape)
                dtype = dtype_np.name

                if ptr == 0:
                    np_arr = p.asnumpy()
                    ptr = np_arr.ctypes.data
                    params.append({
                        "name": name,
                        "ptr": ptr,
                        "size": size,
                        "shape": shape,
                        "dtype": dtype,
                        "np_arr": np_arr,
                        "param_ref": p
                    })
                else:
                    params.append({
                        "name": name,
                        "ptr": ptr,
                        "size": size,
                        "shape": shape,
                        "dtype": dtype,
                        "np_arr": None,
                        "param_ref": p
                    })
        return params

    def save(self, model: ms.nn.Cell, meta_path: str = "checkpoint_meta.pkl"):
        t_start = time.time()
        if self.chunk_size <= 0:
            raise RuntimeError(f"invalid chunk_size={self.chunk_size}, please check npu_nvme_get_max_transfer or requested_chunk_size")
        params = self._prepare_params(model)
        t_prep = time.time()
        zero_copy = sum(1 for p in params if p["np_arr"] is None)
        cpu_copy = len(params) - zero_copy
        raw_total = sum(p["size"] for p in params)
        max_item = max(params, key=lambda x: x["size"])
        print(
            f"[Save] params={len(params)} zero_copy={zero_copy} cpu_copy={cpu_copy} "
            f"chunk_size_bytes={self.chunk_size} ({self.chunk_size/1024/1024:.2f}MB) "
            f"raw_total={raw_total/1024/1024:.2f}MB max_param={max_item['name']} {max_item['size']/1024/1024:.2f}MB",
            flush=True,
        )
        if raw_total > 32 * 1024 * 1024 * 1024:  # >32GB suspicious for gpt2-xl
            print(f"[Save][WARN] raw_total seems too large: {raw_total/1024/1024/1024:.2f}GB", flush=True)
        if self.chunk_size < 1024 * 1024:
            print(f"[Save][WARN] chunk_size <1MB ({self.chunk_size} bytes), expect ~4MB; check C library rebuild", flush=True)
        if self.shard_span_bytes is not None and (self.base_offset_bytes + raw_total) > (self.base_offset_bytes + self.shard_span_bytes):
            raise RuntimeError(
                f"shard_span_bytes too small: need {raw_total}, span {self.shard_span_bytes}, base_offset {self.base_offset_bytes}"
            )
        # 输出参数信息到params.csv，便于调试
        if self.enable_profiling:
            p_out = os.path.join(self.profiling_dir, "params.csv")
            with open(p_out, "w") as f:
                f.write("name,ptr,size,shape,dtype\n")
                for p in params:
                    f.write(f"{p['name']},{p['ptr']},{p['size']},\"{p['shape']}\",{p['dtype']}\n")  
        nvme_offset = 0
        layout = []
        for p in params:
            layout.append({
                **p,
                "offset": nvme_offset
            })
            nvme_offset += int(math.ceil(p["size"] / 4096.0) * 4096)

        # 生成 chunk 列表
        chunks, total = build_chunks(layout, self.chunk_size)
        self.total_size = total
        print(
            f"[Save] chunks={len(chunks)} total={total/1024/1024:.2f}MB "
            f"prep_time={t_prep - t_start:.3f}s",
            flush=True,
        )

        # 准备 ctypes 数组
        num = len(chunks)
        c_ptrs = (ctypes.c_void_p * num)()
        c_offs = (ctypes.c_uint64 * num)()
        c_sizes = (ctypes.c_size_t * num)()
        for i, (p, o, s) in enumerate(chunks):
            c_ptrs[i] = p
            c_offs[i] = ctypes.c_uint64(o.value + self.base_offset_bytes)
            c_sizes[i] = s

        t0 = time.time()
        rc = lib.npu_nvme_write_batch(self.ctx, c_ptrs, c_offs, c_sizes, num)
        if rc != 0:
            raise RuntimeError("write_batch failed")
        t1 = time.time()
        
        # [Corrected] BW should be based on total_elapsed (including D2H copy in _prepare_params)
        # to reflect the real impact on training throughput.
        # Original: bw = total / 1024 / 1024 / (t1 - t0)
        real_time = t1 - t_start
        bw = total / 1024 / 1024 / real_time
        
        print(
            f"[Save] memcpy+write {t1-t0:.3f}s, BW(pure_write)={total/1024/1024/(t1-t0):.1f} MB/s, "
            f"BW(e2e)={bw:.1f} MB/s, total_elapsed={real_time:.3f}s",
            flush=True,
        )

        # 保存元数据
        meta = {
            "chunk_size": self.chunk_size,
            "total_size": total,
            "rank_id": self.rank_id,
            "world_size": self.world_size,
            "base_offset_bytes": self.base_offset_bytes,
            "shard_span_bytes": self.shard_span_bytes,
            "params": {p["name"]: {
                "offset": p["offset"],
                "size": p["size"],
                "shape": p["shape"],
                "dtype": p["dtype"],
            } for p in layout}
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        self.meta = meta
        print(f"[Save] meta saved to {meta_path}")
        
        stats = {
            "prep_time": t_prep - t_start,
            "write_time": t1 - t0,
            "total_time": real_time,
            "bw_pure": total/1024/1024/(t1-t0) if (t1-t0) > 0 else 0,
            "bw_e2e": bw
        }
        return total, len(chunks), real_time, bw, stats

    def load(self, model: ms.nn.Cell, meta_path: str = "checkpoint_meta.pkl"):
        t_start = time.time()
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        chunk_size = min(meta.get("chunk_size", self.chunk_size), self.chunk_size)
        self.meta = meta
        base_off = meta.get("base_offset_bytes", self.base_offset_bytes)
        if self.base_offset_bytes != base_off:
            print(f"[Load][WARN] base_offset mismatch: meta {base_off}, current {self.base_offset_bytes}; use meta", flush=True)
            self.base_offset_bytes = base_off

        # create empty buffers
        t_rebuild = time.time()
        chunks, buffers = rebuild_chunks_from_meta(model, meta["params"], chunk_size)
        t_rebuild_end = time.time()
        
        num = len(chunks)
        c_ptrs = (ctypes.c_void_p * num)()
        c_offs = (ctypes.c_uint64 * num)()
        c_sizes = (ctypes.c_size_t * num)()
        for i, (p, o, s) in enumerate(chunks):
            c_ptrs[i] = p
            c_offs[i] = ctypes.c_uint64(o.value + self.base_offset_bytes)
            c_sizes[i] = s

        # read from NVMe to buffers
        t0 = time.time()
        rc = lib.npu_nvme_read_batch(self.ctx, c_ptrs, c_offs, c_sizes, num)
        if rc != 0:
            raise RuntimeError("read_batch failed")
        t1 = time.time()
        
        # [Fix] Calc bandwidth using actually loaded size (support partial load)
        # total = meta["total_size"]
        total = sum(c[2].value for c in chunks)
        
        # update model weights
        t_update = time.time()
        
        for buf in buffers:
            # If zero-copy load (use_dev=True), data already in place via read_batch -> aclrtMemcpy(H2D)
            if buf.get("use_dev", False):
                continue

            param = buf["param_ref"]
            if buf["np_arr"] is not None:
                # Fallback: copy from numpy buffer to parameter
                # Note: read_batch likely copied to the numpy buffer using H2D (check implicit behavior)
                tensor = ms.Tensor(buf["np_arr"], dtype=param.dtype)
                ops.assign(param, tensor)
        t_end = time.time()
        
        # Determine actual zero-copy ratio
        zc_count = sum(1 for b in buffers if b.get("use_dev", False))
        print(f"[Load] Zero-copy applied to {zc_count}/{len(buffers)} params", flush=True)

        pure_read_time = t1 - t0
        total_time = t_end - t_start
        bw_read = total / 1024 / 1024 / pure_read_time if pure_read_time > 0 else 0
        bw_e2e = total / 1024 / 1024 / total_time if total_time > 0 else 0

        print(
            f"[Load] prepare={t_rebuild_end - t_rebuild:.3f}s read={pure_read_time:.3f}s set_data={t_end - t_update:.3f}s "
            f"BW(pure)={bw_read:.1f} MB/s BW(e2e)={bw_e2e:.1f} MB/s total={total_time:.3f}s",
            flush=True
        )
        stats = {
            "prepare_time": t_rebuild_end - t_rebuild,
            "read_time": pure_read_time,
            "set_data_time": t_end - t_update,
            "total_time": total_time,
            "bw_pure": bw_read,
            "bw_e2e": bw_e2e
        }
        return total, len(chunks), total_time, bw_e2e, stats
