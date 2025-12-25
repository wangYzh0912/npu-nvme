import ctypes
import math
import time
from typing import List, Dict

import torch


# ============================================================
# 绑定 C 接口
# ============================================================
lib = ctypes.CDLL("./out/lib/libnpu_nvme.so")

class NPUNVMEContext(ctypes.Structure):
    pass

# init(ctx**, addr, pipeline_depth, requested_chunk_size)
lib.npu_nvme_init.argtypes = [
    ctypes.POINTER(ctypes.POINTER(NPUNVMEContext)),
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_size_t,
]
lib.npu_nvme_init.restype = ctypes.c_int

# cleanup
lib.npu_nvme_cleanup.argtypes = [ctypes.POINTER(NPUNVMEContext)]
lib.npu_nvme_cleanup.restype = None

# get_max_transfer
lib.npu_nvme_get_max_transfer.argtypes = [ctypes.POINTER(NPUNVMEContext)]
lib.npu_nvme_get_max_transfer.restype = ctypes.c_size_t

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


def rebuild_chunks_from_meta(model: torch.nn.Module, meta: Dict, chunk_size: int):
    """
    根据元数据和 chunk_size，按与写入时一致的规则重建块列表。
    """
    params = []
    for name, param in model.named_parameters():
        if name not in meta:
            continue
        info = meta[name]
        params.append({
            "ptr": param.data_ptr(),
            "size": info["size"],
            "offset": info["offset"]
        })
    # 按 offset 排序，确保顺序一致
    params.sort(key=lambda x: x["offset"])
    # 重建块：沿用当初的连续布局
    chunks = []
    for p in params:
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
    return chunks


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
    ):
        self.ctx = ctypes.POINTER(NPUNVMEContext)()
        self.enable_profiling = enable_profiling

        rc = lib.npu_nvme_init(
            ctypes.byref(self.ctx),
            nvme_addr.encode(),
            npu_device_id,
            pipeline_depth,
            requested_chunk_size,
            enable_profiling
        )
        if rc != 0:
            raise RuntimeError("npu_nvme_init failed")

        # 生效的 chunk_size（已被设备上限裁剪）
        self.chunk_size = lib.npu_nvme_get_max_transfer(self.ctx)
        print(f"[DirectCheckpoint] init ok. "
              f"pipeline_depth={pipeline_depth}, "
              f"requested_chunk={requested_chunk_size/1024/1024:.2f}MB, "
              f"effective_chunk={self.chunk_size/1024/1024:.2f}MB")
        self.chunk_size = requested_chunk_size
        self.meta = {}
        self.total_size = 0

    def cleanup(self):
        if self.ctx:
            lib.npu_nvme_cleanup(self.ctx)
            self.ctx = None

    def _prepare_params(self, model: torch.nn.Module):
        params = []
        for name, p in model.named_parameters():
            ######
            #这里需要添加判断tensor是否在NPU上
            ######
            ptr = p.data_ptr()
            size = p.numel() * p.element_size()
            params.append({
                "name": name,
                "ptr": ptr,
                "size": size,
                "shape": list(p.shape),
                "dtype": str(p.dtype),
            })
        return params

    def save(self, model: torch.nn.Module, meta_path: str = "checkpoint_meta.pt"):
        params = self._prepare_params(model)
        # 输出参数信息到params.csv，便于调试
        if self.enable_profiling:
            with open("params.csv", "w") as f:
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
        print(f"[Save] params={len(params)}, chunks={len(chunks)}, "
              f"total={total/1024/1024:.2f}MB, chunk_size={self.chunk_size/1024/1024:.2f}MB")

        # 准备 ctypes 数组
        num = len(chunks)
        c_ptrs = (ctypes.c_void_p * num)()
        c_offs = (ctypes.c_uint64 * num)()
        c_sizes = (ctypes.c_size_t * num)()
        for i, (p, o, s) in enumerate(chunks):
            c_ptrs[i] = p
            c_offs[i] = o
            c_sizes[i] = s

        t0 = time.time()
        rc = lib.npu_nvme_write_batch(self.ctx, c_ptrs, c_offs, c_sizes, num)
        if rc != 0:
            raise RuntimeError("write_batch failed")
        t1 = time.time()
        bw = total / 1024 / 1024 / (t1 - t0)
        print(f"[Save] done in {t1-t0:.3f}s, BW={bw:.1f} MB/s")

        # 保存元数据
        meta = {
            "chunk_size": self.chunk_size,
            "total_size": total,
            "params": {p["name"]: {
                "offset": p["offset"],
                "size": p["size"],
                "shape": p["shape"],
                "dtype": p["dtype"],
            } for p in layout}
        }
        torch.save(meta, meta_path)
        self.meta = meta
        print(f"[Save] meta saved to {meta_path}")
        return total, len(chunks), t1 - t0, bw

    def load(self, model: torch.nn.Module, meta_path: str = "checkpoint_meta.pt"):
        meta = torch.load(meta_path)
        chunk_size = min(meta.get("chunk_size", self.chunk_size), self.chunk_size)
        self.meta = meta

        chunks = rebuild_chunks_from_meta(model, meta["params"], chunk_size)
        num = len(chunks)
        c_ptrs = (ctypes.c_void_p * num)()
        c_offs = (ctypes.c_uint64 * num)()
        c_sizes = (ctypes.c_size_t * num)()
        for i, (p, o, s) in enumerate(chunks):
            c_ptrs[i] = p
            c_offs[i] = o
            c_sizes[i] = s

        t0 = time.time()
        rc = lib.npu_nvme_read_batch(self.ctx, c_ptrs, c_offs, c_sizes, num)
        if rc != 0:
            raise RuntimeError("read_batch failed")
        t1 = time.time()
        total = meta["total_size"]
        bw = total / 1024 / 1024 / (t1 - t0)
        print(f"[Load] done in {t1-t0:.3f}s, BW={bw:.1f} MB/s")
        return total, len(chunks), t1 - t0, bw
