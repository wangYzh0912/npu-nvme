"""Chunk-building helpers for NPU↔NVMe DMA transfers.

Functions for splitting parameters along chunk_size boundaries with
4 KiB alignment, and for rebuilding chunk lists from saved metadata.
"""

import ctypes
import math
from typing import Dict, List

import numpy as np
import mindspore as ms

# -- Chunk builder for device parameters --
def build_chunks(params: List[Dict], chunk_size: int):
    """Build DMA chunk list from parameter descriptors.

    Each parameter descriptor dict must have keys: ptr, size, offset, name.
    Returns (chunks, total_size) where each chunk is
    (ptr: c_void_p, offset: c_uint64, size: c_size_t, name: str).
    """
    chunks = []
    total_size = 0
    for p in params:
        ptr = p["ptr"]
        remaining = p["size"]
        inner_off = 0
        nvme_offset_bytes = p["offset"]
        name = p.get("name", "unknown")

        while remaining > 0:
            take = min(remaining, chunk_size)
            chunks.append((
                ctypes.c_void_p(ptr + inner_off),
                ctypes.c_uint64(nvme_offset_bytes),
                ctypes.c_size_t(take),
                name
            ))
            remaining -= take
            inner_off += take
            nvme_offset_bytes += int(math.ceil(take / 4096.0)) * 4096
            total_size += take

    return chunks, total_size


def build_chunks_host(ptr_base: int, start_offset: int, total_size: int,
                      chunk_size: int, name: str = ""):
    """Build chunk list for a single host-side buffer (no MindSpore param).

    Wraps a raw pointer + offset + size into the params dict format expected
    by build_chunks().  Returns (chunks_list, total_size) — same shape as
    build_chunks().

    Args:
        ptr_base:     base Python integer address of the host buffer
        start_offset: NVMe byte offset where writes begin
        total_size:   total bytes to write
        chunk_size:   max bytes per chunk (4K-aligned in build_chunks)
        name:         label for debugging (default "")
    Returns:
        (list of (ptr: c_void_p, offset: c_uint64, size: c_size_t, name: str),
         total_size: int)
    """
    params = [{"ptr": ptr_base, "size": total_size, "offset": start_offset, "name": name}]
    return build_chunks(params, chunk_size)


def build_ctypes_arrays(chunks):
    """Convert a chunk list to the three ctypes arrays expected by C APIs.

    Args:
        chunks: list of (ptr, offset, size, name) tuples from build_chunks()
    Returns:
        (c_ptrs, c_offs, c_sizes) — c_void_p*, c_uint64*, c_size_t* arrays
    """
    n = len(chunks)
    ptrs = (ctypes.c_void_p * n)()
    offs = (ctypes.c_uint64 * n)()
    szs = (ctypes.c_size_t * n)()
    for i, (p, o, s, _) in enumerate(chunks):
        ptrs[i] = p
        offs[i] = ctypes.c_uint64(o.value)
        szs[i] = s
    return ptrs, offs, szs


def rebuild_chunks_from_meta(models, params_meta: Dict, chunk_size: int):
    """Rebuild device + host chunk lists from checkpoint metadata for load().

    Args:
        models:      a model or list of models with parameters_and_names()
        params_meta: dict of {param_name: {offset, size, shape, dtype}}
        chunk_size:  max bytes per DMA chunk
    Returns:
        (dev_chunks, host_chunks, buffers) — two chunk lists + detail list
    """
    from direct_checkpoint import get_dev_ptr  # circular-import-safe

    if not isinstance(models, (list, tuple)):
        models = [models]
    buffers = []

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
