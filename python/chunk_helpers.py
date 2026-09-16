"""Chunk-building helpers for NPU↔NVMe DMA transfers.

Functions for splitting parameters along chunk_size boundaries with
4 KiB alignment, and for rebuilding chunk lists from saved metadata.
"""

import ctypes
import math
from typing import Dict, List

import numpy as np
from numbers import Integral

MAX_BATCH_ITEMS = 65536
MAX_BATCH_BYTES = 64 * 1024**3
MAX_DESCRIPTOR_BYTES = 16 * 1024**2
MAX_POINTER = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1


def _integer(value, name, *, minimum=0, maximum=(1 << 64) - 1):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} outside supported bounds")
    return value


def _chunk_size(value):
    value = _integer(value, "chunk_size", minimum=1, maximum=(1 << 32) - 1)
    if value % 4096:
        raise ValueError("chunk_size must be 4 KiB aligned")
    return value


def validate_descriptors(params, chunk_size):
    """Validate source descriptors before allocating snapshots or ctypes arrays."""
    chunk_size = _chunk_size(chunk_size)
    if len(params) > MAX_BATCH_ITEMS:
        raise ValueError("too many parameter descriptors")
    checked = []
    count = total = descriptor_bytes = 0
    for p in params:
        name = p.get("name", "unknown")
        if not isinstance(name, str) or len(name) > 1024:
            raise ValueError("invalid descriptor name")
        descriptor_bytes += len(name.encode("utf-8")) + 64
        if descriptor_bytes > MAX_DESCRIPTOR_BYTES:
            raise ValueError("descriptor byte budget exceeded")
        ptr = _integer(p["ptr"], "ptr", minimum=1, maximum=MAX_POINTER)
        size = _integer(p["size"], "size", minimum=1, maximum=MAX_BATCH_BYTES)
        offset = _integer(p["offset"], "offset")
        if offset % 4096 or ptr > MAX_POINTER - (size - 1):
            raise ValueError("unaligned offset or overflowing pointer")
        if offset > (1 << 64) - 1 - ((size + 4095) // 4096 * 4096):
            raise ValueError("overflowing extent")
        count += (size + chunk_size - 1) // chunk_size
        total += ((size + 4095) // 4096) * 4096
        if count > MAX_BATCH_ITEMS or total > MAX_BATCH_BYTES:
            raise ValueError("batch exceeds item or byte budget")
        checked.append(dict(p, ptr=ptr, size=size, offset=offset))
    return checked


# -- Chunk builder for device parameters --
def build_chunks(params: List[Dict], chunk_size: int):
    """Build DMA chunk list from parameter descriptors.

    Each parameter descriptor dict must have keys: ptr, size, offset, name.
    Returns (chunks, total_size) where each chunk is
    (ptr: c_void_p, offset: c_uint64, size: c_size_t, name: str).
    """
    chunk_size = _chunk_size(chunk_size)
    checked = validate_descriptors(params, chunk_size)
    chunks = []
    total_size = 0
    for p in checked:
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
            nvme_offset_bytes += ((take + 4095) // 4096) * 4096
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
    if n > MAX_BATCH_ITEMS:
        raise ValueError("batch exceeds item budget")
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

    _chunk_size(chunk_size)
    if len(params_meta) > MAX_BATCH_ITEMS:
        raise ValueError("too many metadata descriptors")
    if not isinstance(models, (list, tuple)):
        models = [models]
    planned = []
    total = count = 0
    for model in models:
        if model is None or not hasattr(model, "parameters_and_names"):
            continue
        for name, param in model.parameters_and_names():
            if name not in params_meta:
                continue
            info = params_meta[name]
            if len(info["shape"]) > 32:
                raise ValueError("tensor rank exceeds limit")
            shape = tuple(_integer(x, "shape", minimum=1) for x in info["shape"])
            if not isinstance(info["dtype"], str) or len(info["dtype"]) > 64:
                raise ValueError("invalid dtype descriptor")
            try:
                dtype = np.dtype(info["dtype"])
            except TypeError as error:
                raise ValueError("invalid dtype descriptor") from error
            if dtype.hasobject or dtype.fields or dtype.kind not in 'biufc':
                raise ValueError("unsupported tensor dtype")
            size = _integer(info["size"], "size", minimum=1, maximum=MAX_BATCH_BYTES)
            if math.prod(shape) * dtype.itemsize != size:
                raise ValueError("shape/dtype does not match DMA size")
            actual_shape = tuple(getattr(param, "sliced_shape", None) or param.shape)
            if hasattr(param, "data") and hasattr(param.data, "shape"):
                if math.prod(param.data.shape) < math.prod(actual_shape):
                    actual_shape = tuple(param.data.shape)
            if shape != actual_shape:
                raise ValueError("checkpoint shape does not match target allocation")
            if hasattr(param, "dtype"):
                import mindspore as ms
                if np.dtype(ms.dtype_to_nptype(param.dtype)) != dtype:
                    raise ValueError("checkpoint dtype does not match target allocation")
            offset = _integer(info["offset"], "offset")
            if offset % 4096 or offset > (1 << 64) - 1 - ((size + 4095) // 4096 * 4096):
                raise ValueError("invalid checkpoint extent")
            total += ((size + 4095) // 4096) * 4096
            count += (size + chunk_size - 1) // chunk_size
            if total > MAX_BATCH_BYTES or count > MAX_BATCH_ITEMS:
                raise ValueError("restore exceeds item or byte budget")
            planned.append((name, param, shape, dtype, size, offset))
    # Validate every descriptor before allocation or device pointer lookup.
    buffers = []
    for name, param, shape, dtype, size, offset in planned:
        dev_ptr = get_dev_ptr(param)
        np_arr = None if dev_ptr else np.empty(shape, dtype=dtype)
        buffers.append({"name": name, "ptr": dev_ptr or np_arr.ctypes.data,
                        "size": size, "offset": offset, "np_arr": np_arr,
                        "param_ref": param, "use_dev": bool(dev_ptr)})

    buffers.sort(key=lambda x: x["offset"])

    dev_buffers = [b for b in buffers if b["use_dev"]]
    host_buffers = [b for b in buffers if not b["use_dev"]]

    dev_chunks, _ = build_chunks(dev_buffers, chunk_size)
    host_chunks, _ = build_chunks(host_buffers, chunk_size)

    return dev_chunks, host_chunks, buffers
