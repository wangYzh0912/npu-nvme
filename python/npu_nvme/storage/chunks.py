"""Chunk-building helpers for NPU↔NVMe DMA transfers.

Functions for splitting parameters along chunk_size boundaries with
4 KiB alignment, and for rebuilding chunk lists from saved metadata.
"""

import ctypes
from typing import Dict, List

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


def iter_chunk_windows(params, chunk_size, *, max_items=MAX_BATCH_ITEMS,
                       max_bytes=MAX_BATCH_BYTES, checkpoint_bytes=None,
                       descriptor_bytes=MAX_DESCRIPTOR_BYTES):
    """Stream a checkpoint as independently admitted windows.

    Pointer/extent validation covers each tensor. A caller-provided checkpoint
    budget is separate from native per-request item and byte limits. At most
    one request window is materialized; no aggregate chunk list is allocated.
    """
    chunk_size=_chunk_size(chunk_size)
    max_items=_integer(max_items,'max_items',minimum=1,maximum=MAX_BATCH_ITEMS)
    max_bytes=_integer(max_bytes,'max_bytes',minimum=4096,maximum=MAX_BATCH_BYTES)
    descriptor_bytes=_integer(descriptor_bytes,'descriptor_bytes',minimum=1)
    if chunk_size>max_bytes:raise ValueError('chunk exceeds request byte budget')
    if checkpoint_bytes is not None:checkpoint_bytes=_integer(checkpoint_bytes,'checkpoint_bytes',minimum=1)
    window=[];window_bytes=logical_bytes=description_bytes=0
    for p in params:
        name=p.get('name','unknown')
        if not isinstance(name,str) or len(name)>1024:raise ValueError('invalid descriptor name')
        description_bytes+=len(name.encode('utf-8'))+64
        if description_bytes>descriptor_bytes:raise ValueError('descriptor byte budget exceeded')
        pointer=_integer(p['ptr'],'ptr',minimum=1,maximum=MAX_POINTER)
        size=_integer(p['size'],'size',minimum=1)
        offset=_integer(p['offset'],'offset')
        if offset%4096 or pointer>MAX_POINTER-(size-1):raise ValueError('unaligned offset or overflowing pointer')
        if offset>(1<<64)-1-((size+4095)//4096*4096):raise ValueError('overflowing extent')
        logical_bytes+=size
        if checkpoint_bytes is not None and logical_bytes>checkpoint_bytes:raise ValueError('checkpoint byte budget exceeded')
        cursor=0
        while cursor<size:
            take=min(chunk_size,size-cursor);padded=(take+4095)//4096*4096
            if window and (len(window)>=max_items or window_bytes+padded>max_bytes):
                yield window;window=[];window_bytes=0
            window.append((ctypes.c_void_p(pointer+cursor),ctypes.c_uint64(offset+cursor),ctypes.c_size_t(take),name))
            window_bytes+=padded;cursor+=take
    if window:yield window
