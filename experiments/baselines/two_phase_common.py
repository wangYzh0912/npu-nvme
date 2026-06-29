"""Shared two-phase checkpoint utilities for CheckFreq & PCcheck baselines.

Provides parameter descriptor extraction, D2H/H2D/D2D snapshot via aclrtMemcpy,
kernel FS persist/restore, and contiguous HBM buffer construction.

ACL memcpy kind constants (Ascend aclrtMemcpyKind enum):
  1 = ACL_MEMCPY_HOST_TO_DEVICE   (H2D)
  2 = ACL_MEMCPY_DEVICE_TO_HOST   (D2H)
  3 = ACL_MEMCPY_DEVICE_TO_DEVICE (D2D)
"""

import os
import sys
import time
import ctypes

import numpy as np

import mindspore as ms
from mindspore import Tensor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))

from c_bindings import acl_lib
from direct_checkpoint import get_dev_ptr

# -- ACL memcpy kind constants ------------------------------------------------
ACL_MEMCPY_HOST_TO_DEVICE   = 1
ACL_MEMCPY_DEVICE_TO_HOST   = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


# ---------------------------------------------------------------------------
# 1. Parameter introspection
# ---------------------------------------------------------------------------

def get_param_descriptors(model: ms.nn.Cell) -> list:
    """Extract device pointer metadata for every trainable parameter.

    Must be called AFTER ``warmup_model()`` — otherwise get_dev_ptr()
    returns 0 for parameters whose HBM hasn't been allocated yet.

    Args:
        model: MindSpore model with warm-allocated parameters.

    Returns:
        list of dicts, each with keys:
          name         – str, e.g. "backbone.blocks.0.attention.dense1.weight"
          ptr          – int, NPU device pointer (0 = not allocated → skipped)
          size         – int, bytes (= numel * itemsize)
          dtype_ms     – ms.dtype
          dtype_np     – np.dtype
          dtype_np_str – str, for round-trip deserialisation ('<f2', '<f4', …)
          shape        – list[int], parameter shape
          param_ref    – ms.Parameter, original reference (needed for restore)
    """
    descs = []
    n_skipped = 0

    for p in model.trainable_params():
        ptr = get_dev_ptr(p)
        if ptr == 0:
            n_skipped += 1
            continue

        dtype_np = np.dtype(ms.dtype_to_nptype(p.dtype))
        size = int(p.size) * dtype_np.itemsize
        b = dtype_np.byteorder
        if b == '=' or b == '|':
            endian = '<' if sys.byteorder == 'little' else '>'
        else:
            endian = b

        descs.append({
            "name":         p.name,
            "ptr":          ptr,
            "size":         size,
            "dtype_ms":     p.dtype,
            "dtype_np":     dtype_np,
            "dtype_np_str": f"{endian}{dtype_np.char}{dtype_np.itemsize}",
            "shape":        list(p.shape),
            "param_ref":    p,
        })

    if n_skipped:
        print(f"[two_phase_common] WARNING: {n_skipped} params with ptr==0 skipped "
              f"(did you call warmup_model()?)")

    return descs


def get_total_param_bytes(model: ms.nn.Cell) -> int:
    """Sum raw byte count across all trainable parameters."""
    total = 0
    for p in model.trainable_params():
        itemsize = np.dtype(ms.dtype_to_nptype(p.dtype)).itemsize
        total += int(p.size) * itemsize
    return total


# ---------------------------------------------------------------------------
# 2. Host buffer allocation
# ---------------------------------------------------------------------------

def allocate_host_buffer(total_bytes: int) -> np.ndarray:
    """Allocate a zero-filled byte buffer in host DRAM.

    Uses ``np.empty`` (no zero-init) because the buffer is immediately
    overwritten via aclrtMemcpy D2H.

    For best D2H performance, use ``allocate_pinned_host_buffer`` instead.

    Args:
        total_bytes: size in bytes.

    Returns:
        np.ndarray of shape (total_bytes,), dtype uint8.
    """
    return np.empty(total_bytes, dtype=np.uint8)


def allocate_pinned_host_buffer(total_bytes: int) -> int:
    """Allocate page-locked (pinned) host memory via aclrtMallocHost.

    Pinned memory allows the DMA engine to access host memory directly
    without the per-call page-pinning overhead of regular malloc'd buffers.
    On Ascend 910B this gives ~6× faster D2H bandwidth (~17.5 GB/s vs
    ~3.0 GB/s).

    Must call ``free_pinned_host_buffer`` to release.

    Args:
        total_bytes: size in bytes.

    Returns:
        Device-accessible host pointer (int).  Use with ctypes.c_void_p(ptr).
    """
    if not hasattr(acl_lib, "aclrtMallocHost") or not hasattr(acl_lib, "aclrtFreeHost"):
        raise RuntimeError("aclrtMallocHost/aclrtFreeHost not available")
    if not hasattr(acl_lib.aclrtMallocHost, "argtypes"):
        acl_lib.aclrtMallocHost.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        acl_lib.aclrtMallocHost.restype = ctypes.c_int
    p = ctypes.c_void_p(0)
    ret = acl_lib.aclrtMallocHost(ctypes.byref(p), total_bytes)
    if ret != 0 or p.value is None or p.value == 0:
        raise RuntimeError(f"aclrtMallocHost({total_bytes}) failed with code {ret}")
    return p.value


def free_pinned_host_buffer(ptr: int) -> None:
    """Release pinned host memory allocated by allocate_pinned_host_buffer."""
    if not hasattr(acl_lib.aclrtFreeHost, "argtypes"):
        acl_lib.aclrtFreeHost.argtypes = [ctypes.c_void_p]
        acl_lib.aclrtFreeHost.restype = ctypes.c_int
    if ptr != 0:
        acl_lib.aclrtFreeHost(ctypes.c_void_p(ptr))


# ---------------------------------------------------------------------------
# 3. Offset map
# ---------------------------------------------------------------------------

def build_offset_map(param_descs: list) -> tuple:
    """Build a {param_name → byte_offset} mapping from ordered descriptor list.

    Returns:
        (offset_map: dict[str, int], total_bytes: int)
    """
    offset_map = {}
    cursor = 0
    for p in param_descs:
        offset_map[p["name"]] = cursor
        cursor += p["size"]
    return offset_map, cursor


# ---------------------------------------------------------------------------
# 4. D2H / H2D / D2D snapshot helpers
# ---------------------------------------------------------------------------

def _ensure_acl_device(device_id: int) -> None:
    """Set the active ACL device (idempotent)."""
    if acl_lib is not None and hasattr(acl_lib, "aclrtSetDevice"):
        acl_lib.aclrtSetDevice(device_id)


def _check_acl_ret(ret: int, tag: str) -> None:
    if ret != 0:
        raise RuntimeError(f"aclrtMemcpy failed: {tag} returned {ret}")


def snapshot_d2h(
    param_descs: list,
    host_buf: np.ndarray,
    offset_map: dict,
    device_id: int = 0,
) -> dict:
    """Phase 1 (blocking): copy every parameter from HBM → host DRAM via aclrtMemcpy.

    The ``ms.hal.synchronize()`` wait and the actual aclrtMemcpy transfers are
    timed **separately** because synchronize latency varies with the GE queue
    depth (0–800 ms) and is NOT part of the D2H cost we want to compare.

    Args:
        param_descs:  from ``get_param_descriptors``.
        host_buf:     from ``allocate_host_buffer`` — destination.
        offset_map:   from ``build_offset_map``.
        device_id:    Ascend NPU device id.

    Returns:
        dict with keys:
          sync_ms   — time spent in ``ms.hal.synchronize()``
          memcpy_ms — time spent in the aclrtMemcpy loop
          total_ms  — sync_ms + memcpy_ms
    """
    _ensure_acl_device(device_id)

    t_sync = time.perf_counter()
    ms.hal.synchronize()
    sync_ms = (time.perf_counter() - t_sync) * 1000.0

    t0 = time.perf_counter()
    buf_ptr = host_buf.ctypes.data

    for p in param_descs:
        dst = ctypes.c_void_p(buf_ptr + offset_map[p["name"]])
        src = ctypes.c_void_p(p["ptr"])
        ret = acl_lib.aclrtMemcpy(dst, p["size"], src, p["size"],
                                   ACL_MEMCPY_DEVICE_TO_HOST)
        _check_acl_ret(ret, f"D2H {p['name']}")

    memcpy_ms = (time.perf_counter() - t0) * 1000.0
    return {"sync_ms": round(sync_ms, 3),
            "memcpy_ms": round(memcpy_ms, 3),
            "total_ms": round(sync_ms + memcpy_ms, 3)}


def snapshot_d2h_pinned(
    param_descs: list,
    pinned_ptr: int,
    offset_map: dict,
    device_id: int = 0,
) -> dict:
    """Like ``snapshot_d2h`` but writes to pinned (aclrtMallocHost) memory.

    Pinned memory gives ~6× higher D2H bandwidth on Ascend 910B because
    the DMA engine does not pay per-page pinning overhead.

    Args:
        param_descs:  from ``get_param_descriptors``.
        pinned_ptr:   from ``allocate_pinned_host_buffer``.
        offset_map:   from ``build_offset_map``.
        device_id:    Ascend NPU device id.

    Returns:
        dict with keys: sync_ms, memcpy_ms, total_ms (same shape as snapshot_d2h).
    """
    _ensure_acl_device(device_id)

    t_sync = time.perf_counter()
    ms.hal.synchronize()
    sync_ms = (time.perf_counter() - t_sync) * 1000.0

    t0 = time.perf_counter()
    for p in param_descs:
        dst = ctypes.c_void_p(pinned_ptr + offset_map[p["name"]])
        src = ctypes.c_void_p(p["ptr"])
        ret = acl_lib.aclrtMemcpy(dst, p["size"], src, p["size"],
                                   ACL_MEMCPY_DEVICE_TO_HOST)
        _check_acl_ret(ret, f"D2H(pinned) {p['name']}")

    memcpy_ms = (time.perf_counter() - t0) * 1000.0
    return {"sync_ms": round(sync_ms, 3),
            "memcpy_ms": round(memcpy_ms, 3),
            "total_ms": round(sync_ms + memcpy_ms, 3)}


def persist_to_file_pinned(
    pinned_ptr: int, total_bytes: int, filepath: str
) -> float:
    """Write pinned host buffer to kernel filesystem + fsync.

    Like ``persist_to_file`` but reads from pinned memory via ctypes
    string buffer.

    Returns:
        Wall-clock duration in milliseconds.
    """
    t0 = time.perf_counter()
    buf = (ctypes.c_uint8 * total_bytes).from_address(pinned_ptr)
    with open(filepath, "wb") as f:
        f.write(bytearray(buf))
        f.flush()
        os.fsync(f.fileno())
    return (time.perf_counter() - t0) * 1000.0


def restore_from_file(
    filepath: str,
    param_descs: list,
    offset_map: dict,
    device_id: int = 0,
) -> None:
    """Restore parameters from a raw binary checkpoint file.

    Reads the entire file into host memory then issues per-parameter
    aclrtMemcpy H2D transfers.

    Args:
        filepath:     path to the raw .ckpt file (binary concatenation).
        param_descs:  from ``get_param_descriptors``.
        offset_map:   from ``build_offset_map``.
        device_id:    Ascend NPU device id.
    """
    _ensure_acl_device(device_id)

    with open(filepath, "rb") as f:
        data = f.read()

    host_buf = np.frombuffer(data, dtype=np.uint8)

    for p in param_descs:
        src = ctypes.c_void_p(host_buf.ctypes.data + offset_map[p["name"]])
        dst = ctypes.c_void_p(p["ptr"])
        ret = acl_lib.aclrtMemcpy(dst, p["size"], src, p["size"],
                                   ACL_MEMCPY_HOST_TO_DEVICE)
        _check_acl_ret(ret, f"H2D {p['name']}")


# ---------------------------------------------------------------------------
# 5. Kernel FS persist
# ---------------------------------------------------------------------------

def persist_to_file(host_buf: np.ndarray, filepath: str) -> float:
    """Write host buffer to kernel filesystem + fsync.

    Runs in a background thread — the caller must ensure the host buffer
    stays alive for the duration of the call (the ``host_buf`` argument
    provides this guarantee via Python reference counting).

    Args:
        host_buf:  byte buffer (np.ndarray of uint8).
        filepath:  destination path on NVMe #2 kernel FS.

    Returns:
        Wall-clock duration in milliseconds (write + fsync).
    """
    t0 = time.perf_counter()
    with open(filepath, "wb") as f:
        f.write(host_buf.tobytes())
        f.flush()
        os.fsync(f.fileno())
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# 6. Contiguous HBM buffer (PCcheck gpu_ar equivalent)
# ---------------------------------------------------------------------------

MAX_CHUNK_BYTES = 1020 * 1024 * 1024  # 1020 MB — safely under the ~1 GB aclrtMalloc limit


def _malloc_chunk(size: int) -> int:
    """Allocate one HBM chunk via aclrtMalloc. Returns device pointer."""
    if not hasattr(acl_lib, "aclrtMalloc"):
        raise RuntimeError("aclrtMalloc not available")
    if not hasattr(acl_lib.aclrtMalloc, "argtypes"):
        acl_lib.aclrtMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
        acl_lib.aclrtMalloc.restype = ctypes.c_int
    p = ctypes.c_void_p(0)
    ret = acl_lib.aclrtMalloc(ctypes.byref(p), size, 0)
    if ret != 0 or p.value is None or p.value == 0:
        raise RuntimeError(f"aclrtMalloc({size}) failed with code {ret}")
    return p.value


def _free_chunk(ptr: int) -> None:
    if hasattr(acl_lib, "aclrtFree") and ptr != 0:
        if not hasattr(acl_lib.aclrtFree, "argtypes"):
            acl_lib.aclrtFree.argtypes = [ctypes.c_void_p]
            acl_lib.aclrtFree.restype = ctypes.c_int
        acl_lib.aclrtFree(ctypes.c_void_p(ptr))


def build_flat_hbm_buffer(
    model: ms.nn.Cell,
    param_descs: list,
    offset_map: dict,
    total_bytes: int,
    device_id: int = 0,
) -> list:
    """Build a logical contiguous HBM buffer as N chunks (each ≤1 GB).

    Ascend 910B's aclrtMalloc rejects single allocations > ~1 GB (error
    107000).  We work around this by allocating multiple chunks and
    distributing parameters across them.

    Each chunk is a tuple ``(ptr, start_offset, end_offset)``.

    After calling this, use ``snapshot_d2h_chunked`` to D2H-copy the
    chunks into a contiguous host buffer.

    Args:
        model:        warm-allocated MindSpore model.
        param_descs:  from ``get_param_descriptors``.
        offset_map:   from ``build_offset_map``.
        total_bytes:  total byte count.
        device_id:    Ascend NPU device id.

    Returns:
        list of (chunk_ptr: int, start: int, end: int)
    """
    _ensure_acl_device(device_id)

    chunks = []          # [(ptr, start_byte, end_byte), ...]
    current_start = 0
    params_in_chunk = []

    for p in param_descs:
        p_start = offset_map[p["name"]]
        p_end = p_start + p["size"]

        # If adding this param would overflow the current chunk, seal it
        if params_in_chunk and (p_end - current_start) > MAX_CHUNK_BYTES:
            chunk_size = params_in_chunk[-1]["offset"] + params_in_chunk[-1]["size"] - current_start
            ptr = _malloc_chunk(chunk_size)
            chunks.append((ptr, current_start, current_start + chunk_size))
            print(f"[two_phase_common] chunk {len(chunks)-1}: "
                  f"aclrtMalloc({chunk_size}) → 0x{ptr:016x}")
            # D2D params into this chunk
            for pc in params_in_chunk:
                dst = ctypes.c_void_p(ptr + pc["offset"] - current_start)
                src = ctypes.c_void_p(pc["ptr"])
                acl_lib.aclrtMemcpy(dst, pc["size"], src, pc["size"],
                                     ACL_MEMCPY_DEVICE_TO_DEVICE)
            current_start = p_start
            params_in_chunk = []

        params_in_chunk.append({
            "offset": p_start, "size": p["size"], "ptr": p["ptr"],
            "name": p["name"],
        })

    # Final chunk
    if params_in_chunk:
        chunk_size = total_bytes - current_start
        ptr = _malloc_chunk(chunk_size)
        chunks.append((ptr, current_start, total_bytes))
        print(f"[two_phase_common] chunk {len(chunks)-1}: "
              f"aclrtMalloc({chunk_size}) → 0x{ptr:016x}")
        for pc in params_in_chunk:
            dst = ctypes.c_void_p(ptr + pc["offset"] - current_start)
            src = ctypes.c_void_p(pc["ptr"])
            acl_lib.aclrtMemcpy(dst, pc["size"], src, pc["size"],
                                 ACL_MEMCPY_DEVICE_TO_DEVICE)

    print(f"[two_phase_common] {len(chunks)} chunks, {total_bytes/1e9:.2f} GB total")
    return chunks


def free_flat_hbm_chunks(chunks: list) -> None:
    """Release chunks allocated by build_flat_hbm_buffer."""
    for ptr, start, end in chunks:
        _free_chunk(ptr)


def d2d_to_chunks(
    chunks: list,
    param_descs: list,
    offset_map: dict,
    device_id: int = 0,
) -> None:
    """Re-copy all parameters (D2D) into pre-allocated flat HBM chunks.

    Called at each checkpoint to capture the current parameter state.
    """
    _ensure_acl_device(device_id)
    for ptr, chunk_start, chunk_end in chunks:
        for p in param_descs:
            p_start = offset_map[p["name"]]
            p_end = p_start + p["size"]
            # Does this param fall within this chunk?
            if p_start >= chunk_end or p_end <= chunk_start:
                continue
            local_offset = p_start - chunk_start
            dst = ctypes.c_void_p(ptr + local_offset)
            src = ctypes.c_void_p(p["ptr"])
            ret = acl_lib.aclrtMemcpy(dst, p["size"], src, p["size"],
                                       ACL_MEMCPY_DEVICE_TO_DEVICE)
            _check_acl_ret(ret, f"D2D {p['name']}")


def snapshot_d2h_chunked(
    chunks: list,
    host_buf: np.ndarray,
    device_id: int = 0,
) -> float:
    """Bulk D2H copy of chunked flat HBM buffer → contiguous host buffer.

    Each chunk is copied with a single aclrtMemcpy D2H call.

    Args:
        chunks:    from ``build_flat_hbm_buffer``.
        host_buf:  destination byte buffer (size must match total).
        device_id: Ascend NPU device id.

    Returns:
        Wall-clock duration in milliseconds.
    """
    _ensure_acl_device(device_id)
    ms.hal.synchronize()

    t0 = time.perf_counter()
    buf_ptr = host_buf.ctypes.data

    for ptr, start, end in chunks:
        chunk_size = end - start
        dst = ctypes.c_void_p(buf_ptr + start)
        src = ctypes.c_void_p(ptr)
        ret = acl_lib.aclrtMemcpy(dst, chunk_size, src, chunk_size,
                                   ACL_MEMCPY_DEVICE_TO_HOST)
        _check_acl_ret(ret, f"D2H chunk 0x{ptr:016x}")

    return (time.perf_counter() - t0) * 1000.0


def snapshot_d2h_from_flat(
    flat_ptr: int,
    total_bytes: int,
    host_buf: np.ndarray,
    device_id: int = 0,
) -> float:
    """Single-bulk D2H copy of the contiguous flat HBM buffer → host.

    More efficient than ``snapshot_d2h`` (which issues one aclrtMemcpy
    per parameter) because the PCIe DMA setup overhead is paid only once.

    Args:
        flat_ptr:    device pointer returned by ``build_flat_hbm_buffer``.
        total_bytes: size of the flat buffer.
        host_buf:    destination byte buffer.
        device_id:   Ascend NPU device id.

    Returns:
        Wall-clock duration in milliseconds.
    """
    _ensure_acl_device(device_id)
    ms.hal.synchronize()

    t0 = time.perf_counter()
    dst = ctypes.c_void_p(host_buf.ctypes.data)
    src = ctypes.c_void_p(flat_ptr)
    ret = acl_lib.aclrtMemcpy(dst, total_bytes, src, total_bytes,
                               ACL_MEMCPY_DEVICE_TO_HOST)
    _check_acl_ret(ret, "D2H flat")
    return (time.perf_counter() - t0) * 1000.0
