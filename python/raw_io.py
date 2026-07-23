"""Checkpoint-independent raw NPU-SSD transfer helpers."""

import ctypes

from c_bindings import lib, NPUNVMEContext, NPUNVMERequest
from disk_layout import HEAP_START_OFFSET, BLOCK_SIZE


def _as_ptr_array(ptrs):
    arr = (ctypes.c_void_p * len(ptrs))()
    for i, ptr in enumerate(ptrs):
        arr[i] = ctypes.c_void_p(ptr)
    return arr


def _as_u64_array(values):
    arr = (ctypes.c_uint64 * len(values))()
    for i, value in enumerate(values):
        arr[i] = ctypes.c_uint64(int(value))
    return arr


def _as_size_array(values):
    arr = (ctypes.c_size_t * len(values))()
    for i, value in enumerate(values):
        arr[i] = ctypes.c_size_t(int(value))
    return arr


def _validate_arrays(ptrs, offsets, sizes):
    if not (len(ptrs) == len(offsets) == len(sizes)):
        raise ValueError("ptrs, offsets and sizes must have the same length")
    if len(ptrs) == 0:
        raise ValueError("at least one I/O item is required")


def _validate_raw_layout(offsets, sizes):
    for offset, size in zip(offsets, sizes):
        offset = int(offset)
        size = int(size)
        if offset < HEAP_START_OFFSET:
            raise ValueError(
                f"raw I/O offset {offset} overlaps protected metadata area")
        if offset % BLOCK_SIZE != 0 or size <= 0 or size % BLOCK_SIZE != 0:
            raise ValueError("raw I/O offsets and sizes must be 4 KiB aligned")


def _split_raw_items(ctx, ptrs, offsets, sizes):
    chunk_size = int(lib.npu_nvme_get_max_transfer(ctx))
    if chunk_size <= 0:
        raise RuntimeError("npu_nvme_get_max_transfer returned an invalid value")

    split_ptrs = []
    split_offsets = []
    split_sizes = []
    for ptr, offset, size in zip(ptrs, offsets, sizes):
        base_ptr = int(ptr)
        base_offset = int(offset)
        remaining = int(size)
        inner_offset = 0
        while remaining > 0:
            take = min(remaining, chunk_size)
            split_ptrs.append(base_ptr + inner_offset)
            split_offsets.append(base_offset + inner_offset)
            split_sizes.append(take)
            inner_offset += take
            remaining -= take
    return split_ptrs, split_offsets, split_sizes


class NPUNVMERequestFuture:
    """Small Future-like wrapper around a C npu_nvme_request_t handle."""

    def __init__(self, req, keepalive=(), result_value=None):
        self._req = req
        self._keepalive = tuple(keepalive)
        self._result_value = result_value
        self._closed = False

    def done(self):
        """Return True when the underlying C request is no longer pending."""
        if self._closed or not self._req:
            return True
        return lib.npu_nvme_request_poll(self._req) != 0

    def result(self, timeout=None):
        """Wait for completion, release the C request, and return result_value."""
        if self._closed or not self._req:
            return self._result_value

        timeout_us = 0 if timeout is None else int(float(timeout) * 1000000)
        wait_rc = lib.npu_nvme_request_wait(self._req, ctypes.c_uint64(timeout_us))
        if wait_rc == -2:
            raise TimeoutError("npu_nvme async request timed out")
        result = lib.npu_nvme_request_result(self._req)
        free_rc = lib.npu_nvme_request_free(self._req)
        self._req = None
        self._keepalive = ()
        self._closed = True
        if wait_rc != 0 or result != 0:
            raise RuntimeError(
                f"npu_nvme async request failed: wait_rc={wait_rc}, result={result}")
        if free_rc != 0:
            raise RuntimeError(f"npu_nvme_request_free failed: rc={free_rc}")
        return self._result_value

    def close(self):
        """Release a completed request if possible."""
        if self._closed or not self._req:
            return
        if lib.npu_nvme_request_poll(self._req) != 0:
            lib.npu_nvme_request_free(self._req)
            self._req = None
            self._keepalive = ()
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RawIO:
    def __init__(self, ctx):
        self.ctx = ctx

    def write_host(self, ptrs, offsets, sizes):
        _validate_arrays(ptrs, offsets, sizes)
        _validate_raw_layout(offsets, sizes)
        ptrs, offsets, sizes = _split_raw_items(self.ctx, ptrs, offsets, sizes)
        if not hasattr(lib, "npu_nvme_raw_write_batch_host"):
            raise RuntimeError("raw_write_batch_host is unavailable")
        return lib.npu_nvme_raw_write_batch_host(
            self.ctx, _as_ptr_array(ptrs), _as_u64_array(offsets),
            _as_size_array(sizes), len(ptrs))

    def read_host(self, ptrs, offsets, sizes):
        _validate_arrays(ptrs, offsets, sizes)
        _validate_raw_layout(offsets, sizes)
        ptrs, offsets, sizes = _split_raw_items(self.ctx, ptrs, offsets, sizes)
        if not hasattr(lib, "npu_nvme_raw_read_batch_host"):
            raise RuntimeError("raw_read_batch_host is unavailable")
        return lib.npu_nvme_raw_read_batch_host(
            self.ctx, _as_ptr_array(ptrs), _as_u64_array(offsets),
            _as_size_array(sizes), len(ptrs))

    def write_async(self, npu_ptrs, offsets, sizes):
        """Submit NPU HBM -> NVMe raw writes and return a request future."""
        if not hasattr(lib, "npu_nvme_raw_write_batch_async"):
            raise RuntimeError("raw_write_batch_async is unavailable")
        _validate_arrays(npu_ptrs, offsets, sizes)
        _validate_raw_layout(offsets, sizes)
        npu_ptrs, offsets, sizes = _split_raw_items(
            self.ctx, npu_ptrs, offsets, sizes)
        c_ptrs = _as_ptr_array(npu_ptrs)
        c_offs = _as_u64_array(offsets)
        c_sizes = _as_size_array(sizes)
        req = ctypes.POINTER(NPUNVMERequest)()
        rc = lib.npu_nvme_raw_write_batch_async(
            self.ctx, c_ptrs, c_offs, c_sizes, len(npu_ptrs), ctypes.byref(req))
        if rc != 0 or not req:
            raise RuntimeError(f"npu_nvme_raw_write_batch_async failed, rc={rc}")
        return NPUNVMERequestFuture(req, (c_ptrs, c_offs, c_sizes))

    def read_async(self, npu_ptrs, offsets, sizes):
        """Submit NVMe -> NPU HBM raw reads and return a request future."""
        if not hasattr(lib, "npu_nvme_raw_read_batch_async"):
            raise RuntimeError("raw_read_batch_async is unavailable")
        _validate_arrays(npu_ptrs, offsets, sizes)
        _validate_raw_layout(offsets, sizes)
        npu_ptrs, offsets, sizes = _split_raw_items(
            self.ctx, npu_ptrs, offsets, sizes)
        c_ptrs = _as_ptr_array(npu_ptrs)
        c_offs = _as_u64_array(offsets)
        c_sizes = _as_size_array(sizes)
        req = ctypes.POINTER(NPUNVMERequest)()
        rc = lib.npu_nvme_raw_read_batch_async(
            self.ctx, c_ptrs, c_offs, c_sizes, len(npu_ptrs), ctypes.byref(req))
        if rc != 0 or not req:
            raise RuntimeError(f"npu_nvme_raw_read_batch_async failed, rc={rc}")
        return NPUNVMERequestFuture(req, (c_ptrs, c_offs, c_sizes))
