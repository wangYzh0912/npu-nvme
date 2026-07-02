"""Checkpoint-independent raw NPU-SSD transfer helpers."""

import ctypes

from c_bindings import lib, NPUNVMEContext


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


class RawIO:
    def __init__(self, ctx):
        self.ctx = ctx

    def write_host(self, ptrs, offsets, sizes):
        if not hasattr(lib, "npu_nvme_raw_write_batch_host"):
            raise RuntimeError("raw_write_batch_host is unavailable")
        return lib.npu_nvme_raw_write_batch_host(
            self.ctx, _as_ptr_array(ptrs), _as_u64_array(offsets),
            _as_size_array(sizes), len(ptrs))

    def read_host(self, ptrs, offsets, sizes):
        if not hasattr(lib, "npu_nvme_raw_read_batch_host"):
            raise RuntimeError("raw_read_batch_host is unavailable")
        return lib.npu_nvme_raw_read_batch_host(
            self.ctx, _as_ptr_array(ptrs), _as_u64_array(offsets),
            _as_size_array(sizes), len(ptrs))
