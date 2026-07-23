"""Hardware roundtrip test for Python raw async NPU-SSD transfers."""

import argparse
import ctypes
import json
import time

from c_bindings import acl_lib, lib, NPUNVMEContext
from disk_layout import BLOCK_SIZE, HEAP_START_OFFSET
from raw_io import RawIO


ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
DEFAULT_SIZE_MB = 64
DEFAULT_OFFSET_PADDING = 1024 * 1024 * 1024


def _round_up(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


def _bind_acl_runtime():
    if acl_lib is None:
        raise RuntimeError("libascendcl.so is unavailable")
    acl_lib.aclrtMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    acl_lib.aclrtMalloc.restype = ctypes.c_int
    acl_lib.aclrtFree.argtypes = [ctypes.c_void_p]
    acl_lib.aclrtFree.restype = ctypes.c_int
    acl_lib.aclrtSetDevice.argtypes = [ctypes.c_int]
    acl_lib.aclrtSetDevice.restype = ctypes.c_int


def _device_malloc(size):
    ptr = ctypes.c_void_p()
    rc = acl_lib.aclrtMalloc(
        ctypes.byref(ptr), ctypes.c_size_t(size), ACL_MEM_MALLOC_HUGE_FIRST)
    if rc != 0 or not ptr.value:
        raise RuntimeError(f"aclrtMalloc failed: rc={rc}")
    return ptr


def _copy_to_device(dst, src, size):
    rc = acl_lib.aclrtMemcpy(
        dst, ctypes.c_size_t(size), ctypes.c_void_p(ctypes.addressof(src)),
        ctypes.c_size_t(size), ACL_MEMCPY_HOST_TO_DEVICE)
    if rc != 0:
        raise RuntimeError(f"aclrtMemcpy host->device failed: rc={rc}")


def _copy_from_device(dst, src, size):
    rc = acl_lib.aclrtMemcpy(
        ctypes.c_void_p(ctypes.addressof(dst)), ctypes.c_size_t(size), src,
        ctypes.c_size_t(size), ACL_MEMCPY_DEVICE_TO_HOST)
    if rc != 0:
        raise RuntimeError(f"aclrtMemcpy device->host failed: rc={rc}")


def _make_payload(size):
    pattern = bytes((i * 17 + 3) % 251 for i in range(BLOCK_SIZE))
    return pattern * (size // len(pattern))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pci_addr", nargs="?", default="0000:83:00.0")
    parser.add_argument("npu_id", nargs="?", type=int, default=0)
    parser.add_argument("--size-mb", type=int, default=DEFAULT_SIZE_MB)
    parser.add_argument("--offset", type=int, default=None)
    args = parser.parse_args()

    _bind_acl_runtime()
    rc = acl_lib.aclrtSetDevice(args.npu_id)
    if rc != 0:
        raise RuntimeError(f"aclrtSetDevice failed: rc={rc}")

    size = _round_up(args.size_mb * 1024 * 1024, BLOCK_SIZE)
    offset = args.offset
    if offset is None:
        offset = _round_up(HEAP_START_OFFSET + DEFAULT_OFFSET_PADDING, BLOCK_SIZE)
    if offset % BLOCK_SIZE != 0:
        raise ValueError("offset must be 4 KiB aligned")

    ctx = ctypes.POINTER(NPUNVMEContext)()
    rc = lib.npu_nvme_init(ctypes.byref(ctx), args.pci_addr.encode(), args.npu_id,
                           4, 4 * 1024 * 1024, False, b".")
    if rc != 0:
        raise RuntimeError(f"npu_nvme_init failed: rc={rc}")

    dev_src = ctypes.c_void_p()
    dev_dst = ctypes.c_void_p()
    try:
        total_bytes = lib.npu_nvme_get_total_blocks(ctx)
        if offset + size > total_bytes:
            raise ValueError(
                f"test range [{offset}, {offset + size}) exceeds disk capacity "
                f"{total_bytes}")

        dev_src = _device_malloc(size)
        dev_dst = _device_malloc(size)
        payload = _make_payload(size)
        host_src = ctypes.create_string_buffer(payload, size)
        host_dst = ctypes.create_string_buffer(size)
        _copy_to_device(dev_src, host_src, size)

        raw = RawIO(ctx)

        write_submit_start = time.perf_counter()
        write_future = raw.write_async([dev_src.value], [offset], [size])
        write_submit_us = (time.perf_counter() - write_submit_start) * 1000000.0
        write_done_after_submit = write_future.done()

        write_wait_start = time.perf_counter()
        write_future.result(timeout=30.0)
        write_wait_s = time.perf_counter() - write_wait_start

        read_submit_start = time.perf_counter()
        read_future = raw.read_async([dev_dst.value], [offset], [size])
        read_submit_us = (time.perf_counter() - read_submit_start) * 1000000.0
        read_done_after_submit = read_future.done()

        read_wait_start = time.perf_counter()
        read_future.result(timeout=30.0)
        read_wait_s = time.perf_counter() - read_wait_start
        _copy_from_device(host_dst, dev_dst, size)

        if host_dst.raw[:size] != payload:
            raise RuntimeError("raw async payload roundtrip mismatch")

        elapsed_s = write_wait_s + read_wait_s
        bandwidth_mb_s = (2 * size / 1024 / 1024 / elapsed_s) if elapsed_s > 0 else 0.0
        print(json.dumps({
            "status": "ok",
            "pci_addr": args.pci_addr,
            "npu_id": args.npu_id,
            "offset": offset,
            "size_bytes": size,
            "write_submit_us": write_submit_us,
            "read_submit_us": read_submit_us,
            "write_done_after_submit": write_done_after_submit,
            "read_done_after_submit": read_done_after_submit,
            "write_wait_s": write_wait_s,
            "read_wait_s": read_wait_s,
            "bandwidth_mb_s": bandwidth_mb_s,
        }, ensure_ascii=True))
    finally:
        if dev_src.value:
            acl_lib.aclrtFree(dev_src)
        if dev_dst.value:
            acl_lib.aclrtFree(dev_dst)
        if ctx:
            lib.npu_nvme_cleanup(ctx)


if __name__ == "__main__":
    main()
