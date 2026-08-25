#!/usr/bin/env python3
"""G0/G2 hardware correctness gate for one formatted V2 NVMe namespace.

The test deliberately uses the unallocated gap beginning at 64 GiB.  It never
writes the V2 superblock, metadata replicas, FULL slots, or the tail Delta
ring.  Run as root after checking ``npu-smi info``.
"""

import argparse
import ctypes
import os
import sys
import time


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from c_bindings import NPUNVMEContext, acl_lib, lib  # noqa: E402


TEST_BASE = 64 * 1024**3
CHUNK = 1024 * 1024


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def make_array(values):
    array_type = ctypes.c_void_p * len(values)
    return array_type(*(ctypes.c_void_p(value) for value in values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=1)
    args = parser.parse_args()

    require(acl_lib is not None and lib is not None, "ACL/SPDK libraries are unavailable")
    acl_lib.aclrtMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    acl_lib.aclrtMalloc.restype = ctypes.c_int
    acl_lib.aclrtFree.argtypes = [ctypes.c_void_p]
    acl_lib.aclrtFree.restype = ctypes.c_int
    require(acl_lib.aclrtSetDevice(args.npu) == 0, "aclrtSetDevice failed")

    ctx = ctypes.POINTER(NPUNVMEContext)()
    rc = lib.npu_nvme_init(
        ctypes.byref(ctx), args.pci.encode(), args.npu, 4, 4 * 1024 * 1024,
        False, os.path.join(REPO_ROOT, "experiments", "output", "gates").encode())
    require(rc == 0 and bool(ctx), f"npu_nvme_init failed: {rc}")

    dev_ptrs = []
    try:
        total_bytes = int(lib.npu_nvme_get_total_blocks(ctx))
        require(total_bytes > TEST_BASE + 8 * CHUNK, "test gap is outside device")

        host_a = ctypes.create_string_buffer(bytes([0x3C]) * CHUNK)
        host_b = ctypes.create_string_buffer(bytes([0xA7]) * CHUNK)
        host_read_a = ctypes.create_string_buffer(CHUNK)
        host_read_b = ctypes.create_string_buffer(CHUNK)
        ptrs = make_array([ctypes.addressof(host_a), ctypes.addressof(host_b)])
        offsets = (ctypes.c_uint64 * 2)(TEST_BASE, TEST_BASE + 4 * 1024 * 1024)
        sizes = (ctypes.c_size_t * 2)(CHUNK, CHUNK)

        print("[G0] host multi-item write/read", flush=True)
        require(lib.npu_nvme_write_batch_host(ctx, ptrs, offsets, sizes, 2) == 0,
                "host write failed")
        read_ptrs = make_array([ctypes.addressof(host_read_a), ctypes.addressof(host_read_b)])
        require(lib.npu_nvme_read_batch_host(ctx, read_ptrs, offsets, sizes, 2) == 0,
                "host read failed")
        require(host_read_a.raw == host_a.raw[:CHUNK] and
                host_read_b.raw == host_b.raw[:CHUNK],
                "host roundtrip mismatch: "
                f"a={host_read_a.raw[:16].hex()} b={host_read_b.raw[:16].hex()}")

        print("[G0] invalid alignment/capacity rejection", flush=True)
        bad_offset = (ctypes.c_uint64 * 1)(TEST_BASE + 1)
        one_ptr = make_array([ctypes.addressof(host_a)])
        one_size = (ctypes.c_size_t * 1)(CHUNK)
        require(lib.npu_nvme_write_batch_host(ctx, one_ptr, bad_offset, one_size, 1) != 0,
                "unaligned offset was accepted")
        end_offset = (ctypes.c_uint64 * 1)(total_bytes - 4096)
        too_big = (ctypes.c_size_t * 1)(8192)
        require(lib.npu_nvme_write_batch_host(ctx, one_ptr, end_offset, too_big, 1) != 0,
                "out-of-capacity write was accepted")

        print("[G0] NPU device-buffer write/read", flush=True)
        for fill in (0x5D, 0xC1):
            device_ptr = ctypes.c_void_p()
            require(acl_lib.aclrtMalloc(ctypes.byref(device_ptr), CHUNK, 0) == 0,
                    "aclrtMalloc failed")
            dev_ptrs.append(device_ptr)
            source = ctypes.create_string_buffer(bytes([fill]) * CHUNK)
            require(acl_lib.aclrtMemcpy(
                device_ptr, CHUNK, ctypes.byref(source), CHUNK, 1) == 0,
                    "host-to-device copy failed")
            npu_ptrs = make_array([device_ptr.value])
            one_offset = (ctypes.c_uint64 * 1)(TEST_BASE + (8 + len(dev_ptrs)) * CHUNK)
            one_size = (ctypes.c_size_t * 1)(CHUNK)
            require(lib.npu_nvme_write_batch(ctx, npu_ptrs, one_offset, one_size, 1) == 0,
                    "NPU write failed")
            read_device = ctypes.c_void_p()
            require(acl_lib.aclrtMalloc(ctypes.byref(read_device), CHUNK, 0) == 0,
                    "second aclrtMalloc failed")
            dev_ptrs.append(read_device)
            read_ptr = make_array([read_device.value])
            require(lib.npu_nvme_read_batch(ctx, read_ptr, one_offset, one_size, 1) == 0,
                    "NPU read failed")
            result = ctypes.create_string_buffer(CHUNK)
            require(acl_lib.aclrtMemcpy(
                ctypes.byref(result), CHUNK, read_device, CHUNK, 2) == 0,
                    "device-to-host copy failed")
            require(result.raw == source.raw[:CHUNK], "NPU roundtrip mismatch")

        print("[G2] bounded metadata timeout and reactor recovery", flush=True)
        os.environ.setdefault("NPU_NVME_TEST_META_DELAY_MS", "200")
        require(hasattr(lib, "npu_nvme_set_io_timeout_ms"),
                "timeout API is missing")
        require(lib.npu_nvme_set_io_timeout_ms(ctx, 50) == 0,
                "timeout setter failed")
        meta_value = ctypes.create_string_buffer(b"TIMEOUT-GATE" + b"\0" * (4096 - 12))
        start = time.monotonic()
        timed_out = lib.npu_nvme_sync_meta_io(
            ctx, TEST_BASE + 16 * CHUNK, 4096, 0, ctypes.byref(meta_value))
        elapsed = time.monotonic() - start
        require(timed_out != 0 and elapsed < 0.15,
                f"metadata timeout not bounded: rc={timed_out}, elapsed={elapsed:.3f}s")
        time.sleep(0.35)
        require(lib.npu_nvme_set_io_timeout_ms(ctx, 2000) == 0,
                "timeout reset failed")
        verify = ctypes.create_string_buffer(4096)
        require(lib.npu_nvme_sync_meta_io(
            ctx, TEST_BASE + 16 * CHUNK, 4096, 1, ctypes.byref(verify)) == 0,
                "reactor did not recover after timed-out metadata request")
        require(verify.raw == meta_value.raw[:4096],
                "metadata recovery payload mismatch")
        print("[G0/G2] PASS", flush=True)
    finally:
        for ptr in reversed(dev_ptrs):
            acl_lib.aclrtFree(ptr)
        lib.npu_nvme_cleanup(ctx)


if __name__ == "__main__":
    main()
