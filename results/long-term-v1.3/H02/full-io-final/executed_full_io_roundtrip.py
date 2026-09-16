#!/usr/bin/env python3
"""FULL-only Host/HBM/NVMe roundtrip matrix with context reconstruction."""

import argparse
import ctypes
import hashlib
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from c_bindings import NPUNVMEContext, acl_lib, lib  # noqa: E402


BASE = 64 * 1024**3
ALIGN = 4096
SIZES = (4096, 4096 + 137, 64 * 1024, 1024 * 1024,
         4 * 1024**2, 16 * 1024**2, 20 * 1024**2 + 731)


def aligned(value):
    return (int(value) + ALIGN - 1) & ~(ALIGN - 1)


def pattern(size, seed):
    return bytes(((index * 131 + seed * 17) & 0xFF) for index in range(size))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def open_context(args, shm_id):
    os.environ["SPDK_SHM_ID"] = str(shm_id)
    ctx = ctypes.POINTER(NPUNVMEContext)()
    rc = lib.npu_nvme_init(ctypes.byref(ctx), args.pci.encode(), args.npu,
                           args.depth, args.chunk, True,
                           args.output.encode())
    require(rc == 0 and bool(ctx), f"npu_nvme_init failed: {rc}")
    return ctx


def arrays(pointer, offset, size):
    return ((ctypes.c_void_p * 1)(pointer),
            (ctypes.c_uint64 * 1)(offset),
            (ctypes.c_size_t * 1)(size))


def write_read_host(ctx, payload, offset, chunk):
    source = ctypes.create_string_buffer(payload, len(payload))
    target = ctypes.create_string_buffer(len(payload))
    for inner in range(0, len(payload), chunk):
        size = min(chunk, len(payload) - inner)
        ptrs, offsets, sizes = arrays(ctypes.addressof(source) + inner,
                                      offset + inner, size)
        require(lib.npu_nvme_write_batch_host(ctx, ptrs, offsets, sizes, 1) == 0,
                f"host write failed size={size}")
        read_ptrs, _, _ = arrays(ctypes.addressof(target) + inner,
                                 offset + inner, size)
        require(lib.npu_nvme_read_batch_host(ctx, read_ptrs, offsets, sizes, 1) == 0,
                f"host read failed size={size}")
    require(target.raw == payload, f"host mismatch size={len(payload)}")


def host_matrix(ctx):
    records = []
    offset = BASE
    for index, size in enumerate(SIZES):
        payload = pattern(size, index + 1)
        write_read_host(ctx, payload, offset, args.chunk)
        records.append((offset, size, digest(payload)))
        offset += aligned(size) + ALIGN
    require(lib.npu_nvme_flush(ctx) == 0, "host matrix flush failed")
    return records, offset


def verify_host_after_restart(ctx, records):
    for offset, size, expected in records:
        target = ctypes.create_string_buffer(size)
        for inner in range(0, size, args.chunk):
            part = min(args.chunk, size - inner)
            ptrs, offsets, sizes = arrays(ctypes.addressof(target) + inner,
                                          offset + inner, part)
            require(lib.npu_nvme_read_batch_host(ctx, ptrs, offsets, sizes, 1) == 0,
                    f"restart host read failed size={part}")
        require(digest(target.raw) == expected,
                f"restart host checksum mismatch size={size}")


def hbm_matrix(ctx, start_offset):
    acl_lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_size_t, ctypes.c_int]
    acl_lib.aclrtMalloc.restype = ctypes.c_int
    acl_lib.aclrtFree.argtypes = [ctypes.c_void_p]
    acl_lib.aclrtFree.restype = ctypes.c_int
    require(acl_lib.aclrtSetDevice(args.npu) == 0, "aclrtSetDevice failed")
    offset = aligned(start_offset)
    for index, size in enumerate(SIZES):
        payload = pattern(size, index + 101)
        source_host = ctypes.create_string_buffer(payload, size)
        result_host = ctypes.create_string_buffer(size)
        source_dev = ctypes.c_void_p()
        target_dev = ctypes.c_void_p()
        require(acl_lib.aclrtMalloc(ctypes.byref(source_dev), size, 0) == 0,
                f"source aclrtMalloc failed size={size}")
        require(acl_lib.aclrtMalloc(ctypes.byref(target_dev), size, 0) == 0,
                f"target aclrtMalloc failed size={size}")
        try:
            require(acl_lib.aclrtMemcpy(source_dev, size, source_host, size, 1) == 0,
                    f"H2D failed size={size}")
            for inner in range(0, size, args.chunk):
                part = min(args.chunk, size - inner)
                ptrs, offsets, sizes = arrays(source_dev.value + inner,
                                              offset + inner, part)
                require(lib.npu_nvme_write_batch(ctx, ptrs, offsets, sizes, 1) == 0,
                        f"HBM write failed size={part}")
                read_ptrs, _, _ = arrays(target_dev.value + inner,
                                         offset + inner, part)
                require(lib.npu_nvme_read_batch(ctx, read_ptrs, offsets, sizes, 1) == 0,
                        f"HBM read failed size={part}")
            require(acl_lib.aclrtMemcpy(result_host, size, target_dev, size, 2) == 0,
                    f"D2H failed size={size}")
            require(result_host.raw == payload, f"HBM mismatch size={size}")
        finally:
            acl_lib.aclrtFree(target_dev)
            acl_lib.aclrtFree(source_dev)
        offset += aligned(size) + ALIGN
    require(lib.npu_nvme_flush(ctx) == 0, "HBM matrix flush failed")
    return offset


def negative_matrix(ctx, capacity):
    dummy = ctypes.create_string_buffer(ALIGN)
    null_ptrs = (ctypes.c_void_p * 1)(ctypes.addressof(dummy))
    offsets = (ctypes.c_uint64 * 1)(BASE)
    zero = (ctypes.c_size_t * 1)(0)
    require(lib.npu_nvme_write_batch_host(ctx, null_ptrs, offsets, zero, 1) != 0,
            "empty object was accepted")
    bad_offsets = (ctypes.c_uint64 * 1)(BASE + 1)
    one = (ctypes.c_size_t * 1)(ALIGN)
    require(lib.npu_nvme_write_batch_host(ctx, null_ptrs, bad_offsets, one, 1) != 0,
            "unaligned offset was accepted")
    end = (ctypes.c_uint64 * 1)(capacity - ALIGN)
    too_large = (ctypes.c_size_t * 1)(2 * ALIGN)
    require(lib.npu_nvme_write_batch_host(ctx, null_ptrs, end, too_large, 1) != 0,
            "capacity overflow was accepted")


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--chunk", type=int, default=4 * 1024**2)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--shm-id", type=int, default=18071)
    parser.add_argument("--output", default="/tmp/npu-nvme-full-roundtrip")
    args = parser.parse_args()
    require(lib is not None and acl_lib is not None, "ACL/SPDK unavailable")
    os.makedirs(args.output, exist_ok=True)

    first = open_context(args, args.shm_id)
    try:
        capacity = int(lib.npu_nvme_get_total_blocks(first))
        require(capacity > BASE + 256 * 1024**2, "safe test region unavailable")
        host_records, next_offset = host_matrix(first)
        next_offset = hbm_matrix(first, next_offset)
        # Mixed Host/HBM objects occupy one generation's non-overlapping range.
        require(next_offset < BASE + 256 * 1024**2, "matrix exceeded test region")
        negative_matrix(first, capacity)
    finally:
        lib.npu_nvme_cleanup(first)

    second = open_context(args, args.shm_id + 1)
    try:
        verify_host_after_restart(second, host_records)
    finally:
        lib.npu_nvme_cleanup(second)
    print("[FULL-IO] PASS host/HBM/mixed/negative/context-restart matrix",
          flush=True)


if __name__ == "__main__":
    main()
