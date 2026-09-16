#!/usr/bin/env python3
"""I5 NPU-HBM-SPDK byte-preserving S2 frame loopback."""

import argparse
import ctypes
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

from c_bindings import NPUNVMEContext, acl_lib, lib  # noqa: E402
from s2_delta import S2DeltaOracle  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=951)
    parser.add_argument("--offset", type=int, default=64 * 1024**3 + 16 * 1024**2)
    args = parser.parse_args()
    os.environ.setdefault("SPDK_SHM_ID", str(args.shm_id))

    initial = {
        "backbone.blocks.0.weight": np.arange(33, dtype=np.float32),
        "backbone.layernorm.bias": np.array([1, 2, 3], dtype=np.float16),
    }
    current = {name: value.copy() for name, value in initial.items()}
    current["backbone.blocks.0.weight"][32] = -17.25
    oracle = S2DeltaOracle(initial, block_size=8, small_threshold=4)
    oracle.set_current(current)
    frame = oracle.observe(1)
    aligned = (len(frame) + 4095) // 4096 * 4096

    acl_lib.aclrtSetDevice.argtypes = [ctypes.c_int]
    acl_lib.aclrtSetDevice.restype = ctypes.c_int
    acl_lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    acl_lib.aclrtMalloc.restype = ctypes.c_int
    acl_lib.aclrtFree.argtypes = [ctypes.c_void_p]
    acl_lib.aclrtFree.restype = ctypes.c_int
    acl_lib.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    acl_lib.aclrtMemcpy.restype = ctypes.c_int
    if acl_lib.aclrtSetDevice(args.npu) != 0:
        raise RuntimeError("aclrtSetDevice failed")

    ctx = ctypes.POINTER(NPUNVMEContext)()
    rc = lib.npu_nvme_init(ctypes.byref(ctx), args.pci.encode(), args.npu,
                           4, 4 * 1024 * 1024, False,
                           os.path.join(REPO_ROOT, "experiments", "output", "gates").encode())
    if rc != 0:
        raise RuntimeError(f"npu_nvme_init failed: {rc}")
    write_ptr = ctypes.c_void_p()
    read_ptr = ctypes.c_void_p()
    try:
        for ptr in (write_ptr, read_ptr):
            if acl_lib.aclrtMalloc(ctypes.byref(ptr), aligned, 0) != 0:
                raise RuntimeError("aclrtMalloc failed")
        host = ctypes.create_string_buffer(aligned)
        ctypes.memmove(host, frame, len(frame))
        if acl_lib.aclrtMemcpy(write_ptr, aligned, ctypes.byref(host), aligned, 1) != 0:
            raise RuntimeError("H2D frame copy failed")
        ptrs = (ctypes.c_void_p * 1)(write_ptr.value)
        offsets = (ctypes.c_uint64 * 1)(args.offset)
        sizes = (ctypes.c_size_t * 1)(aligned)
        if lib.npu_nvme_write_batch(ctx, ptrs, offsets, sizes, 1) != 0:
            raise RuntimeError("NPU-HBM frame write failed")
        read_ptrs = (ctypes.c_void_p * 1)(read_ptr.value)
        if lib.npu_nvme_read_batch(ctx, read_ptrs, offsets, sizes, 1) != 0:
            raise RuntimeError("NPU-HBM frame read failed")
        result = ctypes.create_string_buffer(aligned)
        if acl_lib.aclrtMemcpy(ctypes.byref(result), aligned, read_ptr,
                               aligned, 2) != 0:
            raise RuntimeError("D2H frame copy failed")
        actual = bytes(result.raw[:len(frame)])
        if actual != frame:
            raise AssertionError("NPU-HBM-SPDK frame bytes changed")
        decoded = oracle.ack(actual)
        recovered = oracle.recover(initial, [actual])
        if any(not np.array_equal(recovered["state"][name], current[name])
               for name in current):
            raise AssertionError("NPU-HBM-SPDK S2 recovery mismatch")
        print(json.dumps({"status": "pass", "frame_bytes": len(frame),
                          "aligned_bytes": aligned, "offset": args.offset,
                          "ack": decoded, "recovery_generation": recovered["generation"]},
                         indent=2, sort_keys=True))
    finally:
        if write_ptr.value:
            acl_lib.aclrtFree(write_ptr)
        if read_ptr.value:
            acl_lib.aclrtFree(read_ptr)
        lib.npu_nvme_cleanup(ctx)


if __name__ == "__main__":
    main()
