#!/usr/bin/env python3
"""Inject one NVMe submission failure and prove the Reactor remains usable."""
import argparse
import ctypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from c_bindings import NPUNVMEContext, lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    args = parser.parse_args()
    os.environ["NPU_NVME_TEST_FAIL_WRITE_AT"] = "1"
    ctx = ctypes.POINTER(NPUNVMEContext)()
    rc = lib.npu_nvme_init(ctypes.byref(ctx), args.pci.encode(), args.npu,
                           1, 4096, False,
                           str(ROOT / "experiments/output/gates").encode())
    if rc:
        raise RuntimeError(f"init failed: {rc}")
    try:
        payload = b"F" * 4096
        source = ctypes.create_string_buffer(payload)
        ptrs = (ctypes.c_void_p * 1)(ctypes.addressof(source))
        offsets = (ctypes.c_uint64 * 1)(64 * 1024**3)
        sizes = (ctypes.c_size_t * 1)(4096)
        if lib.npu_nvme_write_batch_host(ctx, ptrs, offsets, sizes, 1) == 0:
            raise AssertionError("injected NVMe failure was not observed")
        os.environ.pop("NPU_NVME_TEST_FAIL_WRITE_AT", None)
        if lib.npu_nvme_write_batch_host(ctx, ptrs, offsets, sizes, 1) != 0:
            raise AssertionError("Reactor did not recover after injected failure")
        target = ctypes.create_string_buffer(4096)
        reads = (ctypes.c_void_p * 1)(ctypes.addressof(target))
        if (lib.npu_nvme_read_batch_host(ctx, reads, offsets, sizes, 1) != 0
                or target.raw != payload):
            raise AssertionError("post-failure roundtrip mismatch")
        print("fault injection PASS")
    finally:
        lib.npu_nvme_cleanup(ctx)


if __name__ == "__main__":
    main()
