#!/usr/bin/env python3
"""I5 NPU-HBM-SPDK matrix: non-aligned tail and multi-segment frames."""

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
from c_bindings import NPUNVMEContext, acl_lib, lib  # noqa: E402
from s2_delta import S2DeltaOracle  # noqa: E402


def aligned(value):
    return (value + 4095) // 4096 * 4096


def make_frame(length, step):
    initial = {"p": np.zeros(length, dtype=np.float32)}
    current = {"p": initial["p"].copy()}
    current["p"][step % length] = np.float32(step + 0.25)
    oracle = S2DeltaOracle(initial, block_size=64, small_threshold=0)
    oracle.set_current(current)
    return oracle, oracle.observe(step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=5)
    parser.add_argument("--shm-id", type=int, default=1830)
    parser.add_argument("--offset", type=int, default=64 * 1024**3 + 64 * 1024**2)
    args = parser.parse_args()
    os.environ.setdefault("SPDK_SHM_ID", str(args.shm_id))
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
                           str(REPO_ROOT / "experiments/output/gates").encode())
    if rc != 0:
        raise RuntimeError(f"npu_nvme_init failed: {rc}")
    cases = [("tail_non_aligned", 100), ("medium_non_aligned", 1025),
             ("large_non_aligned", 16385)]
    results = []
    try:
        for index, (name, elements) in enumerate(cases):
            oracle, frame = make_frame(elements, index + 1)
            padded = frame + bytes(aligned(len(frame)) - len(frame))
            chunks = [padded] if name != "large_non_aligned" else [
                piece for piece in (padded[:4096], padded[4096:8192],
                                    padded[8192:]) if piece]
            chunks = [chunk + bytes(aligned(len(chunk)) - len(chunk))
                      for chunk in chunks]
            write_ptrs, read_ptrs, hosts = [], [], []
            try:
                for chunk in chunks:
                    wp, rp = ctypes.c_void_p(), ctypes.c_void_p()
                    # Keep the transfer size exact/aligned, but use a 1 MiB
                    # allocation floor.  Some ACL/CANN combinations reject
                    # tiny device allocations even though the SPDK API can
                    # transfer a 4 KiB-aligned tail.
                    alloc_len = max(len(chunk), 1024 * 1024)
                    write_rc = acl_lib.aclrtMalloc(ctypes.byref(wp), alloc_len, 0)
                    if write_rc != 0:
                        raise RuntimeError(f"aclrtMalloc write failed: {write_rc}")
                    read_rc = acl_lib.aclrtMalloc(ctypes.byref(rp), alloc_len, 0)
                    if read_rc != 0:
                        raise RuntimeError(f"aclrtMalloc read failed: {read_rc}")
                    host = ctypes.create_string_buffer(alloc_len)
                    ctypes.memmove(host, chunk, len(chunk))
                    if acl_lib.aclrtMemcpy(wp, alloc_len, ctypes.byref(host),
                                           alloc_len, 1) != 0:
                        raise RuntimeError("H2D failed")
                    write_ptrs.append(wp); read_ptrs.append(rp); hosts.append(host)
                ptr_array = (ctypes.c_void_p * len(chunks))(*[p.value for p in write_ptrs])
                read_array = (ctypes.c_void_p * len(chunks))(*[p.value for p in read_ptrs])
                offsets = (ctypes.c_uint64 * len(chunks))(*[
                    args.offset + index * 32 * 1024 * 1024 + part * 4 * 1024 * 1024
                    for part in range(len(chunks))])
                sizes = (ctypes.c_size_t * len(chunks))(*[len(c) for c in chunks])
                if lib.npu_nvme_write_batch(ctx, ptr_array, offsets, sizes,
                                            len(chunks)) != 0:
                    raise RuntimeError("write_batch failed")
                if lib.npu_nvme_read_batch(ctx, read_array, offsets, sizes,
                                           len(chunks)) != 0:
                    raise RuntimeError("read_batch failed")
                actual_parts = []
                for rp, chunk in zip(read_ptrs, chunks):
                    out = ctypes.create_string_buffer(len(chunk))
                    if acl_lib.aclrtMemcpy(ctypes.byref(out), len(chunk), rp,
                                           len(chunk), 2) != 0:
                        raise RuntimeError("D2H failed")
                    actual_parts.append(bytes(out.raw))
                actual = b"".join(actual_parts)[:len(frame)]
                if actual != frame:
                    raise AssertionError(f"{name}: frame mismatch")
                oracle.ack(actual)
                recovered = oracle.recover({"p": np.zeros(elements, dtype=np.float32)}, [actual])
                if not np.array_equal(recovered["state"]["p"],
                                      oracle.current["p"]):
                    raise AssertionError(f"{name}: recovery mismatch")
                results.append({"name": name, "frame_bytes": len(frame),
                                "aligned_bytes": sum(len(c) for c in chunks),
                                "segments": len(chunks), "status": "pass"})
            finally:
                for ptr in write_ptrs + read_ptrs:
                    if ptr.value:
                        acl_lib.aclrtFree(ptr)
        print(json.dumps({"status": "pass", "pci": args.pci, "npu": args.npu,
                          "cases": results}, indent=2, sort_keys=True))
    finally:
        lib.npu_nvme_cleanup(ctx)


if __name__ == "__main__":
    main()
