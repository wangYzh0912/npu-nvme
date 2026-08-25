#!/usr/bin/env python3
"""E1 memory-layer microbenchmarks on Ascend 910B.

Measures pageable Host memcpy and ACL H2D/D2H/D2D for 4 KiB--16 MiB.  The
script initializes the existing NPU-NVMe context only to establish the ACL
runtime; it performs no NVMe I/O and writes no checkpoint metadata.
"""

import argparse
import ctypes
import hashlib
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "benchmarks"))
sys.path.insert(0, str(REPO_ROOT / "python"))

from io_matrix import (CHUNK_SIZE, ResultWriter, check_npu_free,
                       environment_snapshot, stats)


ACL_H2D = 1
ACL_D2H = 2
ACL_D2D = 3
SIZES = (4 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024,
         4 * 1024 * 1024, 16 * 1024 * 1024)


def configure_acl(acl_lib):
    acl_lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_size_t, ctypes.c_int]
    acl_lib.aclrtMalloc.restype = ctypes.c_int
    acl_lib.aclrtFree.argtypes = [ctypes.c_void_p]
    acl_lib.aclrtFree.restype = ctypes.c_int
    acl_lib.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.c_int]
    acl_lib.aclrtMemcpy.restype = ctypes.c_int


def alloc_device(acl_lib, size):
    pointer = ctypes.c_void_p()
    rc = acl_lib.aclrtMalloc(ctypes.byref(pointer), size, 0)
    if rc != 0 or not pointer.value:
        raise RuntimeError(f"aclrtMalloc({size}) failed: {rc}")
    return pointer.value


def free_device(acl_lib, pointer):
    if pointer:
        acl_lib.aclrtFree(ctypes.c_void_p(pointer))


def acl_copy(acl_lib, dst, dst_max, src, size, kind):
    rc = acl_lib.aclrtMemcpy(ctypes.c_void_p(dst), size,
                             ctypes.c_void_p(src), size, kind)
    if rc != 0:
        raise RuntimeError(f"aclrtMemcpy(kind={kind}, size={size}) failed: {rc}")


def sample(writer, path, size, index, warmup, acl_lib, host_src, host_dst,
           device_a, device_b):
    seed = bytes((position * 13 + 7) % 256 for position in range(size))
    ctypes.memmove(ctypes.addressof(host_src), seed, size)
    events = []
    start = time.monotonic_ns()
    events.append({"name": "transfer_start", "monotonic_ns": start})

    if path == "host_memcpy":
        enter = time.monotonic_ns()
        events.append({"name": "memcpy_enter", "monotonic_ns": enter})
        ctypes.memmove(ctypes.addressof(host_dst), ctypes.addressof(host_src), size)
        leave = time.monotonic_ns()
        events.append({"name": "memcpy_return", "monotonic_ns": leave})
        actual = bytes(host_dst.raw[:size])
    elif path == "acl_h2d":
        enter = time.monotonic_ns()
        events.append({"name": "acl_h2d_enter", "monotonic_ns": enter})
        acl_copy(acl_lib, device_a, size, ctypes.addressof(host_src), size, ACL_H2D)
        leave = time.monotonic_ns()
        events.append({"name": "acl_h2d_return", "monotonic_ns": leave})
        acl_copy(acl_lib, ctypes.addressof(host_dst), size, device_a, size, ACL_D2H)
        actual = bytes(host_dst.raw[:size])
    elif path == "acl_d2h":
        acl_copy(acl_lib, device_a, size, ctypes.addressof(host_src), size, ACL_H2D)
        enter = time.monotonic_ns()
        events.append({"name": "acl_d2h_enter", "monotonic_ns": enter})
        acl_copy(acl_lib, ctypes.addressof(host_dst), size, device_a, size, ACL_D2H)
        leave = time.monotonic_ns()
        events.append({"name": "acl_d2h_return", "monotonic_ns": leave})
        actual = bytes(host_dst.raw[:size])
    elif path == "acl_d2d":
        acl_copy(acl_lib, device_a, size, ctypes.addressof(host_src), size, ACL_H2D)
        enter = time.monotonic_ns()
        events.append({"name": "acl_d2d_enter", "monotonic_ns": enter})
        acl_copy(acl_lib, device_b, size, device_a, size, ACL_D2D)
        leave = time.monotonic_ns()
        events.append({"name": "acl_d2d_return", "monotonic_ns": leave})
        acl_copy(acl_lib, ctypes.addressof(host_dst), size, device_b, size, ACL_D2H)
        actual = bytes(host_dst.raw[:size])
    else:
        raise ValueError(path)

    if actual != seed:
        raise AssertionError(f"{path} content mismatch at {size} bytes")
    end = time.monotonic_ns()
    events.append({"name": "transfer_end", "monotonic_ns": end})
    return {"run_id": writer.run_id, "checkpoint_id": f"memory_{index:04d}",
            "request_id": f"{writer.run_id}/request_{index:04d}",
            "warmup": warmup, "path": path, "bytes": size,
            "sha256": hashlib.sha256(actual).hexdigest(), "status": "pass",
            "events": events,
            "timeline_us": {"transfer": (leave - enter) / 1000,
                             "end_to_end": (end - start) / 1000}}


def run_one(args, path, size, npu_info):
    from direct_checkpoint import DirectCheckpoint
    from c_bindings import acl_lib

    writer = ResultWriter("E1_MEM", args)
    writer.config.update({"path": path, "size": size, "npu": args.npu,
                          "warmups": args.warmups,
                          "repetitions": args.repetitions})
    writer.write_json("config.json", writer.config)
    writer.write_json("environment.json", environment_snapshot(args, npu_info))
    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu, pipeline_depth=4,
        requested_chunk_size=CHUNK_SIZE, rank_id=0, world_size=1,
        keep_last_n=3, slot_size_gb=10, spdk_shm_id=args.shm_id,
        profiling_dir=str(writer.run_dir / "profiling"))
    configure_acl(acl_lib)
    device_a = alloc_device(acl_lib, size)
    device_b = alloc_device(acl_lib, size)
    host_src = ctypes.create_string_buffer(size)
    host_dst = ctypes.create_string_buffer(size)
    try:
        for index in range(args.warmups + args.repetitions):
            try:
                value = sample(writer, path, size, index, index < args.warmups,
                               acl_lib, host_src, host_dst, device_a, device_b)
                if index >= args.warmups:
                    writer.add_sample(value)
            except BaseException as error:
                writer.add_failure({"index": index, "error": repr(error)})
                if index < args.warmups:
                    raise
    finally:
        free_device(acl_lib, device_a)
        free_device(acl_lib, device_b)
        ckpt.cleanup()
    values = [item["timeline_us"]["transfer"] for item in writer.samples]
    result = writer.finalize({"transfer_us": stats(values),
                              "effective_mib_per_s": stats([
                                  size / (v / 1e6) / (1024 ** 2) for v in values])},
                             status="pass" if len(values) == args.repetitions and not writer.failed else "fail")
    print(f"[E1/MEM] {path} size={size} status={result['status']} "
          f"mean_us={result['summary']['transfer_us']['mean']:.2f}", flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=83)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--paths", nargs="+",
                        default=["host_memcpy", "acl_h2d", "acl_d2h", "acl_d2d"],
                        choices=["host_memcpy", "acl_h2d", "acl_d2h", "acl_d2d"])
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    args = parser.parse_args()
    if args.repetitions < 10:
        raise ValueError("formal repetitions must be at least 10")
    npu_info = check_npu_free(args.npu)
    for path in args.paths:
        for size in args.sizes:
            run_one(args, path, size, npu_info)


if __name__ == "__main__":
    main()
