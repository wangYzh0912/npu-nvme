#!/usr/bin/env python3
"""Stage-4 bounded fault/backpressure/lifecycle gate for one raw namespace."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python")]

from c_bindings import NPUNVMEContext, NPUNVMERequest, acl_lib, lib  # noqa: E402
from ppt_evidence import command, environment_snapshot  # noqa: E402


ALIGN = 4096
OFFSET = 64 * 1024**3 + 512 * 1024**2
PAYLOAD = b"stage4-fault-lifecycle"
PAYLOAD += b"\0" * (ALIGN - len(PAYLOAD))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def arrays(pointer, offset=OFFSET, size=ALIGN):
    return ((ctypes.c_void_p * 1)(pointer),
            (ctypes.c_uint64 * 1)(offset),
            (ctypes.c_size_t * 1)(size))


def init(args, shm):
    os.environ["SPDK_SHM_ID"] = str(shm)
    ctx = ctypes.POINTER(NPUNVMEContext)()
    rc = lib.npu_nvme_init(ctypes.byref(ctx), args.pci.encode(), args.npu,
                           args.depth, ALIGN, True, str(args.output).encode())
    require(rc == 0 and bool(ctx), f"init failed: {rc}")
    return ctx


def host_write(ctx, payload=PAYLOAD):
    source = ctypes.create_string_buffer(payload, len(payload))
    ptrs, offsets, sizes = arrays(ctypes.addressof(source), size=len(payload))
    return lib.npu_nvme_write_batch_host(ctx, ptrs, offsets, sizes, 1)


def host_read(ctx, expected=PAYLOAD):
    target = ctypes.create_string_buffer(len(expected))
    ptrs, offsets, sizes = arrays(ctypes.addressof(target), size=len(expected))
    rc = lib.npu_nvme_read_batch_host(ctx, ptrs, offsets, sizes, 1)
    require(rc == 0 and target.raw == expected, "post-fault readback mismatch")


def superblock_sha256(ctx):
    target = ctypes.create_string_buffer(ALIGN)
    rc = lib.npu_nvme_sync_meta_io(ctx, 0, ALIGN, 1, ctypes.byref(target))
    require(rc == 0, "superblock read failed")
    return hashlib.sha256(target.raw).hexdigest()


def hbm_request(ctx, args, async_mode=True):
    source = ctypes.c_void_p()
    require(acl_lib.aclrtSetDevice(args.npu) == 0, "set device failed")
    require(acl_lib.aclrtMalloc(ctypes.byref(source), ALIGN, 0) == 0,
            "HBM allocation failed")
    try:
        host = ctypes.create_string_buffer(PAYLOAD, ALIGN)
        require(acl_lib.aclrtMemcpy(source, ALIGN, host, ALIGN, 1) == 0,
                "H2D failed")
        ptrs, offsets, sizes = arrays(source.value)
        request = ctypes.POINTER(NPUNVMERequest)()
        rc = lib.npu_nvme_submit_write_batch(
            ctx, ptrs, offsets, sizes, 1, ctypes.byref(request))
        if rc != 0:
            return rc
        rc = lib.npu_nvme_wait_request(request, 1000)
        lib.npu_nvme_release_request(request)
        return rc
    finally:
        acl_lib.aclrtFree(source)


def run_case(args, name, env_name=None, operation="host"):
    if env_name and env_name not in os.environ:
        os.environ[env_name] = "1"
    ctx = None
    try:
        ctx = init(args, args.shm_id + args.cases.index(name))
        if operation == "async":
            rc = hbm_request(ctx, args)
        elif operation == "timeout":
            started = time.monotonic()
            value = ctypes.create_string_buffer(PAYLOAD, ALIGN)
            ptrs, offsets, sizes = arrays(ctypes.addressof(value), size=ALIGN)
            rc = lib.npu_nvme_sync_meta_io(ctx, OFFSET + ALIGN, ALIGN, 0,
                                           ctypes.byref(value))
            elapsed = time.monotonic() - started
            require(rc != 0 and elapsed < args.timeout_bound,
                    f"timeout was not bounded: rc={rc} elapsed={elapsed:.3f}s")
        elif operation == "flush":
            require(host_write(ctx) == 0, "pre-flush write failed")
            rc = lib.npu_nvme_flush(ctx)
        elif operation == "metadata":
            value = ctypes.create_string_buffer(PAYLOAD, ALIGN)
            rc = lib.npu_nvme_sync_meta_io(
                ctx, OFFSET + ALIGN, ALIGN, 0,
                ctypes.byref(value))
        else:
            rc = host_write(ctx)
        require(rc != 0, f"fault {name} was not observed")
    finally:
        os.environ.pop(env_name, None) if env_name else None
        if ctx:
            lib.npu_nvme_cleanup(ctx)
    # A new context proves that failed requests did not poison the namespace.
    fresh = init(args, args.shm_id + 100 + args.cases.index(name))
    try:
        require(host_write(fresh) == 0, f"fresh write failed after {name}")
        host_read(fresh)
    finally:
        lib.npu_nvme_cleanup(fresh)
    return {"case": name, "status": "pass", "fault": env_name}


def run_backpressure(args):
    os.environ["NPU_NVME_TEST_NVME_SUBMIT_DELAY_MS"] = "100"
    ctx = init(args, args.shm_id + 200)
    source = ctypes.c_void_p()
    requests = []
    try:
        require(acl_lib.aclrtSetDevice(args.npu) == 0, "set device failed")
        require(acl_lib.aclrtMalloc(ctypes.byref(source), ALIGN, 0) == 0,
                "HBM allocation failed")
        host = ctypes.create_string_buffer(PAYLOAD, ALIGN)
        require(acl_lib.aclrtMemcpy(source, ALIGN, host, ALIGN, 1) == 0,
                "H2D failed")
        ptrs, offsets, sizes = arrays(source.value)
        busy = 0
        for _ in range(64):
            request = ctypes.POINTER(NPUNVMERequest)()
            rc = lib.npu_nvme_submit_write_batch(
                ctx, ptrs, offsets, sizes, 1, ctypes.byref(request))
            if rc != 0:
                busy = rc
                break
            requests.append(request)
        require(busy != 0, "request ring never returned explicit BUSY")
        for request in requests:
            lib.npu_nvme_wait_request(request, 0)
            lib.npu_nvme_release_request(request)
        stats = __import__("c_bindings").NPUNVMEStats()
        require(lib.npu_nvme_get_stats(ctx, ctypes.byref(stats)) == 0,
                "stats unavailable")
        require(int(stats.request_ring_peak) > 0, "ring peak was not recorded")
        return {"case": "request_ring_busy", "status": "pass",
                "busy_rc": busy, "submitted": len(requests),
                "stats": {name: int(getattr(stats, name))
                          for name, _ in stats._fields_}}
    finally:
        os.environ.pop("NPU_NVME_TEST_NVME_SUBMIT_DELAY_MS", None)
        acl_lib.aclrtFree(source)
        lib.npu_nvme_cleanup(ctx)


def run_crash_child(args):
    ctx = init(args, args.shm_id)
    if args.crash_child == "before_data_complete":
        os.environ["NPU_NVME_TEST_NVME_SUBMIT_DELAY_MS"] = "5000"
        source = ctypes.c_void_p()
        require(acl_lib.aclrtSetDevice(args.npu) == 0, "set device failed")
        require(acl_lib.aclrtMalloc(ctypes.byref(source), ALIGN, 0) == 0,
                "HBM allocation failed")
        host = ctypes.create_string_buffer(PAYLOAD, ALIGN)
        require(acl_lib.aclrtMemcpy(source, ALIGN, host, ALIGN, 1) == 0,
                "H2D failed")
        ptrs, offsets, sizes = arrays(source.value)
        request = ctypes.POINTER(NPUNVMERequest)()
        require(lib.npu_nvme_submit_write_batch(
            ctx, ptrs, offsets, sizes, 1, ctypes.byref(request)) == 0,
            "crash-window submit failed")
        os._exit(86)
    require(host_write(ctx) == 0, "crash-window data write failed")
    require(lib.npu_nvme_flush(ctx) == 0, "crash-window flush failed")
    os._exit(87)


def run_crash_windows(args):
    initial = init(args, args.shm_id + 300)
    try:
        committed = superblock_sha256(initial)
    finally:
        lib.npu_nvme_cleanup(initial)
    records = []
    for index, (mode, expected_rc) in enumerate((
            ("before_data_complete", 86), ("before_metadata_commit", 87))):
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--crash-child", mode, "--pci", args.pci,
                   "--npu", str(args.npu), "--depth", str(args.depth),
                   "--shm-id", str(args.shm_id + 310 + index),
                   "--output", str(args.output)]
        proc = subprocess.run(command, timeout=30)
        require(proc.returncode == expected_rc,
                f"{mode} child exit={proc.returncode}, expected={expected_rc}")
        fresh = init(args, args.shm_id + 320 + index)
        try:
            require(superblock_sha256(fresh) == committed,
                    f"{mode} published metadata before commit")
            require(host_write(fresh) == 0, f"fresh write failed after {mode}")
            host_read(fresh)
        finally:
            lib.npu_nvme_cleanup(fresh)
        records.append({"case": mode, "status": "pass",
                        "metadata_unchanged": True})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--checkpoint-slots", type=int, default=1)
    parser.add_argument("--shm-id", type=int, default=18300)
    parser.add_argument("--output", type=Path, default=Path("/tmp/stage4-fault"))
    parser.add_argument("--timeout-bound", type=float, default=0.20)
    parser.add_argument("--crash-child",
                        choices=("before_data_complete", "before_metadata_commit"))
    parser.add_argument("--cases", nargs="+", default=[
        "acl_copy", "event_query", "event_record", "nvme_submit",
        "nvme_completion", "metadata_write", "flush", "timeout",
    ])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.crash_child:
        run_crash_child(args)
        return
    (args.output / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    (args.output / "environment.json").write_text(
        json.dumps(environment_snapshot(
            pci=args.pci, npu=str(args.npu), repo_root=ROOT,
            npu_info=command(["npu-smi", "info"])), indent=2,
            sort_keys=True), encoding="utf-8")
    (args.output / "commit.json").write_text(json.dumps({
        "repo": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "branch": command(["git", "-C", str(ROOT), "branch", "--show-current"]),
        "status": command(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "spdk": command(["git", "-C", str(ROOT / "third_party" / "spdk"),
                         "rev-parse", "HEAD"]),
    }, indent=2, sort_keys=True), encoding="utf-8")
    mapping = {
        "acl_copy": ("NPU_NVME_TEST_FAIL_ACL_COPY", "async"),
        "event_query": ("NPU_NVME_TEST_FAIL_EVENT_QUERY", "async"),
        "event_record": ("NPU_NVME_TEST_FAIL_EVENT_RECORD", "async"),
        "nvme_submit": ("NPU_NVME_TEST_FAIL_NVME_SUBMIT", "host"),
        "nvme_completion": ("NPU_NVME_TEST_FAIL_NVME_COMPLETION", "host"),
        "metadata_write": ("NPU_NVME_TEST_FAIL_METADATA_WRITE", "metadata"),
        "flush": ("NPU_NVME_TEST_FAIL_FLUSH", "flush"),
        "timeout": ("NPU_NVME_TEST_META_DELAY_MS", "timeout"),
    }
    records = []
    try:
        for case in args.cases:
            env_name, operation = mapping[case]
            if case == "timeout":
                os.environ["NPU_NVME_IO_TIMEOUT_MS"] = "50"
                os.environ[env_name] = "500"
            records.append(run_case(args, case, env_name, operation))
            os.environ.pop("NPU_NVME_IO_TIMEOUT_MS", None)
        records.append(run_backpressure(args))
        records.extend(run_crash_windows(args))
    finally:
        os.environ.pop("NPU_NVME_IO_TIMEOUT_MS", None)
        os.environ.pop("NPU_NVME_TEST_META_DELAY_MS", None)
    result = {"status": "pass", "cases": records}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
