#!/usr/bin/env python3
"""P1 fair path comparison.

Each sample transfers exactly ``--total-bytes``.  ``--sizes`` are request
block sizes and ``--depths`` are queue depths, so QD never changes the amount
of data used for a comparison.  The filesystem phase uses fio; the SPDK
phase uses the host-buffer C API and records the same durable boundary.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import mmap
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from ppt_evidence import EvidenceBundle, command, environment_snapshot, stats
from c_bindings import NPUNVMEContext, lib

SIZES = (4 * 1024, 64 * 1024, 1024 * 1024, 4 * 1024 * 1024,
         256 * 1024 * 1024)
DEPTHS = (1, 4)
TOTAL_BYTES = 1024 * 1024 * 1024
SAFE_OFFSET = 64 * 1024**3


def _buffer(total, fill=0x5A):
    value = mmap.mmap(-1, total, access=mmap.ACCESS_WRITE)
    block = bytes([fill]) * min(total, 1024 * 1024)
    for offset in range(0, total, len(block)):
        value[offset:offset + min(len(block), total - offset)] = block[:min(len(block), total - offset)]
    address = ctypes.addressof(ctypes.c_char.from_buffer(value))
    return value, address


def _arrays(address, total, block, offset):
    count = (total + block - 1) // block
    ptrs = (ctypes.c_void_p * count)()
    offsets = (ctypes.c_uint64 * count)()
    sizes = (ctypes.c_size_t * count)()
    for index in range(count):
        size = min(block, total - index * block)
        ptrs[index] = address + index * block
        offsets[index] = offset + index * block
        sizes[index] = size
    return ptrs, offsets, sizes, count


def _cpu_sample(before):
    after = resource.getrusage(resource.RUSAGE_SELF)
    return ((after.ru_utime - before.ru_utime) +
            (after.ru_stime - before.ru_stime))


def _zero_digest(total):
    digest = hashlib.sha256()
    block = bytes(min(total, 1024 * 1024))
    for offset in range(0, total, len(block)):
        digest.update(block[:min(len(block), total - offset)])
    return digest.hexdigest()


def run_spdk(args, operation, block, depth):
    command_block = min(block, args.spdk_io_unit)
    config = {
        "experiment": "P1", "path": "spdk", "operation": operation,
        "total_bytes": args.total_bytes, "block_size": block,
        "logical_block_size": block, "spdk_command_size": command_block,
        "queue_depth": depth, "warmups": args.warmups,
        "formal_samples": args.samples, "persist_boundary": "nvme_flush",
        "target_pci": args.pci, "raw_offset": args.offset,
        "cross_disk_calibration": True,
    }
    bundle = EvidenceBundle("P1", config, repo_root=ROOT,
        environment=environment_snapshot(
            pci=args.pci, npu=str(args.npu), repo_root=ROOT,
            npu_info=command(["npu-smi", "info"])))
    ctx = ctypes.POINTER(NPUNVMEContext)()
    mmap_obj, address = _buffer(args.total_bytes, 0x3C if operation == "write" else 0)
    expected_read_digest = _zero_digest(args.total_bytes) \
        if operation == "read" else None
    timings, cpu_times, failures = [], [], 0
    try:
        rc = lib.npu_nvme_init(ctypes.byref(ctx), args.pci.encode(), args.npu,
                               depth, command_block, False,
                               str(bundle.raw_dir).encode())
        if rc != 0:
            raise RuntimeError(f"npu_nvme_init rc={rc}")
        for index in range(args.warmups + args.samples):
            sample_offset = args.offset + (index + 1) * args.total_bytes
            ptrs, offsets, sizes, count = _arrays(address, args.total_bytes,
                                                    command_block,
                                                    sample_offset)
            # Reads are preconditioned outside the timed interval.
            if operation == "read":
                if lib.npu_nvme_write_batch_host(ctx, ptrs, offsets, sizes, count) != 0:
                    raise RuntimeError("read precondition write failed")
                if lib.npu_nvme_flush(ctx) != 0:
                    raise RuntimeError("read precondition flush failed")
            before_cpu = resource.getrusage(resource.RUSAGE_SELF)
            started = time.perf_counter_ns()
            if operation == "write":
                rc = lib.npu_nvme_write_batch_host(ctx, ptrs, offsets, sizes, count)
                if rc == 0:
                    rc = lib.npu_nvme_flush(ctx)
            else:
                rc = lib.npu_nvme_read_batch_host(ctx, ptrs, offsets, sizes, count)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            cpu_s = _cpu_sample(before_cpu)
            if rc != 0:
                failures += 1
                bundle.add_failure({"sample": index, "operation": operation,
                                    "rc": rc, "block_size": block,
                                    "queue_depth": depth})
                continue
            if index < args.warmups:
                continue
            digest = hashlib.sha256(mmap_obj).hexdigest()
            if expected_read_digest is not None and digest != expected_read_digest:
                failures += 1
                bundle.add_failure({"sample": index, "operation": operation,
                                    "reason": "readback digest mismatch",
                                    "expected": expected_read_digest,
                                    "actual": digest})
                continue
            row = {"status": "pass", "operation": operation,
                   "block_size": block, "queue_depth": depth,
                   "sample": index - args.warmups, "bytes": args.total_bytes,
                   "latency_ms": elapsed, "cpu_seconds": cpu_s,
                   "sha256": digest,
                   "events": [{"name": "host_batch_submit"},
                              {"name": "nvme_flush" if operation == "write"
                               else "host_batch_complete"}]}
            bundle.add_sample(row, events=row["events"])
            timings.append(elapsed)
            cpu_times.append(cpu_s)
    except BaseException as error:
        bundle.add_failure({"stage": "initialization_or_run", "error": repr(error)})
    finally:
        if ctx:
            lib.npu_nvme_cleanup(ctx)
        mmap_obj.close()
    latency = stats(timings)
    result = bundle.finalize(metrics={
        "model": "synthetic_1GiB", "mode": "spdk_host",
        "operation": operation,
        "logical_bytes": args.total_bytes, "physical_bytes": args.total_bytes * len(timings),
        "chunk_size": block, "spdk_command_size": command_block,
        "pipeline_depth": depth, "slot_count": depth,
        "latency_mean": latency.get("mean"), "latency_p50": latency.get("median"),
        "latency_p95": latency.get("p95"),
        "throughput": args.total_bytes / (latency["mean"] / 1000.0)
        if latency.get("mean") else None,
        "nvme_bytes": args.total_bytes * len(timings),
        "cpu_seconds": stats(cpu_times), "sha256_samples": len(timings),
        "failure_count": failures,
    }, status="pass" if len(timings) == args.samples and not failures else "fail")
    print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                      "path": "spdk", "operation": operation,
                      "block_size": block, "queue_depth": depth}, sort_keys=True), flush=True)


def run_fs(args, mode, operation, block, depth):
    label = f"{mode}_{operation}_bs{block}_qd{depth}"
    target = args.fs_root / f"{label}.bin"
    config = {"experiment": "P1", "path": mode, "operation": operation,
              "total_bytes": args.total_bytes, "block_size": block,
              "queue_depth": depth, "warmups": args.warmups,
              "formal_samples": args.samples,
              "persist_boundary": "fio end_fsync (one fsync per sample)",
              "filesystem": "XFS", "filesystem_pci": "0000:84:00.0",
              "target_path": str(target), "cross_disk_calibration": True}
    bundle = EvidenceBundle("P1", config, repo_root=ROOT,
        environment=environment_snapshot(pci="0000:84:00.0", npu=str(args.npu),
                                          repo_root=ROOT,
                                          npu_info=command(["npu-smi", "info"])))
    timings, cpu_times = [], []
    target.parent.mkdir(parents=True, exist_ok=True)
    if operation == "read":
        precondition = ["fio", f"--name={label}_precondition",
                        f"--filename={target}", "--rw=write",
                        "--ioengine=io_uring", f"--iodepth={depth}",
                        "--numjobs=1", f"--bs={block}",
                        f"--size={args.total_bytes}",
                        "--direct=0" if mode == "buffered" else "--direct=1",
                        "--end_fsync=1", "--output-format=json"]
        completed = subprocess.run(precondition, capture_output=True, text=True,
                                   check=False, timeout=args.timeout)
        (bundle.raw_dir / "fio_read_precondition.json").write_text(
            completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            bundle.add_failure({"stage": "read_precondition",
                                "returncode": completed.returncode,
                                "stderr": completed.stderr})
    for index in range(args.warmups + args.samples):
        argv = ["fio", f"--name={label}", f"--filename={target}",
                f"--rw={operation}", "--ioengine=io_uring",
                f"--iodepth={depth}", "--numjobs=1", f"--bs={block}",
                f"--size={args.total_bytes}", "--direct=0" if mode == "buffered" else "--direct=1",
                "--end_fsync=1" if operation == "write" else "--invalidate=1",
                "--output-format=json"]
        started = time.perf_counter_ns()
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        completed = subprocess.run(argv, capture_output=True, text=True,
                                   check=False, timeout=args.timeout)
        wall_ms = (time.perf_counter_ns() - started) / 1e6
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu = ((after.ru_utime - before.ru_utime) +
                     (after.ru_stime - before.ru_stime))
        (bundle.raw_dir / f"fio_{index:04d}.json").write_text(
            completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            bundle.add_failure({"sample": index, "argv": argv,
                                "returncode": completed.returncode,
                                "stderr": completed.stderr})
            continue
        fio_result = json.loads(completed.stdout)
        job = fio_result["jobs"][0]
        elapsed = float(job.get("job_runtime", wall_ms))
        fio_cpu = elapsed / 1000.0 * (
            float(job.get("usr_cpu", 0)) + float(job.get("sys_cpu", 0))) / 100.0
        if index < args.warmups:
            continue
        bundle.add_sample({"status": "pass", "operation": operation,
                           "block_size": block, "queue_depth": depth,
                           "sample": index - args.warmups,
                           "bytes": args.total_bytes, "latency_ms": elapsed,
                           "wall_ms": wall_ms, "cpu_seconds": fio_cpu,
                           "events": [{"name": "fio_start_end"}]})
        timings.append(elapsed)
        cpu_times.append(fio_cpu)
    latency = stats(timings)
    result = bundle.finalize(metrics={
        "model": "synthetic_1GiB", "mode": mode,
        "operation": operation,
        "logical_bytes": args.total_bytes,
        "physical_bytes": args.total_bytes * len(timings),
        "chunk_size": block, "pipeline_depth": depth, "slot_count": depth,
        "latency_mean": latency.get("mean"), "latency_p50": latency.get("median"),
        "latency_p95": latency.get("p95"),
        "throughput": args.total_bytes / (latency["mean"] / 1000.0)
        if latency.get("mean") else None,
        "nvme_bytes": args.total_bytes * len(timings), "cpu_seconds": stats(cpu_times),
        "fio_formal_samples": len(timings),
        "fio_command_template": argv,
    }, status="pass" if len(timings) == args.samples and not bundle.failures else "fail")
    print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                      "path": mode, "operation": operation,
                      "block_size": block, "queue_depth": depth}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("buffered", "odirect", "spdk", "all"), default="all")
    parser.add_argument("--operations", nargs="+", choices=("write", "read"), default=("write", "read"))
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--depths", nargs="+", type=int, default=DEPTHS)
    parser.add_argument("--total-bytes", type=int, default=TOTAL_BYTES)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--spdk-io-unit", type=int, default=4 * 1024 * 1024,
                        help="maximum physical NVMe command size; larger logical blocks are segmented")
    parser.add_argument("--offset", type=int, default=SAFE_OFFSET)
    parser.add_argument("--fs-root", type=Path, default=Path("/models/npu_nvme_exp/ppt-evidence-20260829"))
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if args.total_bytes <= 0 or args.samples < 30 or args.warmups < 1:
        raise SystemExit("total-bytes > 0, warmups >= 1 and samples >= 30 are required")
    paths = ("buffered", "odirect", "spdk") if args.path == "all" else (args.path,)
    for path in paths:
        for operation in args.operations:
            for block in args.sizes:
                for depth in args.depths:
                    if path == "spdk":
                        run_spdk(args, operation, block, depth)
                    else:
                        run_fs(args, path, operation, block, depth)


if __name__ == "__main__":
    main()
