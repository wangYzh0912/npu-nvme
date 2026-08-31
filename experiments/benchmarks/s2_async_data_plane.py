#!/usr/bin/env python3
"""Stage-2 FULL data-plane screening and formal validation.

This runner intentionally has no MindSpore dependency.  It exercises the
real HBM -> DMA -> NVMe path, then reads into an independent HBM allocation
and compares the bytes in a fresh host buffer.  One process owns one config,
so context teardown is part of every sample's lifecycle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from c_bindings import NPUNVMEContext, NPUNVMEStats, acl_lib, lib  # noqa: E402
from ppt_evidence import command, environment_snapshot  # noqa: E402


def chunks(total, chunk):
    return [(offset, min(chunk, total - offset))
            for offset in range(0, total, chunk)]


def digest_hbm(pointer, size, npu):
    host = ctypes_buffer(size)
    check_acl(acl_lib.aclrtMemcpy(host, size, pointer, size, 2), "D2H digest")
    return hashlib.sha256(host.raw[:size]).hexdigest(), host


def ctypes_buffer(size):
    import ctypes
    return ctypes.create_string_buffer(size)


def check_acl(code, operation):
    if code != 0:
        raise RuntimeError(f"{operation} failed: {code}")


def overlap_from_profile(path):
    rows = []
    if not path.exists():
        return 0.0, 0
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            def number(name):
                try:
                    return float(row.get(name, 0) or 0)
                except (TypeError, ValueError):
                    return 0.0
            rows.append((number("ts_dma_submit_us"), number("ts_dma_done_us"),
                         number("ts_nvme_submit_us"), number("ts_nvme_done_us")))
    valid = [item for item in rows if item[1] > item[0] and item[3] > item[2]]
    if not valid:
        return 0.0, len(rows)
    def union_duration(intervals):
        total = 0.0
        current_end = None
        for start, end in sorted(intervals):
            if current_end is None or start > current_end:
                total += end - start
                current_end = end
            elif end > current_end:
                total += end - current_end
                current_end = end
        return total
    dma = union_duration((start, end) for start, end, _a, _b in valid)
    nvme = union_duration((start, end) for _a, _b, start, end in valid)
    begin = min(item[0] for item in valid)
    end = max(item[3] for item in valid)
    makespan = max(end - begin, 1e-9)
    return max(0.0, min(1.0, (dma + nvme - makespan) /
                        max(min(dma, nvme), 1e-9))), len(rows)


def run_one(args):
    import ctypes

    os.environ["SPDK_SHM_ID"] = str(args.shm_id)
    check_acl(acl_lib.aclrtSetDevice(args.npu), "aclrtSetDevice")
    source = ctypes.c_void_p()
    target = ctypes.c_void_p()
    size = int(args.payload)
    aligned_chunk = int(args.chunk)
    if size <= 0 or aligned_chunk <= 0 or aligned_chunk % 4096:
        raise ValueError("payload and chunk must be positive; chunk must be 4 KiB aligned")
    check_acl(acl_lib.aclrtMalloc(ctypes.byref(source), size, 0), "source aclrtMalloc")
    check_acl(acl_lib.aclrtMalloc(ctypes.byref(target), size, 0), "target aclrtMalloc")
    context = ctypes.POINTER(NPUNVMEContext)()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    payload_host = ctypes_buffer(size)
    pattern = np.arange(size, dtype=np.uint8)
    pattern[:] = (pattern * 131 + args.seed * 17) & 0xFF
    ctypes.memmove(payload_host, pattern.ctypes.data, size)
    check_acl(acl_lib.aclrtMemcpy(source, size, payload_host, size, 1), "H2D source")
    expected = hashlib.sha256(payload_host.raw[:size]).hexdigest()
    base = int(args.offset)
    descriptors = chunks(size, aligned_chunk)
    ptrs = (ctypes.c_void_p * len(descriptors))(*(
        source.value + inner for inner, _part in descriptors))
    offsets = (ctypes.c_uint64 * len(descriptors))(*(
        base + inner for inner, _part in descriptors))
    sizes = (ctypes.c_size_t * len(descriptors))(*(
        part for _inner, part in descriptors))
    try:
        rc = lib.npu_nvme_init(ctypes.byref(context), args.pci.encode(), args.npu,
                               args.depth, aligned_chunk, True,
                               str(run_dir / "raw").encode())
        if rc != 0:
            raise RuntimeError(f"npu_nvme_init failed: {rc}")
        started = time.perf_counter_ns()
        handle = ctypes.POINTER(__import__("c_bindings").NPUNVMERequest)()
        if args.mode == "async":
            rc = lib.npu_nvme_submit_write_batch(
                context, ptrs, offsets, sizes, len(descriptors), ctypes.byref(handle))
            if rc != 0:
                raise RuntimeError(f"async submit failed: {rc}")
            poll_count = 0
            while True:
                done = ctypes.c_int()
                rc = lib.npu_nvme_poll_request(handle, ctypes.byref(done))
                poll_count += 1
                if rc != 0 and done.value:
                    raise RuntimeError(f"async request failed: {rc}")
                if done.value:
                    break
                time.sleep(0.0005)
            lib.npu_nvme_release_request(handle)
        elif args.mode == "serial":
            for index in range(len(descriptors)):
                one_ptr = (ctypes.c_void_p * 1)(ptrs[index])
                one_off = (ctypes.c_uint64 * 1)(offsets[index])
                one_size = (ctypes.c_size_t * 1)(sizes[index])
                rc = lib.npu_nvme_write_batch(context, one_ptr, one_off, one_size, 1)
                if rc != 0:
                    break
            poll_count = 0
            if rc != 0:
                raise RuntimeError(f"serial write failed: {rc}")
        else:
            # queue is the synchronous-D2H path driven by the Reactor FSM.
            rc = lib.npu_nvme_write_batch(context, ptrs, offsets, sizes, len(descriptors))
            poll_count = 0
            if rc != 0:
                raise RuntimeError(f"queue write failed: {rc}")
        check_acl(lib.npu_nvme_flush(context), "flush")
        elapsed = (time.perf_counter_ns() - started) / 1e9

        read_ptrs = (ctypes.c_void_p * len(descriptors))(*(
            target.value + inner for inner, _part in descriptors))
        check_acl(lib.npu_nvme_read_batch(context, read_ptrs, offsets, sizes,
                                          len(descriptors)), "HBM readback")
        actual, readback = digest_hbm(target, size, args.npu)
        if actual != expected:
            raise AssertionError(f"readback checksum mismatch: {actual} != {expected}")
        stats = NPUNVMEStats()
        check_acl(lib.npu_nvme_get_stats(context, ctypes.byref(stats)), "stats")
        counters = {name: int(getattr(stats, name)) for name, _ in stats._fields_}
        overlap, profile_rows = overlap_from_profile(run_dir / "raw" / "time_write.csv")
        result = {
            "status": "pass", "mode": args.mode, "payload_bytes": size,
            "chunk_bytes": aligned_chunk, "depth": args.depth,
            "expected_sha256": expected, "readback_sha256": actual,
            "elapsed_seconds": elapsed, "poll_count": poll_count,
            "nvme_outstanding_peak": counters["nvme_outstanding_peak"],
            "overlap_rate": overlap, "profile_rows": profile_rows,
            "stats": counters,
            "events": [{"name": "submit", "monotonic_ns": started},
                       {"name": "flush_complete", "monotonic_ns": time.monotonic_ns()},
                       {"name": "readback_verified", "monotonic_ns": time.monotonic_ns()}],
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        if context:
            lib.npu_nvme_cleanup(context)
        acl_lib.aclrtFree(target)
        acl_lib.aclrtFree(source)


def child_command(args, run_dir, payload, chunk, depth, mode):
    return [sys.executable, str(Path(__file__).resolve()), "--child",
            "--run-dir", str(run_dir), "--payload", str(payload),
            "--chunk", str(chunk), "--depth", str(depth), "--mode", mode,
            "--npu", str(args.npu), "--pci", args.pci,
            "--offset", str(args.offset), "--seed", str(args.seed),
            "--shm-id", str(args.shm_id)]


def orchestrate(args):
    root = Path(args.output_root or ROOT / "results" / "stage2-async")
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({**vars(args), "phase": "stage2_async_data_plane",
                    "state": "FULL-data-plane-only"}, indent=2,
                   sort_keys=True, default=str), encoding="utf-8")
    (root / "environment.json").write_text(
        json.dumps(environment_snapshot(
            pci=args.pci, npu=str(args.npu), repo_root=ROOT,
            npu_info=command(["npu-smi", "info"])), indent=2,
            sort_keys=True), encoding="utf-8")
    (root / "commit.json").write_text(json.dumps({
        "repo": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "branch": command(["git", "-C", str(ROOT), "branch", "--show-current"]),
        "status": command(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "spdk": command(["git", "-C", str(ROOT / "third_party" / "spdk"),
                         "rev-parse", "HEAD"]),
    }, indent=2, sort_keys=True), encoding="utf-8")
    payloads = args.payloads
    chunksizes = args.chunks
    depths = args.depths
    modes = args.modes
    configs = [(p, c, d, m) for p in payloads for c in chunksizes
               for d in depths for m in modes]
    if args.screening:
        samples = args.screening_samples
        warmups = 0
    else:
        samples = args.samples
        warmups = args.warmups
    records = []
    failures = []
    for index, (payload, chunk, depth, mode) in enumerate(configs):
        for iteration in range(warmups + samples):
            warmup = iteration < warmups
            sample = iteration - warmups
            label = f"w{iteration:02d}" if warmup else f"s{sample:02d}"
            run_dir = root / f"p{payload}_c{chunk}_d{depth}_{mode}_{label}"
            proc = subprocess.run(child_command(args, run_dir, payload, chunk, depth, mode),
                                  capture_output=True, text=True, timeout=args.timeout)
            record_path = run_dir / "result.json"
            if proc.returncode == 0 and record_path.exists():
                record = json.loads(record_path.read_text())
                record.update({"sample": sample, "warmup": warmup,
                               "config_index": index})
                if not warmup:
                    records.append(record)
            else:
                failures.append({"payload": payload, "chunk": chunk, "depth": depth,
                                 "mode": mode, "sample": sample,
                                 "warmup": warmup,
                                 "returncode": proc.returncode,
                                 "stdout": proc.stdout[-4000:],
                                 "stderr": proc.stderr[-4000:]})
                if args.fail_fast:
                    break
        if failures and args.fail_fast:
            break
    depth1_bad = [r for r in records if r["depth"] == 1 and r.get("overlap_rate", 0) > 0.05]
    timeline_missing = [r for r in records if r["mode"] == "async"
                        and r.get("profile_rows", 0) !=
                        (r["payload_bytes"] + r["chunk_bytes"] - 1) // r["chunk_bytes"]]
    depth2_missing = [r for r in records if r["depth"] >= 2 and r["mode"] == "async"
                     and r.get("overlap_rate", 0) <= 0.0]
    gate_failures = ([{"kind": "depth1_overlap", "record": r} for r in depth1_bad] +
                     [{"kind": "timeline_missing", "record": r} for r in timeline_missing] +
                     [{"kind": "depth_ge2_no_overlap", "record": r} for r in depth2_missing])
    summary = {"status": "pass" if not failures and not gate_failures else "fail",
               "configs": len(configs), "warmups_per_config": warmups,
               "requested_samples": len(configs) * samples,
               "samples": len(records), "failures": failures + gate_failures, "records": records,
               "gate": {"all_readback_verified": not failures,
                        "depth1_no_overlap_required": True,
                        "depth_ge2_real_overlap_required": True}}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "samples": len(records),
                      "failures": len(failures), "output": str(root)}, sort_keys=True))
    return 0 if not failures else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--screening", action="store_true")
    parser.add_argument("--payloads", nargs="+", type=int,
                        default=[256 * 1024**2, 1024**3, 1484135432])
    parser.add_argument("--chunks", nargs="+", type=int,
                        default=[1 * 1024**2, 4 * 1024**2, 16 * 1024**2])
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--modes", nargs="+", choices=["serial", "queue", "async"],
                        default=["serial", "queue", "async"])
    parser.add_argument("--screening-samples", type=int, default=3)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--payload", type=int, default=256 * 1024**2)
    parser.add_argument("--chunk", type=int, default=4 * 1024**2)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--mode", choices=["serial", "queue", "async"], default="async")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--offset", type=int, default=64 * 1024**3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--shm-id", type=int, default=18200)
    args = parser.parse_args()
    if args.child:
        args.run_dir = args.run_dir or str(ROOT / "results" / "stage2-async" / "child")
        return 0 if run_one(args)["status"] == "pass" else 1
    return orchestrate(args)


if __name__ == "__main__":
    raise SystemExit(main())
