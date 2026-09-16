#!/usr/bin/env python3
"""E5 real-model, single-owner MPSC pressure test.

Each producer owns a disjoint set of 4 MiB HBM chunks, while all producers
submit through one DirectCheckpoint context and one SPDK Reactor.  The
experiment measures control/data-plane pressure and verifies the full model
round trip by hashing the device parameters before and after each wave.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from experiments.baselines import two_phase_common as tpc  # noqa: E402
from experiments.benchmarks.io_matrix import (  # noqa: E402
    ALIGNMENT, SAFE_OFFSET, ResultWriter, check_npu_free,
    environment_snapshot, stats,
)
from experiments.benchmarks.model_paths import assign_safe_offsets  # noqa: E402
from experiments.common import (  # noqa: E402
    init_env, make_causal_lm_checkpoint_model, warmup_checkpoint_model,
)
from chunk_helpers import build_chunks, build_ctypes_arrays  # noqa: E402


def device_hash_flat(flat_chunks, device_id):
    """Hash a contiguous logical HBM snapshot with one D2H per flat chunk.

    The first version hashed every MindSpore parameter independently.  XL has
    2318 parameters, so that verification path introduced thousands of
    synchronous ACL submissions before the first E5 sample.  The flat HBM
    chunks are already the immutable payload used by the I/O wave; hashing
    those chunks preserves full-state verification while keeping the number
    of D2H operations bounded by the ~1 GiB allocation limit.
    """
    tpc._ensure_acl_device(device_id)
    scratch = __import__("numpy").empty(
        max(end - start for _ptr, start, end in flat_chunks),
                                  dtype=__import__("numpy").uint8)
    digest = hashlib.sha256()
    for ptr, start, end in flat_chunks:
        size = end - start
        ret = tpc.acl_lib.aclrtMemcpy(
            ctypes.c_void_p(scratch.ctypes.data), size,
            ctypes.c_void_p(ptr), size,
            tpc.ACL_MEMCPY_DEVICE_TO_HOST)
        tpc._check_acl_ret(ret, f"E5 flat hash {start}")
        digest.update(memoryview(scratch)[:size])
    return digest.hexdigest()


def aligned_layout(descs):
    """Return a 4 KiB-aligned logical layout for the flat HBM snapshot."""
    offsets = {}
    cursor = 0
    for item in descs:
        cursor = (cursor + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        offsets[item["name"]] = cursor
        cursor += item["size"]
    total = (cursor + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    return offsets, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("gpt2_xl", "gpt2_13b"), required=True)
    parser.add_argument("--npu", type=int, default=4)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--shm-id", type=int, default=9000)
    parser.add_argument("--producers", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--offset", type=int, default=320 * 1024**3)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.samples < 30 or args.chunk_size % ALIGNMENT or args.offset % ALIGNMENT:
        raise SystemExit("samples must be >=30 and offsets/chunk aligned")
    check_npu_free(args.npu)
    init_env(device_id=args.npu)
    model, cfg = make_causal_lm_checkpoint_model(args.model, seq_len=128)
    warmup_checkpoint_model(model, cfg, seq_len=128)
    descs = tpc.get_param_descriptors(model)
    safe_descs, safe_end = assign_safe_offsets(descs, args.offset)
    total = sum(item["size"] for item in descs)
    from direct_checkpoint import DirectCheckpoint
    from c_bindings import lib
    output_root = Path(args.output_root or ROOT / "results/ppt-evidence-20260829/E5/model-pressure")
    all_results = []
    for producer_count in args.producers:
        writer = ResultWriter("E5", args)
        writer.config.update({"model": args.model, "mode": "model_mpsc_single_owner",
                              "producer_count": producer_count,
                              "owner": "single SPDK Reactor owner",
                              "chunk_size": args.chunk_size, "state_bytes": total,
                              "safe_region": [args.offset, safe_end],
                              "persistence": "payload completion; no generation publish"})
        writer.write_json("config.json", writer.config)
        writer.write_json("environment.json", environment_snapshot(args,
                                                                     None))
        ckpt = DirectCheckpoint(nvme_addr=args.pci, npu_device_id=args.npu,
                                pipeline_depth=4,
                                requested_chunk_size=args.chunk_size,
                                spdk_shm_id=args.shm_id + producer_count,
                                profiling_dir=str(writer.run_dir / "profiling"))
        flat = []
        try:
            if safe_end > ckpt.total_bytes:
                raise RuntimeError(f"safe model range exceeds 83 capacity: {safe_end}")
            offset_map, flat_total = aligned_layout(descs)
            flat = tpc.build_flat_hbm_buffer(
                model, descs, offset_map, flat_total, args.npu)
            flat_payloads = [
                {"ptr": ptr, "size": end - start,
                 "offset": args.offset + start,
                 "name": f"flat@{start}"}
                for ptr, start, end in flat
            ]
            chunks, _ = build_chunks(flat_payloads, args.chunk_size)
            groups = [chunks[index::producer_count] for index in range(producer_count)]
            arrays = [build_ctypes_arrays(group) for group in groups]
            pre_hash = device_hash_flat(flat, args.npu)
            for wave in range(args.warmups + args.samples):
                wave_start = time.perf_counter_ns()
                def one(index):
                    ptrs, offsets, sizes = arrays[index]
                    start = time.perf_counter_ns()
                    rc = lib.npu_nvme_write_batch(ckpt.ctx, ptrs, offsets, sizes,
                                                  len(groups[index]))
                    if rc != 0:
                        raise RuntimeError(f"producer {index} write rc={rc}")
                    write_ms = (time.perf_counter_ns() - start) / 1e6
                    read_start = time.perf_counter_ns()
                    rc = lib.npu_nvme_read_batch(ckpt.ctx, ptrs, offsets, sizes,
                                                 len(groups[index]))
                    if rc != 0:
                        raise RuntimeError(f"producer {index} read rc={rc}")
                    return {"producer": index, "chunks": len(groups[index]),
                            "write_ms": write_ms,
                            "read_ms": (time.perf_counter_ns() - read_start) / 1e6}
                with concurrent.futures.ThreadPoolExecutor(max_workers=producer_count) as pool:
                    futures = [pool.submit(one, index) for index in range(producer_count)]
                    producer_results = [future.result() for future in futures]
                digest = device_hash_flat(flat, args.npu)
                if digest != pre_hash:
                    raise AssertionError("model hash mismatch after MPSC round trip")
                if wave >= args.warmups:
                    writer.add_sample({
                        "run_id": writer.run_id,
                        "request_id": f"{writer.run_id}/wave_{wave:04d}",
                        "checkpoint_id": f"model_wave_{wave:04d}",
                        "warmup": False, "status": "pass", "model": args.model,
                        "mode": "model_mpsc_single_owner",
                        "producer_count": producer_count, "state_bytes": total,
                        "bytes": total, "hash": digest,
                        "wave_ms": (time.perf_counter_ns() - wave_start) / 1e6,
                        "producer_results": producer_results,
                        "events": [{"name": "wave_complete", "monotonic_ns": time.monotonic_ns()}],
                    })
        except BaseException as error:
            writer.add_failure({"producer_count": producer_count, "error": repr(error)})
        finally:
            if flat:
                tpc.free_flat_hbm_chunks(flat)
            ckpt.cleanup()
        waves = [row["wave_ms"] for row in writer.samples]
        metrics = {"model": args.model, "mode": "model_mpsc_single_owner",
                   "state_bytes": total, "logical_bytes": total,
                   "physical_bytes": total * len(waves), "chunk_size": args.chunk_size,
                   "pipeline_depth": 4, "slot_count": producer_count,
                   "latency_mean": stats(waves).get("mean"),
                   "latency_p50": stats(waves).get("median"),
                   "latency_p95": stats(waves).get("p95"),
                   "throughput": stats([total / (value / 1000) / 1024**2
                                         for value in waves]),
                   "foreground_wait": None, "step_overhead": None,
                   "host_rss_peak": None, "pinned_dram_peak": None, "hbm_peak": None,
                   "pcie_bytes": 0, "nvme_bytes": total * len(waves),
                   "recovery_error": 0 if waves else None, "loss_deviation": None,
                   "fault_results": {"device_hash_roundtrip": "pass" if waves else "fail"},
                   "producer_count": producer_count,
                   "owner": "single SPDK Reactor owner"}
        status = "pass" if not writer.failed and len(waves) == args.samples else "fail"
        result = writer.finalize(metrics, status=status)
        all_results.append({"run_id": writer.run_id, "status": status,
                            "producer_count": producer_count,
                            "samples": len(waves)})
        print(json.dumps(all_results[-1], sort_keys=True), flush=True)
    if any(item["status"] != "pass" for item in all_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
