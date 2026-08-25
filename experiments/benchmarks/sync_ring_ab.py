#!/usr/bin/env python3
"""A6 controlled synchronous API versus request-ring/FSM microbenchmark.

Both paths use the same raw 83.0.0 device and the same 400 KiB payload in a
dedicated safety region.  The control path calls the bounded synchronous
metadata API directly; the experimental path goes through the normal
request ring and Reactor FSM using host buffers.  This isolates API/control
plane overhead and is intentionally not presented as a full model result.
"""

import argparse
import ctypes
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from disk_layout import META_SLOT_BYTES  # noqa: E402
from experiments.benchmarks.io_matrix import (  # noqa: E402
    ALIGNMENT, ResultWriter, SAFE_OFFSET, check_npu_free, environment_snapshot,
    host_chunk_arrays, stats,
)


SYNC_OFFSET = SAFE_OFFSET + 192 * 1024 * 1024 * 1024
RING_OFFSET = SYNC_OFFSET + 2 * META_SLOT_BYTES


def sync_once(ckpt, offset, payload=None, read=False):
    from c_bindings import lib

    buffer = ctypes.create_string_buffer(payload if not read else META_SLOT_BYTES)
    start = time.perf_counter_ns()
    rc = lib.npu_nvme_sync_meta_io(
        ckpt.ctx, offset, META_SLOT_BYTES, 1 if read else 0,
        ctypes.c_void_p(ctypes.addressof(buffer)))
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    if rc != 0:
        raise RuntimeError(f"sync_meta_io failed ({rc})")
    return elapsed_ms, bytes(buffer.raw[:META_SLOT_BYTES]) if read else None


def ring_once(ckpt, offset, payload, read=False):
    from c_bindings import lib

    source = ctypes.create_string_buffer(payload, len(payload))
    buffers = [source] if not read else [ctypes.create_string_buffer(META_SLOT_BYTES)]
    ptrs, offsets, sizes = host_chunk_arrays(
        buffers, [offset], ckpt.chunk_size)
    start = time.perf_counter_ns()
    if read:
        rc = lib.npu_nvme_read_batch_host(
            ckpt.ctx, ptrs, offsets, sizes, len(sizes))
    else:
        rc = lib.npu_nvme_write_batch_host(
            ckpt.ctx, ptrs, offsets, sizes, len(sizes))
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    if rc != 0:
        raise RuntimeError(f"request-ring I/O failed ({rc})")
    value = bytes(buffers[0].raw[:META_SLOT_BYTES]) if read else None
    return elapsed_ms, value, ckpt.get_last_io_us(read)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="A6")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=210)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if META_SLOT_BYTES % ALIGNMENT:
        raise ValueError("metadata payload must be 4 KiB aligned")

    writer = ResultWriter("A6", args)
    writer.config.update({
        "path": "sync_meta_control_vs_request_ring_fsm",
        "payload_bytes": META_SLOT_BYTES,
        "sync_offset": SYNC_OFFSET,
        "ring_offset": RING_OFFSET,
        "scope": "API/control-plane microbenchmark, not full model checkpoint",
    })
    writer.write_json("config.json", writer.config)
    npu_info = check_npu_free(args.npu)
    writer.write_json("environment.json", environment_snapshot(args, npu_info))

    ckpt = None
    try:
        from direct_checkpoint import DirectCheckpoint

        ckpt = DirectCheckpoint(
            nvme_addr=args.pci, npu_device_id=args.npu, pipeline_depth=4,
            requested_chunk_size=4 * 1024 * 1024, spdk_shm_id=args.shm_id,
            profiling_dir=str(writer.run_dir / "profiling"))
        payload = bytes((index * 19 + 23) % 256
                        for index in range(META_SLOT_BYTES))
        digest = hashlib.sha256(payload).hexdigest()
        for index in range(args.warmups + args.repetitions):
            sync_write, _ = sync_once(ckpt, SYNC_OFFSET, payload, read=False)
            sync_read, value = sync_once(ckpt, SYNC_OFFSET, read=True)
            if value != payload:
                raise AssertionError("sync control readback mismatch")
            ring_write, _, ring_c_write = ring_once(
                ckpt, RING_OFFSET, payload, read=False)
            ring_read, value, ring_c_read = ring_once(
                ckpt, RING_OFFSET, payload, read=True)
            if value != payload:
                raise AssertionError("request-ring readback mismatch")
            sample = {
                "run_id": writer.run_id,
                "request_id": f"{writer.run_id}/request_{index:02d}",
                "checkpoint_id": f"a6_{index:02d}",
                "warmup": index < args.warmups,
                "path": "sync_control_vs_request_ring_fsm",
                "bytes": META_SLOT_BYTES,
                "status": "pass",
                "sha256": digest,
                "events": [],
                "timeline_us": {
                    "sync_write": sync_write * 1000,
                    "sync_read": sync_read * 1000,
                    "ring_write": ring_write * 1000,
                    "ring_read": ring_read * 1000,
                    "end_to_end": (sync_write + sync_read + ring_write + ring_read) * 1000,
                },
                "c_io_us": {"write": ring_c_write, "read": ring_c_read},
            }
            if index >= args.warmups:
                writer.add_sample(sample)
    except BaseException as error:
        writer.add_failure({"error": repr(error)})
    finally:
        if ckpt is not None:
            ckpt.cleanup()

    result = writer.finalize({
        "sync_write_ms": stats([s["timeline_us"]["sync_write"] / 1000
                                 for s in writer.samples]),
        "sync_read_ms": stats([s["timeline_us"]["sync_read"] / 1000
                                for s in writer.samples]),
        "ring_write_ms": stats([s["timeline_us"]["ring_write"] / 1000
                                 for s in writer.samples]),
        "ring_read_ms": stats([s["timeline_us"]["ring_read"] / 1000
                                for s in writer.samples]),
    }, status="pass" if not writer.failed and
    len(writer.samples) == args.repetitions else "fail")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
