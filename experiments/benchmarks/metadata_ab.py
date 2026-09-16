#!/usr/bin/env python3
"""A8 metadata cost experiment in a raw, non-ledger safety region.

The live superblock and A/B metadata slots are read by DirectCheckpoint but
never written here.  The same 400 KiB envelopes are written to two aligned
locations starting at 65 GiB, then read back and verified.  This measures the
metadata I/O and A/B protocol cost without changing the correctness ledger.
"""

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarks.io_matrix import (  # noqa: E402
    ALIGNMENT, SAFE_OFFSET, ResultWriter, check_npu_free,
    environment_snapshot, stats,
)
from disk_layout import META_SLOT_BYTES  # noqa: E402


def io_once(ckpt, offset, payload, read):
    from c_bindings import lib

    buf = ctypes.create_string_buffer(payload if not read else META_SLOT_BYTES)
    start = time.perf_counter_ns()
    ret = lib.npu_nvme_sync_meta_io(
        ckpt.ctx, offset, META_SLOT_BYTES, 1 if read else 0,
        ctypes.c_void_p(ctypes.addressof(buf)))
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if ret != 0:
        raise RuntimeError(f"sync_meta_io failed ({ret}) at {offset}")
    return elapsed, bytes(buf.raw[:META_SLOT_BYTES]) if read else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="A8")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=83)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if META_SLOT_BYTES % ALIGNMENT:
        raise RuntimeError("metadata slot is not 4 KiB aligned")
    writer = ResultWriter("A8", args)
    # Keep clear of the ~3.3 GiB GPT-2 XL raw benchmark image at SAFE_OFFSET.
    base = SAFE_OFFSET + 8 * 1024 * 1024 * 1024
    slot_a = base
    slot_b = base + META_SLOT_BYTES
    writer.config.update({"path": "safe_metadata_region", "slot_bytes": META_SLOT_BYTES,
                          "control_offset": slot_a, "slot_a_offset": slot_a,
                          "slot_b_offset": slot_b})
    writer.write_json("config.json", writer.config)
    npu_info = check_npu_free(args.npu)
    writer.write_json("environment.json", environment_snapshot(args, npu_info))
    ckpt = None
    try:
        from direct_checkpoint import DirectCheckpoint
        ckpt = DirectCheckpoint(nvme_addr=args.pci, npu_device_id=args.npu,
                                pipeline_depth=1, requested_chunk_size=4 * 1024 * 1024,
                                spdk_shm_id=args.shm_id,
                                profiling_dir=str(writer.run_dir / "profiling"))
        payload_a = bytes((index * 13 + 7) % 256 for index in range(META_SLOT_BYTES))
        payload_b = bytes((index * 17 + 11) % 256 for index in range(META_SLOT_BYTES))
        for index in range(args.warmups + args.repetitions):
            control_write, _ = io_once(ckpt, slot_a, payload_a, False)
            control_read, value_a = io_once(ckpt, slot_a, None, True)
            if value_a != payload_a:
                raise AssertionError("A8 single-slot readback mismatch")
            ab_start = time.perf_counter_ns()
            _, _ = io_once(ckpt, slot_a, payload_a, False)
            _, _ = io_once(ckpt, slot_b, payload_b, False)
            _, read_a = io_once(ckpt, slot_a, None, True)
            _, read_b = io_once(ckpt, slot_b, None, True)
            ab_ms = (time.perf_counter_ns() - ab_start) / 1e6
            if read_a != payload_a or read_b != payload_b:
                raise AssertionError("A8 A/B readback mismatch")
            sample = {
                "run_id": writer.run_id, "request_id": f"{writer.run_id}/request_{index:02d}",
                "checkpoint_id": f"metadata_{index:02d}", "warmup": index < args.warmups,
                "path": "A8_metadata_ab", "bytes": META_SLOT_BYTES,
                "status": "pass", "sha256": [hashlib.sha256(payload_a).hexdigest(),
                                                hashlib.sha256(payload_b).hexdigest()],
                "events": [], "timeline_us": {
                    "single_write": control_write * 1000,
                    "single_read": control_read * 1000,
                    "ab_protocol": ab_ms * 1000,
                    "end_to_end": (control_write + control_read + ab_ms) * 1000,
                },
            }
            if index >= args.warmups:
                writer.add_sample(sample)
    except BaseException as error:
        writer.add_failure({"error": repr(error)})
    finally:
        if ckpt is not None:
            ckpt.cleanup()
    result = writer.finalize({
        "single_write_ms": stats([s["timeline_us"]["single_write"] / 1000
                                   for s in writer.samples]),
        "single_read_ms": stats([s["timeline_us"]["single_read"] / 1000
                                  for s in writer.samples]),
        "ab_protocol_ms": stats([s["timeline_us"]["ab_protocol"] / 1000
                                  for s in writer.samples]),
    }, status="pass" if not writer.failed and len(writer.samples) == args.repetitions else "fail")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
