#!/usr/bin/env python3
"""Formal A8 generation/CRC/active-slot protocol experiment.

The live superblock and metadata replicas at the beginning of 83.0.0 are
never touched.  This runner relocates the same metadata envelope and an
active-slot record into a dedicated safety region, then measures the single
slot control, A/B commit, and fallback after corrupting the active replica.
"""

import argparse
import binascii
import ctypes
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from disk_layout import META_SLOT_BYTES, pack_metadata, unpack_metadata  # noqa: E402
from experiments.benchmarks.io_matrix import (  # noqa: E402
    ALIGNMENT, ResultWriter, check_npu_free, environment_snapshot, stats,
)


ACTIVE_MAGIC = b"A8ACTV01"
ACTIVE_BYTES = 4096
ACTIVE_FMT = "<8sQII"
ACTIVE_HEADER_BYTES = struct.calcsize(ACTIVE_FMT)
SAFETY_BASE = 128 * 1024 * 1024 * 1024


def pack_active(generation, slot):
    body = struct.pack("<8sQI", ACTIVE_MAGIC, generation, slot)
    crc = binascii.crc32(body) & 0xFFFFFFFF
    return struct.pack(ACTIVE_FMT, ACTIVE_MAGIC, generation, slot, crc).ljust(
        ACTIVE_BYTES, b"\0")


def unpack_active(raw):
    if len(raw) < ACTIVE_HEADER_BYTES:
        raise ValueError("active record is truncated")
    magic, generation, slot, stored_crc = struct.unpack(
        ACTIVE_FMT, raw[:ACTIVE_HEADER_BYTES])
    body = struct.pack("<8sQI", magic, generation, slot)
    if magic != ACTIVE_MAGIC or slot not in (0, 1):
        raise ValueError("invalid active record")
    if binascii.crc32(body) & 0xFFFFFFFF != stored_crc:
        raise ValueError("active record CRC mismatch")
    return generation, slot


def sync_io(ckpt, offset, size, payload=None, read=False):
    from c_bindings import lib

    buffer = ctypes.create_string_buffer(payload if not read else size)
    start = time.perf_counter_ns()
    ret = lib.npu_nvme_sync_meta_io(
        ckpt.ctx, offset, size, 1 if read else 0,
        ctypes.c_void_p(ctypes.addressof(buffer)))
    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
    if ret != 0:
        raise RuntimeError(f"sync_meta_io failed ({ret}) at {offset}")
    return elapsed_ms, bytes(buffer.raw[:size]) if read else None


def read_replicas(ckpt, slot_offsets, active_offset):
    valid = []
    timings = []
    for slot, offset in enumerate(slot_offsets):
        elapsed, raw = sync_io(ckpt, offset, META_SLOT_BYTES, read=True)
        timings.append(elapsed)
        try:
            generation, payload = unpack_metadata(raw)
            valid.append((generation, slot, payload))
        except (ValueError, json.JSONDecodeError):
            pass
    active_ms, active_raw = sync_io(ckpt, active_offset, ACTIVE_BYTES, read=True)
    active = unpack_active(active_raw)
    if not valid:
        raise AssertionError("both A8 metadata replicas are invalid")
    return max(valid), active, sum(timings) + active_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="A8")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=209)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if META_SLOT_BYTES % ALIGNMENT or SAFETY_BASE % ALIGNMENT:
        raise RuntimeError("A8 safety geometry is not 4 KiB aligned")

    writer = ResultWriter("A8", args)
    slot_a = SAFETY_BASE
    slot_b = slot_a + META_SLOT_BYTES
    active_offset = slot_b + META_SLOT_BYTES
    writer.config.update({
        "path": "metadata_ab_generation_crc_active",
        "slot_bytes": META_SLOT_BYTES,
        "slot_a_offset": slot_a,
        "slot_b_offset": slot_b,
        "active_record_offset": active_offset,
        "active_record_bytes": ACTIVE_BYTES,
    })
    writer.write_json("config.json", writer.config)
    npu_info = check_npu_free(args.npu)
    writer.write_json("environment.json", environment_snapshot(args, npu_info))

    ckpt = None
    try:
        from direct_checkpoint import DirectCheckpoint

        ckpt = DirectCheckpoint(
            nvme_addr=args.pci, npu_device_id=args.npu, pipeline_depth=1,
            requested_chunk_size=4 * 1024 * 1024, spdk_shm_id=args.shm_id,
            profiling_dir=str(writer.run_dir / "profiling"))
        control_payload = bytes((index * 13 + 7) % 256
                                for index in range(META_SLOT_BYTES))
        control_hash = hashlib.sha256(control_payload).hexdigest()
        slot_offsets = (slot_a, slot_b)

        for index in range(args.warmups + args.repetitions):
            flush_start = time.perf_counter_ns()
            single_write, _ = sync_io(
                ckpt, slot_a, META_SLOT_BYTES, control_payload, read=False)
            ckpt.flush_nvme()
            single_flush = (time.perf_counter_ns() - flush_start) / 1e6
            single_read, raw = sync_io(
                ckpt, slot_a, META_SLOT_BYTES, read=True)
            if raw != control_payload:
                raise AssertionError("A8 single-slot control mismatch")

            generation_a = index * 2
            generation_b = generation_a + 1
            crc_start = time.perf_counter_ns()
            payload_a = pack_metadata({"round": index, "slot": "A"}, generation_a)
            payload_b = pack_metadata({"round": index, "slot": "B"}, generation_b)
            crc_generation_ms = (time.perf_counter_ns() - crc_start) / 1e6
            commit_start = time.perf_counter_ns()
            sync_io(ckpt, slot_a, META_SLOT_BYTES, payload_a, read=False)
            ckpt.flush_nvme()
            sync_io(ckpt, slot_b, META_SLOT_BYTES, payload_b, read=False)
            ckpt.flush_nvme()
            sync_io(ckpt, active_offset, ACTIVE_BYTES,
                    pack_active(generation_b, 1), read=False)
            ckpt.flush_nvme()
            winner, active, read_ms = read_replicas(
                ckpt, slot_offsets, active_offset)
            commit_ms = (time.perf_counter_ns() - commit_start) / 1e6
            if winner[0] != generation_b or winner[1] != 1:
                raise AssertionError(f"A8 active winner mismatch: {winner}")
            if active != (generation_b, 1):
                raise AssertionError(f"A8 active record mismatch: {active}")

            # Corrupt the active B replica and verify that generation/CRC
            # validation falls back to the intact A replica.
            corrupt = bytearray(payload_b)
            corrupt[40] ^= 0x01
            fault_start = time.perf_counter_ns()
            sync_io(ckpt, slot_b, META_SLOT_BYTES, bytes(corrupt), read=False)
            fallback, _, _ = read_replicas(ckpt, slot_offsets, active_offset)
            sync_io(ckpt, slot_b, META_SLOT_BYTES, payload_b, read=False)
            fault_ms = (time.perf_counter_ns() - fault_start) / 1e6
            if fallback[0] != generation_a or fallback[1] != 0:
                raise AssertionError(f"A8 fallback mismatch: {fallback}")

            sample = {
                "run_id": writer.run_id,
                "request_id": f"{writer.run_id}/request_{index:02d}",
                "checkpoint_id": f"metadata_{index:02d}",
                "warmup": index < args.warmups,
                "path": "A8_metadata_ab_generation_crc_active",
                "bytes": META_SLOT_BYTES,
                "status": "pass",
                "control_sha256": control_hash,
                "events": [
                    {"name": "inactive_replica_committed", "generation": generation_b},
                    {"name": "active_slot_switched", "slot": 1},
                    {"name": "active_replica_corrupted", "fallback_generation": generation_a},
                    {"name": "active_replica_restored", "generation": generation_b},
                ],
                "timeline_us": {
                    "single_write": single_write * 1000,
                    "single_flush": single_flush * 1000,
                    "crc_generation": crc_generation_ms * 1000,
                    "single_read": single_read * 1000,
                    "ab_commit": commit_ms * 1000,
                    "ab_read_validation": read_ms * 1000,
                    "fault_recovery": fault_ms * 1000,
                    "end_to_end": (single_write + single_read + commit_ms + fault_ms) * 1000,
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
        "single_flush_ms": stats([s["timeline_us"]["single_flush"] / 1000
                                   for s in writer.samples]),
        "crc_generation_ms": stats([s["timeline_us"]["crc_generation"] / 1000
                                     for s in writer.samples]),
        "ab_commit_ms": stats([s["timeline_us"]["ab_commit"] / 1000
                                for s in writer.samples]),
        "fault_recovery_ms": stats([s["timeline_us"]["fault_recovery"] / 1000
                                     for s in writer.samples]),
    }, status="pass" if not writer.failed and
    len(writer.samples) == args.repetitions else "fail")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
