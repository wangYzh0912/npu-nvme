#!/usr/bin/env python3
"""A9 explicit host snapshot-slot lifecycle experiment.

One DirectCheckpoint/Reactor owner is shared by all workers.  Each pre-
allocated slot is reused through FREE -> SNAPSHOT -> READY -> IO -> FREE;
the experiment varies slot count, not the Reactor pipeline depth.  The
payload is host-generated, so this is a storage-slot lifecycle result rather
than a model HBM snapshot result.
"""

import argparse
import ctypes
import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from experiments.benchmarks.io_matrix import (  # noqa: E402
    ALIGNMENT, ResultWriter, check_npu_free, environment_snapshot,
    host_chunk_arrays, round_up, stats,
)


BASE_OFFSET = 320 * 1024 * 1024 * 1024


@dataclass
class Slot:
    slot_id: int
    item_bytes: int
    offset: int
    state: str = "FREE"

    def __post_init__(self):
        payload = bytes((index * 23 + self.slot_id * 7 + 5) % 256
                        for index in range(self.item_bytes))
        self.payload = ctypes.create_string_buffer(payload, self.item_bytes)
        self.expected = bytes(self.payload.raw[:self.item_bytes])
        self.destination = ctypes.create_string_buffer(self.item_bytes)
        self.digest = hashlib.sha256(self.expected).hexdigest()
        self.lock = threading.Lock()


def host_io(ckpt, slot, read=False):
    from c_bindings import lib

    buffers = [slot.destination if read else slot.payload]
    ptrs, offsets, sizes = host_chunk_arrays(
        buffers, [slot.offset], ckpt.chunk_size)
    start = time.monotonic_ns()
    if read:
        rc = lib.npu_nvme_read_batch_host(
            ckpt.ctx, ptrs, offsets, sizes, len(sizes))
    else:
        rc = lib.npu_nvme_write_batch_host(
            ckpt.ctx, ptrs, offsets, sizes, len(sizes))
    end = time.monotonic_ns()
    if rc != 0:
        raise RuntimeError(f"slot {slot.slot_id} host I/O failed: {rc}")
    return (end - start) / 1000, ckpt.get_last_io_us(read)


def run_slot_count(args, slot_count):
    writer = ResultWriter("A9", args)
    writer.config.update({
        "path": "host_snapshot_slot_lifecycle",
        "slot_count": slot_count,
        "pipeline_depth": args.pipeline_depth,
        "item_bytes": args.item_bytes,
        "base_offset": BASE_OFFSET,
        "scope": "host snapshot slots; not model HBM snapshot",
    })
    writer.write_json("config.json", writer.config)
    check = check_npu_free(args.npu)
    writer.write_json("environment.json", environment_snapshot(args, check))
    from direct_checkpoint import DirectCheckpoint

    ckpt = DirectCheckpoint(
        nvme_addr=args.pci, npu_device_id=args.npu,
        pipeline_depth=args.pipeline_depth,
        requested_chunk_size=min(args.item_bytes, 4 * 1024 * 1024),
        spdk_shm_id=args.shm_id + slot_count,
        profiling_dir=str(writer.run_dir / "profiling"))
    slots = [Slot(index, args.item_bytes,
                  BASE_OFFSET + index * round_up(args.item_bytes))
             for index in range(slot_count)]
    try:
        for wave in range(args.warmups + args.repetitions):
            barrier = threading.Barrier(slot_count)
            outputs = [None] * slot_count
            errors = []

            def worker(slot):
                try:
                    barrier.wait()
                    snapshot_start = time.monotonic_ns()
                    with slot.lock:
                        if slot.state != "FREE":
                            raise RuntimeError(f"slot {slot.slot_id} not FREE")
                        slot.state = "SNAPSHOT"
                    snapshot_end = time.monotonic_ns()
                    with slot.lock:
                        slot.state = "READY"
                    with slot.lock:
                        slot.state = "IO"
                    write_us, c_write = host_io(ckpt, slot, read=False)
                    read_us, c_read = host_io(ckpt, slot, read=True)
                    if bytes(slot.destination.raw[:slot.item_bytes]) != slot.expected:
                        raise AssertionError(f"slot {slot.slot_id} readback mismatch")
                    end = time.monotonic_ns()
                    with slot.lock:
                        slot.state = "FREE"
                    outputs[slot.slot_id] = {
                        "run_id": writer.run_id,
                        "request_id": f"{writer.run_id}/wave_{wave:02d}/slot_{slot.slot_id}",
                        "checkpoint_id": f"slot_{slot.slot_id:02d}_wave_{wave:02d}",
                        "warmup": wave < args.warmups,
                        "path": "A9_host_snapshot_slot_lifecycle",
                        "slot_id": slot.slot_id,
                        "slot_count": slot_count,
                        "bytes": slot.item_bytes,
                        "sha256": slot.digest,
                        "status": "pass",
                        "events": [
                            {"name": "snapshot_start", "monotonic_ns": snapshot_start},
                            {"name": "snapshot_end", "monotonic_ns": snapshot_end},
                            {"name": "slot_ready", "monotonic_ns": snapshot_end},
                            {"name": "slot_free", "monotonic_ns": end},
                        ],
                        "timeline_us": {
                            "snapshot": (snapshot_end - snapshot_start) / 1000,
                            "write": write_us,
                            "read": read_us,
                            "end_to_end": (end - snapshot_start) / 1000,
                        },
                        "c_io_us": {"write": c_write, "read": c_read},
                    }
                except BaseException as error:
                    errors.append(repr(error))
                    with slot.lock:
                        slot.state = "FREE"

            threads = [threading.Thread(target=worker, args=(slot,))
                       for slot in slots]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if errors:
                raise RuntimeError("; ".join(errors))
            if any(slot.state != "FREE" for slot in slots):
                raise AssertionError("slot lifecycle did not return to FREE")
            if wave >= args.warmups:
                for sample in outputs:
                    writer.add_sample(sample)
    except BaseException as error:
        writer.add_failure({"slot_count": slot_count, "error": repr(error)})
    finally:
        ckpt.cleanup()

    expected = args.repetitions * slot_count
    result = writer.finalize({
        "slot_count": slot_count,
        "samples_per_slot": args.repetitions,
        "end_to_end_ms": stats([s["timeline_us"]["end_to_end"] / 1000
                                  for s in writer.samples]),
        "effective_mib_per_s": stats([
            s["bytes"] / (s["timeline_us"]["end_to_end"] / 1e6) /
            (1024 ** 2) for s in writer.samples]),
    }, status="pass" if not writer.failed and
    len(writer.samples) == expected else "fail")
    print(json.dumps({"run_id": writer.run_id, "slot_count": slot_count,
                      "status": result["status"], "summary": result["summary"]},
                     indent=2, sort_keys=True), flush=True)
    if result["status"] != "pass":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="A9")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=211)
    parser.add_argument("--item-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--slots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.item_bytes % ALIGNMENT or args.pipeline_depth < 1:
        raise ValueError("item size and pipeline depth must be valid")
    for slot_count in args.slots:
        run_slot_count(args, slot_count)


if __name__ == "__main__":
    main()
