#!/usr/bin/env python3
"""A9 safe concurrent request-slot experiment.

One DirectCheckpoint context remains the sole SPDK owner.  N Python workers
prepare independent host snapshot slots and submit through the C MPSC request
ring.  The device/repository state is therefore not exposed to the unsafe
multi-owner setup from the archived baseline.
"""

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from experiments.benchmarks.io_matrix import (  # noqa: E402
    ALIGNMENT, ResultWriter, SAFE_OFFSET, check_npu_free,
    environment_snapshot, host_spdk_sample, stats,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("A9", "E5"), default="E5")
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=83)
    parser.add_argument("--item-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--slots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--offset", type=int, default=160 * 1024 * 1024 * 1024)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    if args.item_bytes % ALIGNMENT or args.offset % ALIGNMENT:
        raise ValueError("item-bytes and offset must be 4 KiB aligned")
    check = check_npu_free(args.npu)
    from direct_checkpoint import DirectCheckpoint

    for slot_count in args.slots:
        writer = ResultWriter(args.experiment, args)
        writer.config.update({"path": "mpsc_request_ring", "slot_count": slot_count,
                              "pipeline_depth": args.pipeline_depth,
                              "item_bytes": args.item_bytes,
                              "producer_count": slot_count,
                              "owner": "single SPDK Reactor owner",
                              "measurement_kind": "control pressure; model payload size only"})
        writer.write_json("config.json", writer.config)
        writer.write_json("environment.json", environment_snapshot(args, check))
        ckpt = DirectCheckpoint(
            nvme_addr=args.pci, npu_device_id=args.npu,
            pipeline_depth=args.pipeline_depth,
            requested_chunk_size=min(args.item_bytes, 4 * 1024 * 1024),
            spdk_shm_id=args.shm_id,
            profiling_dir=str(writer.run_dir / "profiling"))
        try:
            for wave in range(args.warmups + args.repetitions):
                with concurrent.futures.ThreadPoolExecutor(max_workers=slot_count) as pool:
                    futures = [pool.submit(
                        host_spdk_sample, ckpt, writer, args.item_bytes, 1,
                        wave * slot_count + slot, wave < args.warmups,
                        args.offset)
                        for slot in range(slot_count)]
                    samples = [future.result() for future in futures]
                if wave >= args.warmups:
                    for slot, sample in enumerate(samples):
                        sample["slot_id"] = slot
                        sample["slot_count"] = slot_count
                        writer.add_sample(sample)
        except BaseException as error:
            writer.add_failure({"slot_count": slot_count, "error": repr(error)})
        finally:
            ckpt.cleanup()
        expected = args.repetitions * slot_count
        status = "pass" if not writer.failed and len(writer.samples) == expected else "fail"
        summary = {
            "slot_count": slot_count,
            "end_to_end_us": stats([s["timeline_us"]["end_to_end"]
                                     for s in writer.samples]),
            "effective_mib_per_s": stats([
                s["bytes"] / (s["timeline_us"]["end_to_end"] / 1e6) /
                (1024 ** 2) for s in writer.samples]),
            "samples_per_slot": args.repetitions,
        }
        result = writer.finalize(summary, status=status)
        print(json.dumps({"run_id": writer.run_id, "slot_count": slot_count,
                          "status": result["status"], "summary": summary},
                         indent=2, sort_keys=True), flush=True)
        if status != "pass":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
