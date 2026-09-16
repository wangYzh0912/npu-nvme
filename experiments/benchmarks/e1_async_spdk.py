#!/usr/bin/env python3
"""E1 SPDK half: one-owner asynchronous qpair path on raw 83.0.0.

The C reactor submits NVMe operations through an asynchronous qpair, while
the legacy Python call waits for the durable completion.  This is therefore
labelled ``spdk_async_qpair`` and is not presented as ACL DMA overlap; E2 owns
that stronger claim.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "experiments" / "benchmarks"))

from io_matrix import (CHUNK_SIZE, SAFE_OFFSET, check_npu_free,  # noqa: E402
                       host_spdk_sample)
from ppt_evidence import EvidenceBundle, command, environment_snapshot, stats  # noqa: E402


class SampleOwner:
    def __init__(self, run_id):
        self.run_id = run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--shm-id", type=int, default=241)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[4 * 1024, 64 * 1024, 1024 * 1024,
                                 4 * 1024 * 1024, 256 * 1024 * 1024])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--offset", type=int, default=SAFE_OFFSET)
    args = parser.parse_args()
    if args.samples < 30:
        raise SystemExit("formal samples must be >=30")
    npu_info = check_npu_free(args.npu)
    from direct_checkpoint import DirectCheckpoint

    lock_path = Path("/tmp/npu_nvme_83_e1.lock")
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for size in args.sizes:
            for depth in args.depths:
                bundle = EvidenceBundle("E1", {
                    "model": "synthetic_io", "seed": 17,
                    "mode": "spdk_async_qpair", "operation": "write_read",
                    "state_bytes": size,
                    "chunk_size": min(size, CHUNK_SIZE),
                    "pipeline_depth": depth, "slot_count": depth,
                    "warmups": args.warmups, "formal_samples": args.samples,
                    "persistence": "nvme_flush+metadata_commit",
                    "target_pci": args.pci, "npu": args.npu,
                    "raw_offset": args.offset, "cross_disk_label": True,
                    "owner": "single SPDK Reactor",
                }, repo_root=ROOT, environment=environment_snapshot(
                    pci=args.pci, npu=str(args.npu),
                    numa="recorded by snapshot", repo_root=ROOT,
                    npu_info=npu_info))
                owner = SampleOwner(bundle.run_id)
                ckpt = None
                failures = 0
                try:
                    ckpt = DirectCheckpoint(
                        nvme_addr=args.pci, npu_device_id=args.npu,
                        pipeline_depth=depth,
                        requested_chunk_size=min(size, CHUNK_SIZE), rank_id=0,
                        world_size=1, keep_last_n=3, slot_size_gb=10,
                        spdk_shm_id=args.shm_id,
                        profiling_dir=str(bundle.raw_dir / "profiling"))
                    for index in range(args.warmups + args.samples):
                        try:
                            sample = host_spdk_sample(
                                ckpt, owner, size, 1, index,
                                index < args.warmups, args.offset)
                            if index >= args.warmups:
                                bundle.add_sample(sample, events=sample["events"])
                        except BaseException as error:
                            failures += 1
                            bundle.add_failure({"index": index,
                                                "warmup": index < args.warmups,
                                                "error": repr(error)})
                            if index < args.warmups:
                                raise
                except BaseException as error:
                    bundle.add_failure({"stage": "initialization", "error": repr(error)})
                finally:
                    if ckpt is not None:
                        ckpt.cleanup()
                formal = bundle.samples
                end_to_end = [x["timeline_us"]["end_to_end"] for x in formal]
                writes = [x["timeline_us"]["write_api"] for x in formal]
                reads = [x["timeline_us"]["read_api"] for x in formal]
                result = bundle.finalize(metrics={
                    "model": "synthetic_io", "seed": 17,
                    "mode": "spdk_async_qpair", "state_bytes": size,
                    "logical_bytes": size,
                    "physical_bytes": size * len(formal),
                    "chunk_size": min(size, CHUNK_SIZE),
                    "pipeline_depth": depth, "slot_count": depth,
                    "latency_mean": stats(end_to_end).get("mean"),
                    "latency_p50": stats(end_to_end).get("median"),
                    "latency_p95": stats(end_to_end).get("p95"),
                    "throughput": size / (stats(end_to_end).get("mean", 0) / 1e6)
                    if end_to_end else None,
                    "pcie_bytes": 0, "nvme_bytes": size * len(formal),
                    "fault_results": {"failed_operations": failures},
                    "write_api_us": stats(writes),
                    "read_api_us": stats(reads),
                    "end_to_end_us": stats(end_to_end),
                    "sample_note": "qpair async submission; blocking API waits for durable completion",
                }, status="pass" if len(formal) == args.samples and not failures else "fail")
                print(json.dumps({"run_id": result["run_id"],
                                  "status": result["status"], "size": size,
                                  "depth": depth,
                                  "samples": result["samples"]},
                                 sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
