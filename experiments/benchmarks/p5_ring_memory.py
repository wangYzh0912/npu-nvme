#!/usr/bin/env python3
"""P5 ring-buffer memory matrix.

Runs the complete Host-staging baseline and an instrumented DMA-pool lane for
1/2/4 slots and 1/4/16 MiB chunks.  The optional HBM lifecycle lane is kept
behind ``--include-hbm-slots`` because P5's required metric is DRAM RSS.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = (Path("/home/user7/miniconda3/envs/ms_2.5/bin/python")
                 if Path("/home/user7/miniconda3/envs/ms_2.5/bin/python").exists()
                 else Path(sys.executable))


def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=stream,
                              stderr=subprocess.STDOUT, text=True, check=False)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", nargs="+", type=int,
                        default=(1, 4, 16), help="MiB")
    parser.add_argument("--slots", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--pci", default="0000:83:00.0")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--ckpt-every", type=int, default=5)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--include-hbm-slots", action="store_true",
                        help="also run the separate real-HBM lifecycle lane")
    args = parser.parse_args()
    root = Path(args.output_root or ROOT / "results/ppt-evidence-20260829/P5")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for chunk_mib in args.chunks:
        chunk = chunk_mib * 1024 ** 2
        host_dir = root / f"host_{chunk_mib}MiB"
        code = run([str(DEFAULT_PYTHON), str(ROOT / "experiments/benchmarks/e3_host_staging.py"),
                    "--model", "gpt2_xl", "--warmups", str(args.warmups),
                    "--samples", str(args.samples), "--chunk-size", str(chunk),
                    "--complete-training-state",
                    "--output-root", str(host_dir)], host_dir / "stdout.log")
        records.append({"mode": "host_staging", "chunk_size": chunk,
                        "returncode": code, "path": str(host_dir)})
        for slots in args.slots:
            out = root / f"slots_{slots}_{chunk_mib}MiB"
            out.mkdir(parents=True, exist_ok=True)
            dma_code = run([str(DEFAULT_PYTHON), str(ROOT / "experiments/benchmarks/p5_dma_pool_memory.py"),
                            "--npu", str(args.npu), "--pci", args.pci,
                            "--slots", str(slots), "--chunk-size", str(chunk),
                            "--warmups", str(args.warmups), "--samples", str(args.samples),
                            "--shm-id", str(16000 + slots * 100 + chunk_mib),
                            "--output-root", str(out)], out / "dma_pool.stdout.log")
            records.append({"mode": "dma_ring", "slots": slots,
                            "chunk_size": chunk, "returncode": dma_code,
                            "path": str(out)})
            if not args.include_hbm_slots:
                continue
            code = run([str(DEFAULT_PYTHON), str(ROOT / "experiments/benchmarks/e3_hbm_evidence.py"),
                        "--npu", str(args.npu), "--slots", str(slots),
                        "--steps", str(args.steps), "--ckpt-every", str(args.ckpt_every),
                        "--warmups", str(args.warmups), "--raw-root", str(out / "raw"),
                        "--chunk-size", str(chunk),
                        "--pipeline-depth", str(slots),
                        "--evidence-root", str(out)], out / "stdout.log")
            records.append({"mode": "hbm_slots", "slots": slots,
                            "chunk_size": chunk, "returncode": code,
                            "path": str(out)})
    (root / "matrix.json").write_text(json.dumps({"experiment": "P5",
        "records": records, "rss_metrics": ["baseline", "incremental", "peak",
        "slot_wait_ms"]}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"experiment": "P5", "records": records}, sort_keys=True))


if __name__ == "__main__":
    main()
