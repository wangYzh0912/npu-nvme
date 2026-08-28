#!/usr/bin/env python3
"""E5 single-owner control-plane evidence adapter.

The legacy ``sync_ring_ab.py`` already compares the bounded synchronous
metadata API with the normal request-ring/FSM batch API on 83.0.0.  This
runner keeps that implementation unchanged and converts its completed run to
the common PPT evidence contract.  It is deliberately labelled as a control
microbenchmark, not a full-model checkpoint result.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
from ppt_evidence import EvidenceBundle, environment_snapshot, stats  # noqa: E402


def adapt(raw_run: Path, args, stdout_path: Path) -> Path:
    result = json.loads((raw_run / "result.json").read_text())
    config = json.loads((raw_run / "config.json").read_text())
    env_path = raw_run / "environment.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    samples_path = raw_run / "samples.jsonl"
    samples = ([json.loads(line) for line in samples_path.read_text().splitlines()
                if line.strip()] if samples_path.exists() else [])
    bundle = EvidenceBundle("E5", {
        "model": "control_microbench",
        "seed": None,
        "mode": "single_owner_sync_vs_request_ring_batch",
        "pci": config.get("pci", "0000:83:00.0"),
        "npu": config.get("npu", args.npu),
        "numa": None,
        "state_bytes": config.get("payload_bytes"),
        "payload_bytes": config.get("payload_bytes"),
        "producer_count": 1,
        "owner": "single SPDK Reactor owner",
        "warmups": config.get("warmups"),
        "formal_samples": len(samples),
        "measurement_kind": "control microbenchmark; not full model",
    }, root=args.evidence_root, repo_root=REPO_ROOT,
    environment=env or environment_snapshot(
        pci=config.get("pci", "0000:83:00.0"), npu=args.npu))
    shutil.copytree(raw_run, bundle.raw_dir / "legacy_run")
    if stdout_path.exists():
        shutil.copy2(stdout_path, bundle.raw_dir / "child.stdout.log")
    for sample in samples:
        record = dict(sample)
        record.update({"experiment": "E5", "mode": bundle.config["mode"],
                       "state_bytes": config.get("payload_bytes"),
                       "logical_bytes": config.get("payload_bytes"),
                       "physical_bytes": None,
                       "owner": "single SPDK Reactor owner"})
        bundle.add_sample(record)
    failures_path = raw_run / "failures.jsonl"
    if failures_path.exists():
        for line in failures_path.read_text().splitlines():
            if line.strip():
                bundle.add_failure(json.loads(line))
    ring_total = [s["timeline_us"]["ring_write"] / 1000 +
                  s["timeline_us"]["ring_read"] / 1000 for s in samples]
    ring_write = [s["timeline_us"]["ring_write"] / 1000 for s in samples]
    ring_read = [s["timeline_us"]["ring_read"] / 1000 for s in samples]
    metrics = {
        "model": "control_microbench", "mode": bundle.config["mode"],
        "state_bytes": config.get("payload_bytes"),
        "logical_bytes": config.get("payload_bytes"),
        "physical_bytes": None, "chunk_size": 4 * 1024 * 1024,
        "pipeline_depth": 4, "slot_count": 1,
        "latency_mean": stats(ring_total),
        "latency_p50": stats(ring_total).get("median"),
        "latency_p95": stats(ring_total).get("p95"),
        "throughput": stats([config.get("payload_bytes", 0) /
                              (value / 1000) / (1024 ** 2)
                              for value in ring_total]),
        "foreground_wait": None, "step_overhead": None,
        "host_rss_peak": None, "pinned_dram_peak": None, "hbm_peak": None,
        "pcie_bytes": None, "nvme_bytes": None,
        "recovery_error": 0 if result.get("status") == "pass" else None,
        "loss_deviation": None,
        "fault_results": {"sync_vs_ring_readback": "pass" if result.get("status") == "pass" else "fail"},
        "sync_write_ms": stats([s["timeline_us"]["sync_write"] / 1000 for s in samples]),
        "sync_read_ms": stats([s["timeline_us"]["sync_read"] / 1000 for s in samples]),
        "ring_write_ms": stats(ring_write), "ring_read_ms": stats(ring_read),
        "producer_count": 1,
        "protocol_note": "request-ring API is asynchronously serviced by the Reactor but this benchmark waits for durable completion per request",
    }
    status = "pass" if result.get("status") == "pass" and len(samples) >= args.required_samples else "fail"
    bundle.finalize(metrics=metrics, status=status)
    return bundle.run_dir


def newest(root: Path) -> Path:
    runs = sorted(root.glob("A6_*/result.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no sync_ring_ab result under {root}")
    return runs[-1].parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--shm-id", type=int, default=7000)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--required-samples", type=int, default=30)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--evidence-root", default=None)
    args = parser.parse_args()
    args.raw_root = args.raw_root or str(REPO_ROOT / "results/ppt-evidence-20260829/E5/raw")
    args.evidence_root = args.evidence_root or str(REPO_ROOT / "results/ppt-evidence-20260829")
    raw_root = Path(args.raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(REPO_ROOT / "experiments/benchmarks/sync_ring_ab.py"),
           "--npu", str(args.npu), "--shm-id", str(args.shm_id),
           "--warmups", str(args.warmups), "--repetitions", str(args.repetitions),
           "--output-root", str(raw_root)]
    log = raw_root / "child.stdout.log"
    stream = log.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"},
                            start_new_session=True)
    def relay():
        for line in proc.stdout:
            stream.write(line)
            stream.flush()
    relay_thread = threading.Thread(target=relay, daemon=True)
    relay_thread.start()
    try:
        proc.wait()
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
        raise
    relay_thread.join(timeout=10)
    stream.close()
    raw_run = newest(raw_root)
    bundle = adapt(raw_run, args, log)
    print(json.dumps({"run_id": str(bundle), "returncode": proc.returncode,
                      "raw_run": str(raw_run)}, indent=2), flush=True)
    if proc.returncode:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
