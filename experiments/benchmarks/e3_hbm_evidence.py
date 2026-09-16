#!/usr/bin/env python3
"""E3 evidence adapter and runner for real HBM snapshot slots.

The original A9 runner already exercises the real MindSpore HBM snapshot,
single-owner SPDK writer and 83.0.0 read-back path.  This wrapper adds the
uniform PPT evidence contract and process/NPU memory telemetry without
changing the storage path.  It intentionally reports host pinned memory as
``None`` when the kernel does not expose VmPin; a derived slot size is never
labelled as an observed RSS value.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from ppt_evidence import EvidenceBundle, environment_snapshot, stats  # noqa: E402


def _status(pid: int) -> dict:
    result = {}
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return result
    state = re.search(r"^State:\s+([^\n]+)$", text, re.MULTILINE)
    if state:
        result["state"] = state.group(1).strip()
    for key in ("VmRSS", "VmPin", "VmLck", "RssAnon", "RssFile"):
        match = re.search(rf"^{key}:\s+(\d+)\s+kB$", text, re.MULTILINE)
        if match:
            result[f"{key.lower()}_bytes"] = int(match.group(1)) * 1024
    return result


def _npu_hbm(npu: int) -> int | None:
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True,
                             text=True, check=False, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = out.splitlines()
    for index, line in enumerate(lines):
        if re.search(rf"\|\s*{npu}\s+910B3\s+\|", line):
            # npu-smi prints the HBM pair on the following chip/bus line.
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            values = re.findall(r"(\d+)\s*/\s*\d+", next_line)
            if values:
                return int(values[-1]) * 1024 * 1024
    return None


def monitor(pid: int, npu: int, interval_s: float = 0.5) -> dict:
    samples = []
    last_hbm_ns = 0
    last_hbm = None
    while True:
        row = {"monotonic_ns": time.monotonic_ns(), **_status(pid)}
        if str(row.get("state", "")).startswith("Z"):
            break
        # npu-smi is a relatively slow CLI on this host.  Poll it sparsely
        # while retaining process-level telemetry at the finer interval.
        now = row["monotonic_ns"]
        if now - last_hbm_ns >= 5_000_000_000:
            last_hbm = _npu_hbm(npu)
            last_hbm_ns = now
        if last_hbm is not None:
            row["hbm_bytes"] = last_hbm
        samples.append(row)
        if not Path(f"/proc/{pid}").exists():
            break
        time.sleep(interval_s)
    def peak(key):
        values = [int(item[key]) for item in samples if key in item]
        return max(values) if values else None
    return {
        "samples": samples,
        "peak": {key: peak(key) for key in
                 ("vmrss_bytes", "vmpin_bytes", "vmlck_bytes",
                  "rssanon_bytes", "rssfile_bytes", "hbm_bytes")},
    }


def _newest_run(root: Path) -> Path:
    runs = sorted(root.glob("A9_HBM_*/result.json"),
                  key=lambda path: path.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"no A9 result under {root}")
    return runs[-1].parent


def adapt_run(raw_run: Path, args, telemetry: dict, command_result: dict,
              measurement_kind="new real-HBM run", status_override=None) -> Path:
    result = json.loads((raw_run / "result.json").read_text())
    config = json.loads((raw_run / "config.json").read_text())
    samples_path = raw_run / "samples.jsonl"
    old_samples = ([json.loads(line) for line in samples_path.read_text().splitlines()
                    if line.strip()] if samples_path.exists() else [])
    formal = [sample for sample in old_samples if not sample.get("warmup")]
    env = {}
    env_path = raw_run / "environment.json"
    if env_path.exists():
        env = json.loads(env_path.read_text())
    # Keep old output auditable inside the new bundle.
    bundle = EvidenceBundle("E3", {
        "model": "gpt2_xl",
        "seed": config.get("seed"),
        "mode": "real_hbm_snapshot_slot",
        "storage": "single-owner SPDK async qpair + flush/metadata commit",
        "pci": config.get("pci", "0000:83:00.0"),
        "npu": config.get("npu", args.npu),
        "numa": None,
        "slot_count": config.get("slot_count", args.slots),
        "chunk_size": config.get("chunk_size", 4 * 1024 * 1024),
        "pipeline_depth": config.get("pipeline_depth", 4),
        "warmups": config.get("warmups"),
        "formal_samples": len(formal),
        "measurement_kind": measurement_kind,
        "memory_scope": {
            "host_staging": "not allocated; HBM slot lifecycle",
            "observed": "process VmRSS/VmPin/VmLck and npu-smi HBM",
            "missing": "pinned DRAM is None if VmPin is unavailable",
        },
    }, root=args.evidence_root, repo_root=REPO_ROOT,
    environment=env or environment_snapshot(
        pci=config.get("pci", "0000:83:00.0"), npu=args.npu))
    shutil.copytree(raw_run, bundle.raw_dir / "a9_run")
    stdout_path = command_result.get("stdout_path")
    if stdout_path and Path(stdout_path).exists():
        shutil.copy2(stdout_path, bundle.raw_dir / "child.stdout.log")
    (bundle.raw_dir / "command.json").write_text(
        json.dumps(command_result, indent=2, sort_keys=True) + "\n")
    (bundle.raw_dir / "telemetry.json").write_text(
        json.dumps(telemetry, indent=2, sort_keys=True) + "\n")

    hbm_peak = telemetry["peak"].get("hbm_bytes")
    rss_peak = telemetry["peak"].get("vmrss_bytes")
    pinned_peak = telemetry["peak"].get("vmpin_bytes")
    for old in formal:
        sample = dict(old)
        sample.update({
            "experiment": "E3",
            "mode": "real_hbm_snapshot_slot",
            "state_bytes": old.get("bytes"),
            "logical_bytes": old.get("bytes"),
            "physical_bytes": None,
            "host_rss_peak_bytes": rss_peak,
            "pinned_dram_peak_bytes": pinned_peak,
            "hbm_peak_bytes": hbm_peak,
            "memory_measurement_scope": "run-level peak telemetry",
        })
        bundle.add_sample(sample)
    e2e = [s["timeline_us"]["end_to_end"] / 1000 for s in formal]
    waits = [s["slot_wait_ms"] for s in formal]
    payload = formal[0].get("bytes") if formal else None
    metrics = {
        "model": "gpt2_xl", "seed": config.get("seed"),
        "mode": "real_hbm_snapshot_slot",
        "state_bytes": payload, "logical_bytes": payload,
        "hbm_slot_bytes": result.get("summary", {}).get("hbm_bytes_per_slot"),
        "physical_bytes": None, "chunk_size": config.get("chunk_size"),
        "pipeline_depth": config.get("pipeline_depth"),
        "slot_count": config.get("slot_count"),
        "latency_mean": stats(e2e), "latency_p50": stats(e2e).get("median"),
        "latency_p95": stats(e2e).get("p95"),
        "throughput": stats([payload / (value / 1000) / (1024 ** 2)
                              for value in e2e]) if payload else None,
        "foreground_wait": stats(waits), "step_overhead": None,
        "host_rss_peak": rss_peak, "pinned_dram_peak": pinned_peak,
        "hbm_peak": hbm_peak, "pcie_bytes": None, "nvme_bytes": None,
        "recovery_error": 0 if result.get("status") == "pass" else None,
        "loss_deviation": None,
        "fault_results": {"readback_sha256": "pass" if all(
            s.get("frozen_sha256") == s.get("readback_sha256") for s in formal)
            else "fail"},
        "telemetry_scope": "peak over one subprocess; HBM includes baseline",
        "source_a9_run": str(raw_run),
        "source_result_summary": result.get("summary", {}),
    }
    status = status_override or ("pass" if len(formal) >= args.required_samples
                                 and not result.get("failed_samples") else "fail")
    bundle.finalize(metrics=metrics, status=status)
    return bundle.run_dir


def run_one(args, slots: int) -> Path:
    raw_root = Path(args.raw_root) / f"slots_{slots}"
    raw_root.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(REPO_ROOT / "experiments/benchmarks/a9_hbm_slots.py"),
           "--pci", "0000:83:00.0", "--npu", str(args.npu),
           "--slots", str(slots), "--steps", str(args.steps),
           "--ckpt-every", str(args.ckpt_every),
           "--warmups", str(args.warmups),
           "--pipeline-depth", str(args.pipeline_depth),
           "--chunk-size", str(args.chunk_size),
           "--shm-id", str(args.shm_id + slots),
           "--io-timeout-s", str(args.io_timeout_s),
           "--output-root", str(raw_root)]
    if args.io_delay_ms:
        cmd.extend(["--io-delay-ms", str(args.io_delay_ms)])
    start = time.monotonic_ns()
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    child_log_path = raw_root / "child.stdout.log"
    child_log = child_log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            env=child_env, start_new_session=True)
    # Drain the child continuously in a small relay thread.  A file-backed
    # descriptor alone would avoid pipe backpressure, but retaining a pipe
    # lets us return the complete output in the command result.
    import threading
    def relay():
        for line in proc.stdout:
            child_log.write(line)
            child_log.flush()
    relay_thread = threading.Thread(target=relay, daemon=True)
    relay_thread.start()
    try:
        telemetry = monitor(proc.pid, args.npu)
    except BaseException:
        # Do not leave an NPU process or forkserver behind when the wrapper
        # is interrupted while the child is compiling its first graph.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        raise
    proc.wait()
    relay_thread.join(timeout=10)
    child_log.close()
    output = child_log_path.read_text(encoding="utf-8", errors="replace")
    print(output, end="", flush=True)
    command_result = {"argv": cmd, "returncode": proc.returncode,
                      "duration_s": (time.monotonic_ns() - start) / 1e9,
                      "stdout": output,
                      "stdout_path": str(child_log_path)}
    raw_run = _newest_run(raw_root)
    bundle_dir = adapt_run(raw_run, args, telemetry, command_result)
    if proc.returncode != 0:
        raise RuntimeError(f"A9 slot run failed: {proc.returncode}")
    print(json.dumps({"slots": slots, "bundle": str(bundle_dir),
                      "telemetry_peak": telemetry["peak"]}, sort_keys=True),
          flush=True)
    return bundle_dir


def import_legacy(args):
    """Import the completed A9 HBM matrix without relabelling it as new."""
    legacy_root = REPO_ROOT / "results/wp2-20260826/a9_hbm"
    targets = sorted(legacy_root.glob("formal/A9_HBM_*/result.json"))
    targets += sorted(legacy_root.glob("slow/A9_HBM_*/result.json"))
    imported = []
    for result_path in targets:
        raw_run = result_path.parent
        config = json.loads((raw_run / "config.json").read_text())
        delay = float(config.get("io_delay_ms", 0.0))
        telemetry = {"samples": [], "peak": {key: None for key in
                    ("vmrss_bytes", "vmpin_bytes", "vmlck_bytes",
                     "rssanon_bytes", "rssfile_bytes", "hbm_bytes")}}
        imported.append(str(adapt_run(
            raw_run, args, telemetry,
            {"argv": ["legacy-import", str(raw_run)], "returncode": 0,
             "source": "completed A9 result"},
            measurement_kind=("historical A9 real-HBM run; 5s delay" if delay
                               else "historical A9 real-HBM run"),
            status_override="historical_pass_insufficient_samples")))
    print(json.dumps({"imported": imported, "count": len(imported)},
                     indent=2, sort_keys=True), flush=True)


def record_failed_attempt(args):
    """Persist an interrupted run as a failure, never as a zero sample."""
    attempt = Path(args.failed_path)
    configs = sorted(attempt.glob("A9_HBM_*/config.json"),
                     key=lambda path: path.stat().st_mtime)
    config = json.loads(configs[-1].read_text()) if configs else {}
    env_path = attempt / (configs[-1].parent.name if configs else "") / "environment.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    bundle = EvidenceBundle("E3", {
        "model": "gpt2_xl", "seed": config.get("seed"),
        "mode": "real_hbm_snapshot_slot", "pci": "0000:83:00.0",
        "npu": config.get("npu", args.npu),
        "slot_count": config.get("slot_count", 1),
        "chunk_size": config.get("chunk_size", 4 * 1024 * 1024),
        "pipeline_depth": config.get("pipeline_depth", 4),
        "measurement_kind": "failed new run",
        "failure_phase": args.failure_phase,
    }, root=args.evidence_root, repo_root=REPO_ROOT, environment=env)
    shutil.copytree(attempt, bundle.raw_dir / "attempt")
    bundle.add_failure({"phase": args.failure_phase,
                        "reason": args.failure_reason,
                        "source_attempt": str(attempt),
                        "samples_excluded": True})
    bundle.finalize(metrics={"model": "gpt2_xl", "mode": "real_hbm_snapshot_slot",
                             "slot_count": config.get("slot_count", 1),
                             "host_rss_peak": None, "pinned_dram_peak": None,
                             "hbm_peak": None}, status="fail")
    print(json.dumps({"failed_bundle": str(bundle.run_dir)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npu", type=int, default=7)
    parser.add_argument("--slots", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--ckpt-every", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--pipeline-depth", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--shm-id", type=int, default=3000)
    parser.add_argument("--io-timeout-s", type=float, default=120.0)
    parser.add_argument("--io-delay-ms", type=float, default=0.0)
    parser.add_argument("--required-samples", type=int, default=30)
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--evidence-root", default=None)
    parser.add_argument("--import-legacy", action="store_true")
    parser.add_argument("--failed-path", default=None)
    parser.add_argument("--failure-phase", default="driver_sync")
    parser.add_argument("--failure-reason", default="no checkpoint before observation timeout")
    args = parser.parse_args()
    args.raw_root = args.raw_root or str(REPO_ROOT / "results" /
                                          "ppt-evidence-20260829/E3/raw-a9")
    args.evidence_root = args.evidence_root or str(REPO_ROOT / "results" /
                                                    "ppt-evidence-20260829")
    if args.import_legacy:
        import_legacy(args)
        return
    if args.failed_path:
        record_failed_attempt(args)
        return
    for slots in args.slots:
        run_one(args, slots)


if __name__ == "__main__":
    main()
