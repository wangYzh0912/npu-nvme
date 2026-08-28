"""Shared evidence bundle and statistics helpers for PPT experiments.

The module is intentionally dependency-light so host-only and hardware
benchmarks use the same result contract.  Missing measurements are represented
by ``None`` and an explanation in the caller's config; they are never silently
converted to zero.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import time
import uuid
from pathlib import Path


RESULT_FIELDS = (
    "status", "model", "seed", "mode", "state_bytes", "logical_bytes",
    "physical_bytes", "chunk_size", "pipeline_depth", "slot_count",
    "latency_mean", "latency_p50", "latency_p95", "throughput",
    "foreground_wait", "step_overhead", "host_rss_peak",
    "pinned_dram_peak", "hbm_peak", "cube_util", "vector_util",
    "hbm_bandwidth", "pcie_bytes", "nvme_bytes", "recovery_error",
    "loss_deviation", "fault_results",
)


def command(argv, timeout=10):
    """Return a serialisable command result without raising on diagnostics."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                check=False, timeout=timeout)
        return {"argv": list(argv), "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": list(argv), "returncode": -1, "stdout": "",
                "stderr": repr(error)}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values):
    """Calculate the common descriptive statistics for successful samples.

    The CI uses a two-sided Student-t critical value when scipy is available;
    the fallback values are conservative for the small sample sizes used by
    the existing evidence.  P99 is intentionally absent for n < 30.
    """
    values = [float(value) for value in values]
    if not values:
        return {"n": 0, "mean": None, "median": None, "stdev": None,
                "ci95": None, "p95": None, "p99": None,
                "p99_status": "not reported (n=0)"}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    if len(values) > 1:
        try:
            from scipy.stats import t as student_t
            critical = float(student_t.ppf(0.975, len(values) - 1))
            method = f"student-t(df={len(values) - 1})"
        except ImportError:
            critical = (2.262 if len(values) <= 10 else
                        2.045 if len(values) <= 30 else 1.96)
            method = "student-t conservative fallback"
        margin = critical * stdev / math.sqrt(len(values))
    else:
        margin = 0.0
        method = "single-sample descriptive interval"
    ordered = sorted(values)

    def percentile(percent):
        position = (len(ordered) - 1) * percent / 100.0
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "n": len(values), "mean": mean, "median": statistics.median(values),
        "stdev": stdev, "ci95": [mean - margin, mean + margin],
        "ci95_method": method, "p95": percentile(95),
        "p99": percentile(99) if len(values) >= 30 else None,
        "p99_status": "reported" if len(values) >= 30 else
                      f"not reported (n={len(values)}<30)",
        "min": min(values), "max": max(values),
    }


def _first_existing(paths):
    for path in paths:
        if Path(path).is_file():
            return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    return None


def environment_snapshot(*, pci=None, npu=None, numa=None, repo_root=None,
                         spdk_root=None, npu_info=None):
    """Capture reproducibility metadata used by every new experiment."""
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
    spdk_root = Path(spdk_root or repo_root / "third_party" / "spdk")
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "software": {
            "python": platform.python_version(),
            "mindspore": package_version("mindspore"),
            "mindformers": package_version("mindformers"),
            "numpy": package_version("numpy"),
            "cann": _first_existing((
                "/usr/local/Ascend/ascend-toolkit/latest/version.info",
                "/usr/local/Ascend/ascend-toolkit/8.0.RC3/runtime/version.info",
                "/usr/local/Ascend/version.info")),
            "compiler": command(["cc", "--version"]),
        },
        "repo": {
            "root": str(repo_root),
            "commit": command(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
            "branch": command(["git", "-C", str(repo_root), "branch", "--show-current"]),
            "status": command(["git", "-C", str(repo_root), "status", "--porcelain"]),
            "spdk_commit": command(["git", "-C", str(spdk_root), "rev-parse", "HEAD"]),
        },
        "hardware": {
            "npu": npu,
            "numa": numa,
            "target_pci": pci,
            "target_pci_info": command(["lspci", "-s", str(pci), "-nnk"])
            if pci else None,
            "npu_smi_before_init": npu_info,
            "kernel": platform.uname()._asdict(),
            "numactl": command(["numactl", "-H"]),
        },
    }


class EvidenceBundle:
    """Write one self-contained experiment run."""

    def __init__(self, experiment_id, config, *, root=None, repo_root=None,
                 environment=None):
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[1])
        root = Path(root or self.repo_root / "results" /
                    "ppt-evidence-20260829")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_id}_{stamp}_{uuid.uuid4().hex[:8]}"
        self.run_dir = root / experiment_id / self.run_id
        self.raw_dir = self.run_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=False)
        self.samples_path = self.run_dir / "samples.jsonl"
        self.timeline_path = self.run_dir / "timeline.jsonl"
        self.failures_path = self.run_dir / "failures.jsonl"
        self.samples = []
        self.failures = []
        self.config = dict(config)
        self.config.update({"experiment_id": experiment_id,
                            "run_id": self.run_id})
        self._write("config.json", self.config)
        self._write("commit.json", self._commit_snapshot())
        # Keep the file present even for host-only/import runs.  An empty
        # object is preferable to a broken manifest path; hardware runners
        # should pass environment_snapshot(...).
        self._write("environment.json", environment or {})
        self.failures_path.touch()

    def _commit_snapshot(self):
        root = self.repo_root
        spdk = root / "third_party" / "spdk"
        return {
            "repo": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
            "branch": command(["git", "-C", str(root), "branch", "--show-current"]),
            "status": command(["git", "-C", str(root), "status", "--porcelain"]),
            "spdk": command(["git", "-C", str(spdk), "rev-parse", "HEAD"]),
        }

    def _write(self, name, value):
        path = self.run_dir / name
        path.write_text(json.dumps(value, indent=2, sort_keys=True,
                                   default=str) + "\n", encoding="utf-8")

    def add_sample(self, sample, events=None):
        record = dict(sample)
        record.setdefault("run_id", self.run_id)
        self.samples.append(record)
        with self.samples_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        event_record = {"run_id": self.run_id,
                        "request_id": record.get("request_id"),
                        "events": events if events is not None else
                                  record.get("events", [])}
        with self.timeline_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event_record, sort_keys=True,
                                    default=str) + "\n")

    def add_failure(self, failure):
        record = dict(failure)
        record.setdefault("run_id", self.run_id)
        self.failures.append(record)
        with self.failures_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def finalize(self, *, metrics=None, status=None):
        metrics = dict(metrics or {})
        result = {field: metrics.pop(field, None) for field in RESULT_FIELDS}
        result.update(metrics)
        result.update({
            "status": status or result.get("status") or
                      ("pass" if not self.failures else "fail"),
            "run_id": self.run_id,
            "samples": len(self.samples),
            "failed_samples": len(self.failures),
            "sample_policy": "failed samples excluded from statistics",
            "paths": {"config": "config.json", "environment": "environment.json",
                      "commit": "commit.json", "samples": "samples.jsonl",
                      "timeline": "timeline.jsonl", "failures": "failures.jsonl",
                      "raw": "raw/", "result": "result.json"},
        })
        self._write("result.json", result)
        return result


__all__ = ["EvidenceBundle", "RESULT_FIELDS", "command",
           "environment_snapshot", "sha256_file", "stats"]
