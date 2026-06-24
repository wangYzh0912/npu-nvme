"""Unified profiling module for NPU-NVMe SPDK benchmarks.

Provides timing, bandwidth calculation, C-layer CSV ingestion, per-step
latency tracking, console summaries, and structured JSON export.

Usage:
    from profiler import SpdkProfiler

    prof = SpdkProfiler("spdk_bench", output_dir="output/")
    prof.phase("full_write")
    rc = prof.wrap_c_call("write_batch", lib.npu_nvme_write_batch, ctx, ...)
    prof.ingest_csv("output/profiling/time_write.csv", direction="write")
    prof.end_phase("full_write", total_bytes=2900000000)
    prof.to_json()
"""

import csv
import json
import os
import threading
import time
from typing import Dict, List, Optional


class SpdkProfiler:
    """Collects and reports SPDK I/O and training performance metrics.

    Thread-safe: phase boundaries are protected by a lock so that the
    background I/O thread in DirectCheckpoint.save() can call end_phase()
    without racing with the main thread.
    """

    def __init__(self, label: str = "spdk_bench", output_dir: str = "output/"):
        self.label = label
        self.output_dir = output_dir
        self._lock = threading.Lock()
        self._reset()

    def _reset(self):
        self.created_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.phases: Dict[str, dict] = {}       # phase_name → {metrics}
        self.step_latencies: List[float] = []    # per-step elapsed (seconds)
        self._current_phase: Optional[str] = None
        self._phase_start: float = 0.0
        self._events: List[dict] = []            # lightweight event log
        self._config: dict = {}

    # -- Configuration ----------------------------------------------------------

    def set_config(self, **kwargs):
        """Record benchmark configuration (model, pipeline_depth, etc.)."""
        self._config.update(kwargs)

    # -- Phase lifecycle --------------------------------------------------------

    def phase(self, name: str):
        """Begin a named timing phase."""
        with self._lock:
            self._current_phase = name
            self._phase_start = time.perf_counter()
            if name not in self.phases:
                self.phases[name] = {}

    def end_phase(self, name: str, **metrics):
        """End the current phase and record summary metrics.

        Args:
            name: must match the current phase name.
            **metrics: key=value pairs to store (e.g. total_bytes=...).
        """
        with self._lock:
            elapsed = time.perf_counter() - self._phase_start
            if name not in self.phases:
                self.phases[name] = {}
            self.phases[name]["elapsed_ms"] = round(elapsed * 1000, 2)
            self.phases[name].update(metrics)
            # Auto-compute bandwidth when total_bytes is present
            if "total_bytes" in metrics and elapsed > 0:
                mb = metrics["total_bytes"] / (1024 * 1024)
                self.phases[name]["total_mb"] = round(mb, 2)
                self.phases[name]["bw_mbps"] = round(mb / elapsed, 1)
            self._current_phase = None
        self._print_phase(name)

    # -- C API wrapper ----------------------------------------------------------

    def wrap_c_call(self, label: str, c_func, *args):
        """Call *c_func(*args)*, timing the wall-clock duration.

        Returns the C function's return code unchanged.
        The elapsed time is stored in the current phase under *label*.
        """
        t0 = time.perf_counter()
        rc = c_func(*args)
        elapsed = time.perf_counter() - t0
        with self._lock:
            entry = {"elapsed_ms": round(elapsed * 1000, 2), "rc": rc}
            if self._current_phase and self._current_phase in self.phases:
                self.phases[self._current_phase][label] = entry
            else:
                # Standalone call outside a phase block
                if "_standalone" not in self.phases:
                    self.phases["_standalone"] = {}
                self.phases["_standalone"][label] = entry
        return rc

    # -- C-layer CSV ingestion --------------------------------------------------

    def ingest_csv(self, csv_path: str, direction: str = "write"):
        """Parse a C-layer time_*.csv and store per-chunk breakdown.

        Args:
            csv_path: path to time_write.csv or time_read.csv.
            direction: "write" or "read" — determines column interpretation.
        """
        if not os.path.exists(csv_path):
            return
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            return

        # Column names are identical for write and read CSVs; only the order differs.
        # csv.DictReader handles this correctly regardless of column order.
        npu_key, spdk_key = "npu_async_us", "spdk_nvme_us"
        npu_total = sum(int(r.get(npu_key, 0)) for r in rows)
        spdk_total = sum(int(r.get(spdk_key, 0)) for r in rows)
        e2e_total = sum(int(r.get("total_e2e_us", 0)) for r in rows)

        chunk_data = {
            "n_chunks": len(rows),
            "npu_dma_ms": round(npu_total / 1000, 2),
            "spdk_nvme_ms": round(spdk_total / 1000, 2),
            "e2e_sum_ms": round(e2e_total / 1000, 2),
        }
        with self._lock:
            if self._current_phase and self._current_phase in self.phases:
                self.phases[self._current_phase]["c_layer"] = chunk_data

    # -- Per-step latency -------------------------------------------------------

    def record_step(self, step_id: int, elapsed_s: float):
        """Record a single training step latency.

        Called from a MindSpore callback (per-step or per-epoch).
        """
        with self._lock:
            self.step_latencies.append(elapsed_s)
        # Lightweight console output for live monitoring
        ms = elapsed_s * 1000
        if step_id % 5 == 0 or step_id <= 2:
            print(f"  [step {step_id:>4d}]  {ms:7.1f} ms", flush=True)

    def record_event(self, event_type: str, **kwargs):
        """Log a lightweight timestamped event (e.g. listener trigger)."""
        with self._lock:
            self._events.append({
                "t": round(time.perf_counter(), 3),
                "type": event_type,
                **kwargs
            })

    # -- Console output ---------------------------------------------------------

    def _print_phase(self, name: str):
        p = self.phases.get(name, {})
        elapsed = p.get("elapsed_ms", 0)
        total_mb = p.get("total_mb")
        bw = p.get("bw_mbps")
        cl = p.get("c_layer", {})

        print(f"\n-- {name} --", flush=True)
        if total_mb is not None:
            print(f"     data: {total_mb:,.0f} MB  |  {elapsed:,.0f} ms  |  BW: {bw:,.0f} MB/s", flush=True)
        else:
            print(f"     elapsed: {elapsed:,.0f} ms", flush=True)
        if cl:
            print(f"     chunk breakdown: npu_dma={cl['npu_dma_ms']}ms  "
                  f"spdk_nvme={cl['spdk_nvme_ms']}ms  n_chunks={cl['n_chunks']}", flush=True)
        for k, v in p.items():
            if k not in ("elapsed_ms", "total_mb", "bw_mbps", "c_layer") and not k.startswith("wrap_"):
                if isinstance(v, dict):
                    print(f"     {k}: elapsed={v.get('elapsed_ms','?')}ms  rc={v.get('rc','?')}", flush=True)
                else:
                    print(f"     {k}: {v}", flush=True)

    def summary(self):
        """Print a compact summary of all phases to the console."""
        print(f"\n{'='*60}")
        print(f"  SPDK Benchmark Summary — {self.label}")
        print(f"  Started: {self.created_at}")
        print(f"{'='*60}")
        for name, p in self.phases.items():
            if name == "_standalone":
                continue
            elapsed = p.get("elapsed_ms", 0)
            total_mb = p.get("total_mb")
            bw = p.get("bw_mbps")
            cl = p.get("c_layer", {})
            if total_mb:
                npu = cl.get("npu_dma_ms", 0)
                spdk = cl.get("spdk_nvme_ms", 0)
                print(f"  {name:20s}  {total_mb:6,.0f} MB  {elapsed:6,.0f} ms  {bw:6,.0f} MB/s  "
                      f"(npu={npu}ms  spdk={spdk}ms)", flush=True)
            else:
                print(f"  {name:20s}  {'--':>6s}     {elapsed:6,.0f} ms", flush=True)

        if self.step_latencies:
            ms_vals = [s * 1000 for s in self.step_latencies]
            ms_vals.sort()
            mean = sum(ms_vals) / len(ms_vals)
            p50 = ms_vals[len(ms_vals) // 2]
            p99 = ms_vals[min(int(len(ms_vals) * 0.99), len(ms_vals) - 1)]
            print(f"  {'training_steps':20s}  n={len(ms_vals)}  mean={mean:.1f}ms  "
                  f"p50={p50:.1f}ms  p99={p99:.1f}ms", flush=True)
        print(f"{'='*60}\n", flush=True)

    # -- JSON export ------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return all collected data as a dictionary for serialisation."""
        return {
            "label": self.label,
            "created_at": self.created_at,
            "config": self._config,
            "phases": self.phases,
            "training": {
                "n_steps": len(self.step_latencies),
                "latencies_s": self.step_latencies,
            } if self.step_latencies else None,
            "events": self._events if self._events else None,
        }

    def to_json(self, filename: str = None) -> str:
        """Write results to a JSON file and return the path.

        Args:
            filename: base name (e.g. "spdk_bench.json").  If omitted,
                      defaults to "{label}.json".
        Returns:
            Absolute path to the written file.
        """
        fname = filename or f"{self.label}.json"
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, fname)
        data = self.to_dict()
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[Profiler] Results written to {path}", flush=True)
        return path
