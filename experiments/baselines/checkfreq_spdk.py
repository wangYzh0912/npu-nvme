#!/usr/bin/env python3
"""§13 — CheckFreq Two-Phase + SPDK persist (NVMe #1).

Same two-phase overlap as CheckFreq but persist uses SPDK DMA
instead of kernel FS I/O.  D2D snapshot (fast) + SPDK background write.

Usage:
  sudo python experiments/baselines/checkfreq_spdk.py --device-id 1 --steps 30
"""

import os, sys, time, json, argparse, threading, ctypes
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, REPO)

import mindspore as ms
from experiments.common import make_gpt2xl_training, init_env, warmup_model

OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
CHUNK_SIZE = 4 * 1024 * 1024


class CheckFreqTwoPhaseSPDK:
    """CheckFreq two-phase checkpoint with SPDK DMA persist → NVMe #1.

    Snapshot:  D2D copy (HBM→HBM) via aclrtMemcpy — fast (~5ms)
    Persist:   SPDK DMA from snapshot buffer → NVMe #1 — user-space

    Uses DirectCheckpoint for the SPDK I/O layer.
    """

    def __init__(self, device_id=1):
        from direct_checkpoint import DirectCheckpoint
        self.ckpt = DirectCheckpoint(
            nvme_addr="0000:83:00.0", npu_device_id=device_id,
            pipeline_depth=8, requested_chunk_size=CHUNK_SIZE,
            keep_last_n=3, slot_size_gb=50)
        self._lock = threading.Lock()
        self.in_flight = False
        self.snapshot_times_ms = []
        self.blocking_times_ms = []
        self.ckpt_events = []

    def checkpoint(self, model, step):
        """Two-phase checkpoint with D2D snapshot + SPDK persist.

        Phase 1 (blocking): sync + D2D snapshot via DirectCheckpoint.save()
        Phase 2 (background): SPDK DMA write from snapshot buffers
        """
        blocking_ms = 0.0
        with self._lock:
            if self.in_flight:
                t_wait = time.perf_counter()
                self.ckpt.wait_for_io_completion()
                blocking_ms = (time.perf_counter() - t_wait) * 1000.0
                self.blocking_times_ms.append(blocking_ms)
                self.in_flight = False

        ms.hal.synchronize()
        t0 = time.perf_counter()
        self.ckpt.save(model, step=step,
                       meta_path=f"/tmp/checkfreq_spdk_step{step}.pkl",
                       commit_meta=False)
        snapshot_ms = (time.perf_counter() - t0) * 1000.0
        self.snapshot_times_ms.append(snapshot_ms)
        self.in_flight = True

        event = {
            "step": step,
            "snapshot_ms": round(snapshot_ms, 3),
            "blocking_ms": round(blocking_ms, 3),
        }
        self.ckpt_events.append(event)
        return snapshot_ms, blocking_ms

    def wait_all(self):
        self.ckpt.wait_for_io_completion()
        self.in_flight = False

    def cleanup(self):
        self.wait_all()
        self.ckpt.cleanup()

    def get_stats(self):
        def _stats(lst):
            if not lst:
                return {"mean": 0, "std": 0, "max": 0, "n": 0}
            a = np.array(lst)
            return {
                "mean": round(float(np.mean(a)), 3),
                "std":  round(float(np.std(a, ddof=1)) if len(a) > 1 else 0.0, 3),
                "max":  round(float(np.max(a)), 3),
                "n":    len(a),
            }
        return {
            "snapshot_ms": _stats(self.snapshot_times_ms),
            "blocking_ms": _stats(self.blocking_times_ms),
        }


def run_checkfreq_spdk(device_id=1, steps=30, ckpt_every=10):
    print("=" * 60)
    print("[B1/B2] CheckFreq Two-Phase — SPDK Path (NVMe #1)")
    print(f"  Snapshot: D2D copy → HBM snapshot buffer")
    print(f"  Persist:  SPDK DMA → NVMe #1 (0000:83:00.0)")
    print("=" * 60)

    init_env(device_id=device_id)
    model, ds, opt = make_gpt2xl_training(total_steps=steps, device_id=device_id)
    warmup_model(model, opt, ds)

    total_bytes = sum(int(p.size) * np.dtype(ms.dtype_to_nptype(p.dtype)).itemsize
                      for p in model.trainable_params())
    print(f"\n  Total params: {total_bytes / 1e9:.2f} GB")

    from direct_checkpoint import ProbeTrainOneStepCell
    cell = ProbeTrainOneStepCell(model, opt, enable_probe=False, ckpt_interval=9999)

    engine = CheckFreqTwoPhaseSPDK(device_id=device_id)

    it = ds.create_tuple_iterator()
    _ = cell(*next(it))

    step_times = []
    ckpt_events = []
    t_start = time.perf_counter()

    for s in range(1, steps + 1):
        try:
            data = next(it)
        except StopIteration:
            it = ds.create_tuple_iterator()
            data = next(it)

        t0 = time.perf_counter()
        loss = cell(*data)
        dt_ms = (time.perf_counter() - t0) * 1000

        event = {"step": s, "dt_ms": dt_ms}
        if s % ckpt_every == 0:
            snap_ms, block_ms = engine.checkpoint(model, s)
            event["snapshot_ms"] = snap_ms
            print(f"  Step {s:3d}: dt={dt_ms:.1f}ms  snap={snap_ms:.1f}ms")

        step_times.append(dt_ms)
        ckpt_events.append(event)

    engine.cleanup()
    total_time = (time.perf_counter() - t_start) * 1000

    stats = engine.get_stats()
    results = {
        "experiment": "checkfreq_spdk",
        "method": "CheckFreq Two-Phase (SPDK DMA → NVMe #1)",
        "total_bytes": total_bytes,
        "steps": steps,
        "ckpt_every": ckpt_every,
        "concurrency": None,
        "step_mean_ms": float(np.mean(step_times)),
        "step_std_ms": float(np.std(step_times)),
        "total_time_ms": total_time,
        "timing": stats,
        "ckpt_events": engine.ckpt_events,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "checkfreq_spdk.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"  Step mean: {results['step_mean_ms']:.1f}ms")
    print(f"  Snapshot mean: {stats['snapshot_ms']['mean']:.1f}ms")
    print(f"  Blocking mean: {stats['blocking_ms']['mean']:.1f}ms  "
          f"(n={stats['blocking_ms']['n']})")
    return results


def main():
    parser = argparse.ArgumentParser(description="CheckFreq Two-Phase SPDK")
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--ckpt-every", type=int, default=10)
    args = parser.parse_args()
    run_checkfreq_spdk(device_id=args.device_id, steps=args.steps,
                       ckpt_every=args.ckpt_every)


if __name__ == "__main__":
    main()
