#!/usr/bin/env python3
"""§13 — PCcheck Concurrent Multi-Checkpoint Pipeline (SPDK path → NVMe #1).

Extends CheckFreq two-phase overlap with N concurrent checkpoint slots.
Each slot has its own HBM snapshot buffer + independent SPDK persist thread.
When all slots are busy, the next checkpoint blocks until one frees.

Usage:
  sudo python experiments/baselines/pccheck_spdk.py --device-id 1 --steps 30 --concurrency 3
"""

import os, sys, time, json, argparse, threading, queue
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, REPO)

import mindspore as ms
from experiments.common import make_gpt2xl_training, init_env, warmup_model

OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
CHUNK_SIZE = 4 * 1024 * 1024


class PCCheckConcurrent:
    """PCcheck concurrent multi-checkpoint pipeline using SPDK DMA.

    Maintains N independent slot states, each going through:
      IDLE → SNAPSHOTTING → PERSISTING → DONE → IDLE

    The free_slots queue ensures at most N in-flight checkpoints.
    When a slot's persist completes, it returns to free_slots.
    """

    def __init__(self, device_id=1, concurrency=3,
                 nvme_addr="0000:83:00.0"):
        self.concurrency = concurrency
        self.device_id = device_id
        self.free_slots = queue.Queue(maxsize=concurrency)
        for i in range(concurrency):
            self.free_slots.put(i)

        from direct_checkpoint import DirectCheckpoint

        # Each slot gets its own DirectCheckpoint with different NVMe slot offsets
        self.slots = []
        for i in range(concurrency):
            ckpt = DirectCheckpoint(
                nvme_addr=nvme_addr, npu_device_id=device_id,
                pipeline_depth=8, requested_chunk_size=CHUNK_SIZE,
                keep_last_n=concurrency + 2, slot_size_gb=10)
            self.slots.append({
                "id": i,
                "ckpt": ckpt,
                "state": "IDLE",
            })

        self._lock = threading.Lock()
        self.snapshot_times_ms = []
        self.persist_times_ms = []
        self.blocking_times_ms = []
        self.ckpt_events = []
        self._persist_errors = []

    def checkpoint(self, model, step):
        """Non-blocking checkpoint: acquire a slot, snapshot, launch persist.

        Returns (snapshot_ms, blocking_ms).
        blocking_ms > 0 means we waited for a free slot.
        """
        # Wait for a free slot (blocks if all N are in-flight)
        t_wait0 = time.perf_counter()
        slot_id = self.free_slots.get()
        blocking_ms = (time.perf_counter() - t_wait0) * 1000.0
        if blocking_ms > 0.1:
            self.blocking_times_ms.append(blocking_ms)

        slot = self.slots[slot_id]
        ms.hal.synchronize()

        t0 = time.perf_counter()
        slot["ckpt"].save(model, step=step,
                         meta_path=f"/tmp/pccheck_slot{slot_id}_step{step}.pkl",
                         commit_meta=False)
        snapshot_ms = (time.perf_counter() - t0) * 1000.0
        self.snapshot_times_ms.append(snapshot_ms)
        slot["state"] = "PERSISTING"

        event = {
            "step":        step,
            "slot_id":     slot_id,
            "snapshot_ms": round(snapshot_ms, 3),
            "blocking_ms": round(blocking_ms, 3),
            "persist_ms":  0.0,
        }
        self.ckpt_events.append(event)

        # Launch background thread to wait for persist and return slot
        t = threading.Thread(
            target=self._on_persist_done, args=(slot, slot_id, step))
        t.daemon = True
        t.start()
        return snapshot_ms, blocking_ms

    def _on_persist_done(self, slot, slot_id, step):
        """Wait for SPDK persist, then return slot to free pool."""
        try:
            t0 = time.perf_counter()
            slot["ckpt"].wait_for_io_completion()
            persist_ms = (time.perf_counter() - t0) * 1000.0
            self.persist_times_ms.append(persist_ms)

            # Retroactively fill persist_ms.
            with self._lock:
                for ev in reversed(self.ckpt_events):
                    if ev["step"] == step and ev.get("slot_id") == slot_id:
                        ev["persist_ms"] = round(persist_ms, 3)
                        break
        except BaseException as exc:
            with self._lock:
                self._persist_errors.append(exc)
        finally:
            slot["state"] = "IDLE"
            self.free_slots.put(slot_id)

    def wait_all(self):
        """Wait for ALL in-flight checkpoints to complete."""
        acquired = [self.free_slots.get() for _ in range(self.concurrency)]
        for slot_id in acquired:
            self.free_slots.put(slot_id)
        if self._persist_errors:
            raise RuntimeError("background SPDK persist failed") from self._persist_errors[0]

    def cleanup(self):
        self.wait_all()
        for slot in self.slots:
            slot["ckpt"].cleanup()

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
            "concurrency":  self.concurrency,
            "snapshot_ms":  _stats(self.snapshot_times_ms),
            "persist_ms":   _stats(self.persist_times_ms),
            "blocking_ms":  _stats(self.blocking_times_ms),
        }


def run_pccheck(device_id=1, steps=30, ckpt_every=5, concurrency=3,
                nvme_addr="0000:83:00.0"):
    print("=" * 60)
    print(f"[B3] PCcheck Concurrent — SPDK Path (NVMe #1)")
    print(f"  Concurrency: {concurrency}")
    print(f"  Steps: {steps}, ckpt every: {ckpt_every}")
    print("=" * 60)

    init_env(device_id=device_id)
    model, ds, opt = make_gpt2xl_training(total_steps=steps, device_id=device_id)
    warmup_model(model, opt, ds)

    total_bytes = sum(int(p.size) * np.dtype(ms.dtype_to_nptype(p.dtype)).itemsize
                      for p in model.trainable_params())
    print(f"\n  Total params: {total_bytes / 1e9:.2f} GB")

    from direct_checkpoint import ProbeTrainOneStepCell
    cell = ProbeTrainOneStepCell(model, opt, enable_probe=False, ckpt_interval=9999)

    engine = PCCheckConcurrent(device_id=device_id, concurrency=concurrency,
                               nvme_addr=nvme_addr)

    it = ds.create_tuple_iterator()
    _ = cell(*next(it))  # compile

    step_times = []
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

        if s % ckpt_every == 0:
            snap_ms, block_ms = engine.checkpoint(model, s)
            print(f"  Step {s:3d}: dt={dt_ms:.1f}ms  snap={snap_ms:.1f}ms  "
                  f"block={block_ms:.1f}ms")

        step_times.append(dt_ms)

    engine.cleanup()
    total_time = (time.perf_counter() - t_start) * 1000

    # Skip first step (compilation)
    step_times = step_times[1:]

    stats = engine.get_stats()
    results = {
        "experiment": "pccheck_spdk",
        "method": f"PCcheck Concurrent (SPDK DMA → NVMe #1, N={concurrency})",
        "total_bytes": total_bytes,
        "steps": steps,
        "ckpt_every": ckpt_every,
        "concurrency": concurrency,
        "nvme_addr": nvme_addr,
        "step_mean_ms": float(np.mean(step_times)),
        "step_std_ms": float(np.std(step_times)),
        "total_time_ms": total_time,
        "timing": stats,
        "ckpt_events": engine.ckpt_events,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"pccheck_n{concurrency}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"  Step mean: {results['step_mean_ms']:.1f}ms")
    print(f"  Snapshot mean: {stats['snapshot_ms']['mean']:.1f}ms")
    print(f"  Persist mean: {stats['persist_ms']['mean']:.1f}ms")
    print(f"  Blocking mean: {stats['blocking_ms']['mean']:.1f}ms  "
          f"(n={stats['blocking_ms']['n']})")
    return results


def main():
    parser = argparse.ArgumentParser(description="PCcheck Concurrent SPDK")
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--ckpt-every", type=int, default=5)
    parser.add_argument("--concurrency", "-n", type=int, default=3)
    parser.add_argument("--pci-addr", default="0000:83:00.0")
    args = parser.parse_args()
    run_pccheck(device_id=args.device_id, steps=args.steps,
                ckpt_every=args.ckpt_every, concurrency=args.concurrency,
                nvme_addr=args.pci_addr)


if __name__ == "__main__":
    main()
