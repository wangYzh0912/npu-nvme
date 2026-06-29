#!/usr/bin/env python3
"""§13 — PCcheck Concurrent Multi-Checkpoint (kernel FS path → NVMe #2).

Correct reproduction of PCcheck (ASPLOS'25) N-slot concurrent checkpoint:

  Snapshot (blocking):  aclrtMemcpy D2H per-parameter → host DRAM (~1059 ms)
  Persist (background):  open().write() + fsync() in background thread.

N pre-allocated host-buffer slots allow N persists to run concurrently.
When all N slots are busy, the next checkpoint blocks on free_slots.get().

Usage:
  sudo python experiments/baselines/pccheck_kernel_fs.py --device-id 1 --steps 20 -n 3
"""

import os, sys, time, json, argparse, threading, queue, gc
from dataclasses import dataclass
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, REPO)

import mindspore as ms
from experiments.common import make_gpt2xl_training, init_env, warmup_model
from experiments.baselines import two_phase_common as tpc

NVME2_DIR = "/models/nvme_baseline"
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")


@dataclass
class PCCheckSlot:
    id: int
    pinned_ptr: int = 0      # pre-allocated pinned host buffer (or 0 if numpy)
    host_buf: np.ndarray = None  # fallback numpy buffer if pinned fails
    in_use: bool = False
    persist_ms: float = 0.0


class PCCheckKernelFS:
    """PCcheck (ASPLOS'25) N-slot concurrent checkpoint using kernel FS I/O.

    N pre-allocated host buffers allow N concurrent persists.
    Training blocks only when all N slots are occupied.
    """

    def __init__(self, output_dir=NVME2_DIR, device_id=0, concurrency=3):
        self.output_dir = output_dir
        self.device_id = device_id
        self.concurrency = concurrency
        self.free_slots = queue.Queue(maxsize=concurrency)
        self.slots = []
        self._param_descs = None
        self._offset_map = None
        self._total_bytes = 0
        self._lock = threading.Lock()
        self.snapshot_ms_list = []
        self.persist_ms_list = []
        self.blocking_ms_list = []
        self.ckpt_events = []
        os.makedirs(self.output_dir, exist_ok=True)

    def init_params(self, model):
        """Allocate host buffers (pinned preferred, numpy fallback)."""
        tpc._ensure_acl_device(self.device_id)
        self._total_bytes = tpc.get_total_param_bytes(model)

        use_pinned = False
        try:
            for i in range(self.concurrency):
                pinned_ptr = tpc.allocate_pinned_host_buffer(self._total_bytes)
                self.slots.append(PCCheckSlot(id=i, pinned_ptr=pinned_ptr, host_buf=None))
            use_pinned = True
        except RuntimeError as e:
            print(f"[PCCheck] pinned alloc failed ({e}), using numpy fallback")
            for i in range(self.concurrency):
                host_buf = tpc.allocate_host_buffer(self._total_bytes)
                self.slots.append(PCCheckSlot(id=i, pinned_ptr=0, host_buf=host_buf))

        for i in range(self.concurrency):
            self.free_slots.put(i)
        self._use_pinned = use_pinned
        self._param_descs = tpc.get_param_descriptors(model)
        self._offset_map, _ = tpc.build_offset_map(self._param_descs)
        print(f"[PCCheck] {len(self._param_descs)} params, "
              f"{self._total_bytes/1e9:.2f} GB, N={self.concurrency}, "
              f"pinned={use_pinned}")

    def checkpoint(self, model, step):
        # Acquire free slot (blocks if all N busy)
        t_wait = time.perf_counter()
        slot_id = self.free_slots.get()
        blocking_ms = (time.perf_counter() - t_wait) * 1000.0
        if blocking_ms > 0.1:
            self.blocking_ms_list.append(blocking_ms)
        slot = self.slots[slot_id]
        slot.in_use = True

        # Snapshot: per-parameter D2H into slot's host buffer
        if self._use_pinned:
            snap = tpc.snapshot_d2h_pinned(
                self._param_descs, slot.pinned_ptr, self._offset_map, self.device_id)
        else:
            snap = tpc.snapshot_d2h(
                self._param_descs, slot.host_buf, self._offset_map, self.device_id)
        snapshot_ms = snap["total_ms"]
        self.snapshot_ms_list.append(snapshot_ms)

        # Launch background persist
        filepath = os.path.join(self.output_dir,
                               f"pccheck_n{self.concurrency}_step{step}_slot{slot_id}.ckpt")
        t = threading.Thread(target=self._persist_worker,
                             args=(slot_id, filepath, step), daemon=True)
        t.start()

        ev = {"step": step, "slot_id": slot_id,
              "snapshot_ms": round(snapshot_ms, 3),
              "sync_ms": snap["sync_ms"], "memcpy_ms": snap["memcpy_ms"],
              "blocking_ms": round(blocking_ms, 3), "persist_ms": 0.0}
        self.ckpt_events.append(ev)
        return ev

    def _persist_worker(self, slot_id, filepath, step):
        slot = self.slots[slot_id]
        if self._use_pinned:
            persist_ms = tpc.persist_to_file_pinned(
                slot.pinned_ptr, self._total_bytes, filepath)
        else:
            persist_ms = tpc.persist_to_file(slot.host_buf, filepath)
        slot.persist_ms = persist_ms
        self.persist_ms_list.append(persist_ms)
        with self._lock:
            for ev in reversed(self.ckpt_events):
                if ev["step"] == step and ev.get("slot_id") == slot_id:
                    ev["persist_ms"] = round(persist_ms, 3); break
        slot.in_use = False
        self.free_slots.put(slot_id)

    def wait_all(self):
        for _ in range(self.concurrency):
            self.free_slots.get()
        for i in range(self.concurrency):
            self.free_slots.put(i)

    def get_stats(self):
        def _s(lst):
            if not lst: return {"mean": 0, "std": 0, "max": 0, "n": 0}
            a = np.array(lst)
            return {"mean": round(float(np.mean(a)), 3),
                    "std": round(float(np.std(a, ddof=1)) if len(a) > 1 else 0.0, 3),
                    "max": round(float(np.max(a)), 3), "n": len(a)}
        return {"concurrency": self.concurrency,
                "snapshot_ms": _s(self.snapshot_ms_list),
                "persist_ms": _s(self.persist_ms_list),
                "blocking_ms": _s(self.blocking_ms_list)}


def run_pccheck_kernel_fs(device_id=1, steps=20, ckpt_every=5, concurrency=3):
    print("=" * 60)
    print(f"[B3] PCcheck Concurrent — Kernel FS Path (N={concurrency})")
    print(f"  Steps: {steps}, ckpt every: {ckpt_every}")
    print("=" * 60)

    init_env(device_id=device_id)
    model, ds, opt = make_gpt2xl_training(total_steps=steps, device_id=device_id)
    warmup_model(model, opt, ds)
    total_bytes = tpc.get_total_param_bytes(model)
    print(f"\n  Total params: {total_bytes / 1e9:.2f} GB")

    engine = PCCheckKernelFS(output_dir=NVME2_DIR, device_id=device_id,
                             concurrency=concurrency)
    engine.init_params(model)

    from direct_checkpoint import ProbeTrainOneStepCell
    cell = ProbeTrainOneStepCell(model, opt, enable_probe=False, ckpt_interval=9999)
    it = ds.create_tuple_iterator()
    _ = cell(*next(it))

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
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if s % ckpt_every == 0:
            ev = engine.checkpoint(model, s)
            print(f"  Step {s:3d}: dt={dt_ms:.1f}ms  sync={ev['sync_ms']:.1f}ms  "
                  f"memcpy={ev['memcpy_ms']:.1f}ms  block={ev['blocking_ms']:.1f}ms", flush=True)
        elif s <= 3 or s % 10 == 0:
            print(f"  Step {s:3d}: dt={dt_ms:.1f}ms", flush=True)
        step_times.append(dt_ms)

    print("  Waiting for persists...", flush=True)
    engine.wait_all()
    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    step_times = step_times[1:]
    stats = engine.get_stats()

    non_ckpt = [r for i, r in enumerate(step_times) if (i + 2) % ckpt_every != 0]
    ckpt_st = [r for i, r in enumerate(step_times) if (i + 2) % ckpt_every == 0]

    results = {
        "experiment": "pccheck_kernel_fs",
        "method": f"PCcheck Concurrent (aclrtMemcpy D2H + kernel FS, N={concurrency})",
        "total_bytes": total_bytes, "steps": steps, "ckpt_every": ckpt_every,
        "concurrency": concurrency,
        "step_mean_ms": round(float(np.mean(step_times)), 1),
        "step_std_ms": round(float(np.std(step_times, ddof=1)) if len(step_times) > 1 else 0.0, 1),
        "step_wo_ckpt_mean_ms": round(float(np.mean(non_ckpt)), 1) if non_ckpt else 0,
        "step_ckpt_mean_ms": round(float(np.mean(ckpt_st)), 1) if ckpt_st else 0,
        "total_time_ms": total_time_ms,
        "timing": stats, "ckpt_events": engine.ckpt_events,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"pccheck_kernel_fs_n{concurrency}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    s = stats["snapshot_ms"]; p = stats["persist_ms"]; b = stats["blocking_ms"]
    print(f"\n{'=' * 60}")
    print(f"PCCHECK KERNEL FS RESULTS (N={concurrency})")
    print(f"  Step mean: {results['step_mean_ms']:.1f} ms  (w/o ckpt: {results['step_wo_ckpt_mean_ms']:.1f})")
    print(f"  Snapshot:  mean={s['mean']:.1f} std={s['std']:.1f} ms  (n={s['n']})")
    print(f"  Persist:   mean={p['mean']:.1f} std={p['std']:.1f} ms  (n={p['n']})")
    print(f"  Blocking:  mean={b['mean']:.1f} max={b['max']:.1f} ms  (n={b['n']})")
    print(f"  Saved: {out_path}")
    print(f"{'=' * 60}")

    for s in range(ckpt_every, steps + 1, ckpt_every):
        for slot in range(concurrency):
            p = os.path.join(NVME2_DIR, f"pccheck_n{concurrency}_step{s}_slot{slot}.ckpt")
            if os.path.exists(p): os.remove(p)

    del engine, cell, model, opt, ds, it
    gc.collect()
    return results


def main():
    p = argparse.ArgumentParser(description="PCcheck Concurrent Kernel FS")
    p.add_argument("--device-id", type=int, default=1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--concurrency", "-n", type=int, default=3)
    args = p.parse_args()
    run_pccheck_kernel_fs(device_id=args.device_id, steps=args.steps,
                          ckpt_every=args.ckpt_every, concurrency=args.concurrency)


if __name__ == "__main__":
    main()
