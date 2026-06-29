#!/usr/bin/env python3
"""§13 — CheckFreq Two-Phase Checkpoint (kernel FS path → NVMe #2).

Correct reproduction of CheckFreq (FAST'21) two-phase checkpoint pipeline:

  Phase 1 (blocking):  aclrtMemcpy D2H — copy each parameter from HBM to
                        host DRAM.  (~1059 ms for 3.28 GB / 772 params on
                        Ascend 910B — per-call DMA overhead dominates)

  Phase 2 (background): open().write() + os.fsync() → NVMe #2.
                        (~6-7s kernel FS write)

  At-most-1 in-flight:  join previous persist thread before new snapshot.

NOTE: On Ascend 910B, aclrtMemcpy per-call overhead (~0.9 ms/call) makes
per-parameter D2H slower than the theoretical PCIe bandwidth suggests.
We report this honestly — the relative blocking behaviour (CheckFreq vs
PCcheck vs SPDK) is what matters for the paper comparison.

Usage:
  sudo python experiments/baselines/checkfreq_kernel_fs.py --device-id 1 --steps 20
"""

import os, sys, time, json, argparse, threading, gc
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, REPO)

import mindspore as ms
from experiments.common import make_gpt2xl_training, init_env, warmup_model
from experiments.baselines import two_phase_common as tpc

NVME2_DIR = "/models/nvme_baseline"
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")


class CheckFreqTwoPhaseKernelFS:
    """CheckFreq (FAST'21) two-phase: per-param D2H + kernel FS persist."""

    def __init__(self, output_dir=NVME2_DIR, device_id=0):
        self.output_dir = output_dir
        self.device_id = device_id
        self._param_descs = None
        self._offset_map = None
        self._total_bytes = 0
        self._lock = threading.Lock()
        self.in_flight = False
        self._persist_thread = None
        self._pinned_ptr = 0
        self._use_pinned = False
        self.snapshot_ms_list = []
        self.persist_ms_list = []
        self.blocking_ms_list = []
        self.ckpt_events = []
        os.makedirs(self.output_dir, exist_ok=True)

    def init_params(self, model):
        self._param_descs = tpc.get_param_descriptors(model)
        self._offset_map, self._total_bytes = tpc.build_offset_map(self._param_descs)
        print(f"[CheckFreq] {len(self._param_descs)} params, {self._total_bytes/1e9:.2f} GB")

    def alloc_pinned(self):
        """Try to allocate pinned host buffer (best-effort)."""
        try:
            tpc._ensure_acl_device(self.device_id)
            self._pinned_ptr = tpc.allocate_pinned_host_buffer(self._total_bytes)
            self._use_pinned = True
            print(f"[CheckFreq] pinned buf 0x{self._pinned_ptr:016x}")
        except RuntimeError as e:
            self._use_pinned = False
            print(f"[CheckFreq] pinned alloc failed ({e}), falling back to numpy")

    def checkpoint(self, model, step):
        blocking_ms = 0.0
        # Check for in-flight persist — release lock before joining to avoid
        # deadlock with _persist_worker which also acquires self._lock.
        join_thread = None
        with self._lock:
            if self.in_flight and self._persist_thread is not None:
                join_thread = self._persist_thread
                self.in_flight = False  # let persist worker safely update it too

        if join_thread is not None:
            t_wait = time.perf_counter()
            join_thread.join()
            blocking_ms = (time.perf_counter() - t_wait) * 1000.0
            self.blocking_ms_list.append(blocking_ms)

        if self._use_pinned:
            snap = tpc.snapshot_d2h_pinned(
                self._param_descs, self._pinned_ptr, self._offset_map, self.device_id)
            snapshot_ms = snap["total_ms"]
            filepath = os.path.join(self.output_dir, f"checkfreq_step{step}.ckpt")
            self._persist_thread = threading.Thread(
                target=self._persist_worker_pinned, args=(filepath, step), daemon=False)
        else:
            host_buf = tpc.allocate_host_buffer(self._total_bytes)
            snap = tpc.snapshot_d2h(
                self._param_descs, host_buf, self._offset_map, self.device_id)
            snapshot_ms = snap["total_ms"]
            filepath = os.path.join(self.output_dir, f"checkfreq_step{step}.ckpt")
            self._persist_thread = threading.Thread(
                target=self._persist_worker, args=(host_buf, filepath, step), daemon=False)

        self.snapshot_ms_list.append(snapshot_ms)
        self._persist_thread.start()
        self.in_flight = True

        ev = {"step": step, "snapshot_ms": round(snapshot_ms, 3),
              "sync_ms": snap["sync_ms"], "memcpy_ms": snap["memcpy_ms"],
              "blocking_ms": round(blocking_ms, 3), "persist_ms": 0.0}
        self.ckpt_events.append(ev)
        return ev

    def _persist_worker(self, host_buf, filepath, step):
        persist_ms = tpc.persist_to_file(host_buf, filepath)
        self.persist_ms_list.append(persist_ms)
        with self._lock:
            for ev in reversed(self.ckpt_events):
                if ev["step"] == step: ev["persist_ms"] = round(persist_ms, 3); break
        self.in_flight = False

    def _persist_worker_pinned(self, filepath, step):
        persist_ms = tpc.persist_to_file_pinned(
            self._pinned_ptr, self._total_bytes, filepath)
        self.persist_ms_list.append(persist_ms)
        with self._lock:
            for ev in reversed(self.ckpt_events):
                if ev["step"] == step: ev["persist_ms"] = round(persist_ms, 3); break
        self.in_flight = False

    def wait_all(self):
        thread = None
        with self._lock:
            thread = self._persist_thread
            self.in_flight = False
        if thread is not None and thread.is_alive():
            thread.join()

    def get_stats(self):
        def _s(lst):
            if not lst: return {"mean": 0, "std": 0, "max": 0, "n": 0}
            a = np.array(lst)
            return {"mean": round(float(np.mean(a)), 3),
                    "std": round(float(np.std(a, ddof=1)) if len(a) > 1 else 0.0, 3),
                    "max": round(float(np.max(a)), 3), "n": len(a)}
        return {"snapshot_ms": _s(self.snapshot_ms_list),
                "persist_ms": _s(self.persist_ms_list),
                "blocking_ms": _s(self.blocking_ms_list)}


def run_checkfreq_bench(device_id=1, steps=20, ckpt_every=10):
    print("=" * 60)
    print("[B1] CheckFreq Two-Phase — Kernel FS Path")
    print(f"  Steps: {steps}, ckpt every: {ckpt_every}")
    print("=" * 60)

    init_env(device_id=device_id)
    model, ds, opt = make_gpt2xl_training(total_steps=steps, device_id=device_id)
    warmup_model(model, opt, ds)
    total_bytes = tpc.get_total_param_bytes(model)
    print(f"\n  Total params: {total_bytes / 1e9:.2f} GB")

    engine = CheckFreqTwoPhaseKernelFS(output_dir=NVME2_DIR, device_id=device_id)
    # MUST allocate pinned BEFORE init_params — get_dev_ptr() changes ACL
    # state in a way that makes subsequent aclrtMallocHost fail.
    engine._total_bytes = total_bytes
    engine.alloc_pinned()
    engine.init_params(model)

    from direct_checkpoint import ProbeTrainOneStepCell
    cell = ProbeTrainOneStepCell(model, opt, enable_probe=False, ckpt_interval=9999)
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
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if s % ckpt_every == 0:
            ev = engine.checkpoint(model, s)
            print(f"  Step {s:3d}: dt={dt_ms:.1f}ms  sync={ev['sync_ms']:.1f}ms  "
                  f"memcpy={ev['memcpy_ms']:.1f}ms  block={ev['blocking_ms']:.1f}ms", flush=True)
        elif s <= 3 or s % 10 == 0:
            print(f"  Step {s:3d}: dt={dt_ms:.1f}ms", flush=True)
        step_times.append(dt_ms)

    print("  Waiting for persist...", flush=True)
    engine.wait_all()
    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    step_times = step_times[1:]  # skip compile
    stats = engine.get_stats()

    non_ckpt = [r for i, r in enumerate(step_times) if (i + 2) % ckpt_every != 0]
    ckpt_st = [r for i, r in enumerate(step_times) if (i + 2) % ckpt_every == 0]

    results = {
        "experiment": "checkfreq_kernel_fs",
        "method": "CheckFreq Two-Phase (aclrtMemcpy D2H + kernel FS)",
        "total_bytes": total_bytes, "steps": steps, "ckpt_every": ckpt_every,
        "concurrency": 1,
        "step_mean_ms": round(float(np.mean(step_times)), 1),
        "step_std_ms": round(float(np.std(step_times, ddof=1)) if len(step_times) > 1 else 0.0, 1),
        "step_wo_ckpt_mean_ms": round(float(np.mean(non_ckpt)), 1) if non_ckpt else 0,
        "step_ckpt_mean_ms": round(float(np.mean(ckpt_st)), 1) if ckpt_st else 0,
        "total_time_ms": total_time_ms,
        "timing": stats, "ckpt_events": engine.ckpt_events,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "checkfreq_kernel_fs.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    s = stats["snapshot_ms"]; p = stats["persist_ms"]; b = stats["blocking_ms"]
    print(f"\n{'=' * 60}")
    print("CHECKFREQ KERNEL FS RESULTS")
    print(f"  Step mean: {results['step_mean_ms']:.1f} ms  (w/o ckpt: {results['step_wo_ckpt_mean_ms']:.1f})")
    print(f"  Snapshot:  mean={s['mean']:.1f} std={s['std']:.1f} ms  (n={s['n']})")
    print(f"  Persist:   mean={p['mean']:.1f} std={p['std']:.1f} ms  (n={p['n']})")
    print(f"  Blocking:  mean={b['mean']:.1f} max={b['max']:.1f} ms  (n={b['n']})")
    print(f"  Saved: {out_path}")
    print(f"{'=' * 60}")

    for s in range(ckpt_every, steps + 1, ckpt_every):
        p = os.path.join(NVME2_DIR, f"checkfreq_step{s}.ckpt")
        if os.path.exists(p): os.remove(p)

    del engine, cell, model, opt, ds, it
    gc.collect()
    return results


def main():
    p = argparse.ArgumentParser(description="CheckFreq Two-Phase Kernel FS")
    p.add_argument("--device-id", type=int, default=1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=10)
    args = p.parse_args()
    run_checkfreq_bench(device_id=args.device_id, steps=args.steps, ckpt_every=args.ckpt_every)


if __name__ == "__main__":
    main()
