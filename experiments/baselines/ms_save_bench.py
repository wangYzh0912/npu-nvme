#!/usr/bin/env python3
"""ms.save_checkpoint() benchmark — measure persist time only (snapshot is N/A).

Usage:
  sudo python experiments/baselines/ms_save_bench.py --device-id 1 --steps 30 --ckpt-every 5
"""

import os, sys, time, json, argparse, threading, gc
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, REPO)

import mindspore as ms
from experiments.common import make_gpt2xl_training, init_env, warmup_model

NVME2_DIR = os.environ.get(
    "NPU_NVME_BASELINE_DIR",
    os.path.join(REPO, "experiments", "output", "baselines", "payload"))
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")


class MSSaveBaseline:
    """ms.save_checkpoint() in a background thread — old (broken) approach.

    Kept for comparison: snapshot and persist are NOT separated here.
    """

    def __init__(self, output_dir=NVME2_DIR):
        self.output_dir = output_dir
        self._lock = threading.Lock()
        self.in_flight = False
        self._thread = None
        self._persist_error = None
        self.persist_ms_list = []
        self.blocking_ms_list = []
        self.ckpt_events = []
        os.makedirs(self.output_dir, exist_ok=True)

    def checkpoint(self, model, step):
        join_thread = None
        with self._lock:
            if self.in_flight and self._thread is not None:
                join_thread = self._thread
                self.in_flight = False
        blocking_ms = 0.0
        if join_thread is not None:
            t_wait = time.perf_counter()
            join_thread.join()
            blocking_ms = (time.perf_counter() - t_wait) * 1000.0
            self.blocking_ms_list.append(blocking_ms)

        path = os.path.join(self.output_dir, f"ms_save_step{step}.ckpt")
        self._thread = threading.Thread(target=self._worker, args=(model, path, step))
        self._thread.start()
        self.in_flight = True

        ev = {"step": step, "blocking_ms": round(blocking_ms, 3), "persist_ms": 0.0}
        self.ckpt_events.append(ev)
        return ev

    def _worker(self, model, path, step):
        try:
            t0 = time.perf_counter()
            ms.save_checkpoint(model, path)
            persist_ms = (time.perf_counter() - t0) * 1000.0
            self.persist_ms_list.append(persist_ms)
            with self._lock:
                for ev in reversed(self.ckpt_events):
                    if ev["step"] == step: ev["persist_ms"] = round(persist_ms, 3); break
        except BaseException as exc:
            self._persist_error = exc
        finally:
            self.in_flight = False

    def wait_all(self):
        t = None
        with self._lock:
            t = self._thread
            self.in_flight = False
        if t is not None and t.is_alive():
            t.join()
        if self._persist_error is not None:
            raise RuntimeError("background ms.save_checkpoint failed") from self._persist_error

    def get_stats(self):
        def _s(lst):
            if not lst: return {"mean": 0, "std": 0, "max": 0, "n": 0}
            a = np.array(lst)
            return {"mean": round(float(np.mean(a)), 3),
                    "std": round(float(np.std(a, ddof=1)) if len(a) > 1 else 0.0, 3),
                    "max": round(float(np.max(a)), 3), "n": len(a)}
        return {"persist_ms": _s(self.persist_ms_list),
                "blocking_ms": _s(self.blocking_ms_list)}


def run_ms_save_bench(device_id=1, steps=30, ckpt_every=5,
                      output_dir=NVME2_DIR):
    print("=" * 60)
    print(f"[ms.save] ms.save_checkpoint() Baseline")
    print(f"  Steps: {steps}, ckpt every: {ckpt_every}")
    print("=" * 60)

    init_env(device_id=device_id)
    model, ds, opt = make_gpt2xl_training(total_steps=steps, device_id=device_id)
    warmup_model(model, opt, ds)
    total_bytes = sum(int(p.size) * np.dtype(ms.dtype_to_nptype(p.dtype)).itemsize
                      for p in model.trainable_params())
    print(f"\n  Total params: {total_bytes / 1e9:.2f} GB")

    engine = MSSaveBaseline(output_dir=output_dir)

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
            print(f"  Step {s:3d}: dt={dt_ms:.1f}ms  block={ev['blocking_ms']:.1f}ms", flush=True)
        elif s <= 3 or s % 10 == 0:
            print(f"  Step {s:3d}: dt={dt_ms:.1f}ms", flush=True)
        step_times.append(dt_ms)

    print("  Waiting for persists...", flush=True)
    engine.wait_all()
    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    step_times = step_times[1:]
    stats = engine.get_stats()

    results = {
        "experiment": "ms_save_checkpoint",
        "method": "ms.save_checkpoint() (snapshot N/A — internal black box)",
        "total_bytes": total_bytes, "steps": steps, "ckpt_every": ckpt_every,
        "payload_output": os.path.abspath(output_dir),
        "step_mean_ms": round(float(np.mean(step_times)), 1),
        "total_time_ms": total_time_ms,
        "timing": stats, "ckpt_events": engine.ckpt_events,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "ms_save_checkpoint.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    p = stats["persist_ms"]; b = stats["blocking_ms"]
    print(f"\nMS.SAVE RESULTS")
    print(f"  Step mean: {results['step_mean_ms']:.1f} ms")
    print(f"  Persist:   mean={p['mean']:.1f} ms  (n={p['n']})")
    print(f"  Blocking:  mean={b['mean']:.1f} ms  (n={b['n']})")
    print(f"  Saved: {out_path}")

    for s in range(ckpt_every, steps + 1, ckpt_every):
        p = os.path.join(output_dir, f"ms_save_step{s}.ckpt")
        if os.path.exists(p): os.remove(p)

    del engine, cell, model, opt, ds, it
    gc.collect()
    return results


def main():
    p = argparse.ArgumentParser(description="ms.save_checkpoint() benchmark")
    p.add_argument("--device-id", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--output", default=NVME2_DIR,
                   help="filesystem path on the baseline NVMe device")
    args = p.parse_args()
    run_ms_save_bench(device_id=args.device_id, steps=args.steps,
                      ckpt_every=args.ckpt_every, output_dir=args.output)


if __name__ == "__main__":
    main()
