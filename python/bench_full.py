#!/usr/bin/env python3
"""Full delta-checkpoint benchmark for GPT-2 XL.

Measures three configurable stages, each independently skippable:

  Stage 1 — Baseline: pure training, no checkpointing at all.
            → baseline step time (zero overhead reference).

  Stage 2 — Delta pipeline: DeltaTrainCell + FaF listener.
            Delta writes every step (FaF async, ckpt_interval=1).
            FULL ckpt every --ckpt-every steps (sync SPDK blocking).
            → delta overhead, FULL ckpt sync latency, FaF correctness.

  Stage 3 — FULL ckpt only: registered model params, sync SPDK write,
            no delta computation.
            → pure FULL ckpt throughput (BW, sync latency).

Output: experiments/output/bench_full.json

Usage:
  # All stages (default)
  sudo python python/bench_full.py --device-id 1 --steps 50

  # Only delta pipeline
  sudo python python/bench_full.py --device-id 1 --skip-baseline --skip-full

  # Only FULL ckpt benchmark
  sudo python python/bench_full.py --device-id 1 --skip-baseline --skip-delta \
      --ckpt-every 5 --steps 30
"""

import os, sys, time, json, argparse
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, REPO)

import mindspore as ms

BLOCK_SIZE = 524288
TOP_K_FRAC = 0.10
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")


# -- Helpers --

def _loss_first(loss):
    """Unwrap tuple loss from TrainOneStepCell wrappers."""
    return loss[0] if isinstance(loss, (tuple, list)) else loss


def _full_ckpt_sync(ckpt, model, step, tag):
    """Save a FULL checkpoint and block until the SPDK write completes."""
    t0 = time.perf_counter()
    ckpt.save(model, step=step,
              meta_path=f"/tmp/bench_full_{tag}_step{step}.pkl")
    ckpt.wait_for_io_completion()
    return (time.perf_counter() - t0) * 1000


def _stats(times_ms):
    """Return {mean, std, min, max} for a list of millisecond values."""
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms":  float(np.std(times_ms)),
        "min_ms":  float(np.min(times_ms)),
        "max_ms":  float(np.max(times_ms)),
    }


# -- Benchmark ----------------------------------------------------------------

def run_benchmark(device_id=1, steps=50, ckpt_every=10,
                  skip_baseline=False, skip_delta=False, skip_full=False):
    """Run the selected benchmark stages.

    Args:
        device_id:     Ascend NPU device ID.
        steps:         total training steps per stage.
        ckpt_every:    save a FULL checkpoint every N steps.
        skip_baseline: skip Stage 1 (baseline step time).
        skip_delta:    skip Stage 2 (delta pipeline + FaF).
        skip_full:     skip Stage 3 (FULL ckpt only, no delta).
    """
    from experiments.common import make_gpt2xl_training, make_ckpt, init_env
    from delta_cell import DeltaTrainCell
    from direct_checkpoint import ProbeTrainOneStepCell

    ms.set_recursion_limit(10000)
    init_env(device_id=device_id)

    results = {
        "experiment": "bench_full",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model": "gpt2_xl", "steps": steps,
            "ckpt_every": ckpt_every, "block_size": BLOCK_SIZE,
            "top_k_frac": TOP_K_FRAC, "device_id": device_id,
            "stages": {
                "baseline": not skip_baseline,
                "delta":    not skip_delta,
                "full":     not skip_full,
            },
        },
    }

    baseline_mean = None  # for overhead calculation if baseline is skipped

    # ==================================================================
    # Stage 1 — Baseline (no delta ops, no checkpointing)
    # ==================================================================
    if not skip_baseline:
        print("=" * 60)
        print("[Stage 1] Baseline — pure training, no checkpointing")
        print("=" * 60)

        bl_model, bl_ds, bl_opt = make_gpt2xl_training(
            total_steps=steps, device_id=device_id)
        bl_cell = ProbeTrainOneStepCell(bl_model, bl_opt, enable_probe=False,
                                         ckpt_interval=9999)

        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                       device_id=device_id)

        it = bl_ds.create_tuple_iterator()
        _ = bl_cell(*next(it))

        bl_steps = []
        t_start = time.perf_counter()
        for s in range(1, steps + 1):
            try:
                data = next(it)
            except StopIteration:
                it = bl_ds.create_tuple_iterator()
                data = next(it)
            t0 = time.perf_counter()
            loss = bl_cell(*data)
            dt_ms = (time.perf_counter() - t0) * 1000
            bl_steps.append({"step": s, "dt_ms": dt_ms,
                              "loss": float(_loss_first(loss).asnumpy().flat[0])})
            if s % 10 == 0 or s == 1:
                print(f"  baseline step {s:3d}/{steps}  "
                      f"dt={dt_ms:.1f}ms  loss={bl_steps[-1]['loss']:.4f}")

        bl_times = [x["dt_ms"] for x in bl_steps]
        results["baseline"] = {
            "steps": bl_steps,
            **_stats(bl_times),
            "total_ms": (time.perf_counter() - t_start) * 1000,
        }
        baseline_mean = results["baseline"]["mean_ms"]
        print(f"  Baseline: {baseline_mean:.1f}ms +- "
              f"{results['baseline']['std_ms']:.1f}ms")
    else:
        print("[Stage 1] Baseline — SKIPPED")

    # ==================================================================
    # Stage 2 — Delta pipeline (DeltaTrainCell + FaF listener)
    # ==================================================================
    if not skip_delta:
        print()
        print("=" * 60)
        print("[Stage 2] Delta pipeline — DeltaTrainCell + FaF listener")
        print("=" * 60)
        print(f"  delta: every step (FaF async)  |  "
              f"FULL: every {ckpt_every} steps (sync SPDK)")

        model, ds, opt = make_gpt2xl_training(
            total_steps=steps, device_id=device_id)
        cell = DeltaTrainCell(model, opt, block_size=BLOCK_SIZE,
                               top_k_frac=TOP_K_FRAC)

        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                       device_id=device_id)

        ckpt = make_ckpt(device_id=device_id, pipeline_depth=8)

        it2 = ds.create_tuple_iterator()
        _ = cell(*next(it2))

        dev_flag, dev_step = ckpt.register_delta_tasks(cell, ckpt_interval=1)
        ckpt.delta_init(256, 128)

        delta_steps = []
        full_ckpt_times = []
        t_start2 = time.perf_counter()

        for s in range(1, steps + 1):
            try:
                data = next(it2)
            except StopIteration:
                it2 = ds.create_tuple_iterator()
                data = next(it2)

            t0 = time.perf_counter()
            loss = cell(*data)
            dt_ms = (time.perf_counter() - t0) * 1000

            try:
                flag = ckpt.read_probe_flag_dev()
            except Exception:
                flag = -1
            sc = int(cell.step_counter.value().asnumpy().flat[0])

            delta_steps.append({
                "step": s, "dt_ms": dt_ms,
                "loss": float(_loss_first(loss).asnumpy().flat[0]),
                "flag": flag, "step_counter": sc,
            })

            if s % ckpt_every == 0:
                dt_full = _full_ckpt_sync(ckpt, model, s, "delta")
                full_ckpt_times.append({"step": s, "dt_ms": dt_full})
                print(f"  delta step {s:3d}/{steps}  dt={dt_ms:.1f}ms  "
                      f"loss={delta_steps[-1]['loss']:.4f}  "
                      f"flag={flag}  sc={sc}  FULL(sync)={dt_full:.0f}ms")
            elif s % 10 == 0 or s == 1:
                print(f"  delta step {s:3d}/{steps}  dt={dt_ms:.1f}ms  "
                      f"loss={delta_steps[-1]['loss']:.4f}  "
                      f"flag={flag}  sc={sc}")

        dt_total2 = (time.perf_counter() - t_start2) * 1000

        p_old_val = float(np.abs(cell.delta_p_old.value().asnumpy()).sum())
        quant_val = float(np.abs(cell.delta_quant_buf.value().asnumpy()).sum())

        dt_times = [x["dt_ms"] for x in delta_steps]
        results["delta"] = {
            "steps": delta_steps,
            **_stats(dt_times),
            "total_ms": dt_total2,
            "full_ckpt_times": full_ckpt_times,
            "final_p_old_abs_sum": p_old_val,
            "final_quant_abs_sum": quant_val,
        }

        if baseline_mean is not None:
            overhead_ms = results["delta"]["mean_ms"] - baseline_mean
            overhead_pct = (overhead_ms / baseline_mean) * 100
            results["delta"]["overhead_ms"] = overhead_ms
            results["delta"]["overhead_pct"] = overhead_pct

        print(f"\n  Delta:    {results['delta']['mean_ms']:.1f}ms +- "
              f"{results['delta']['std_ms']:.1f}ms")
        if baseline_mean is not None:
            print(f"  Overhead: {results['delta']['overhead_ms']:+.1f}ms "
                  f"({results['delta']['overhead_pct']:+.1f}%)")
        if full_ckpt_times:
            avg_full = float(np.mean([x["dt_ms"] for x in full_ckpt_times]))
            print(f"  FULL ckpt: avg {avg_full:.0f}ms x {len(full_ckpt_times)} "
                  f"(every {ckpt_every} steps)")
        print(f"  P_old sum: {p_old_val:.1e}  quant sum: {quant_val:.1e}")

        ckpt.cleanup()
    else:
        print("[Stage 2] Delta pipeline — SKIPPED")

    # ==================================================================
    # Stage 3 — FULL ckpt only (registered params, sync write, no delta)
    # ==================================================================
    if not skip_full:
        print()
        print("=" * 60)
        print("[Stage 3] FULL ckpt only — registered params, sync SPDK, no delta")
        print("=" * 60)
        print(f"  FULL every {ckpt_every} steps (sync SPDK blocking write)")

        model3, ds3, opt3 = make_gpt2xl_training(
            total_steps=steps, device_id=device_id)
        cell3 = ProbeTrainOneStepCell(model3, opt3, enable_probe=True,
                                       ckpt_interval=ckpt_every)

        ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                       device_id=device_id)

        ckpt3 = make_ckpt(device_id=device_id, pipeline_depth=8)

        it3 = ds3.create_tuple_iterator()
        _ = cell3(*next(it3))

        ckpt3.register_tasks(model3, step=0)
        print("  Registered model params for FULL ckpt")

        full_steps = []
        full_ckpt_stats = []
        t_start3 = time.perf_counter()

        for s in range(1, steps + 1):
            try:
                data = next(it3)
            except StopIteration:
                it3 = ds3.create_tuple_iterator()
                data = next(it3)
            t0 = time.perf_counter()
            loss = cell3(*data)
            dt_ms = (time.perf_counter() - t0) * 1000

            if s % ckpt_every == 0:
                dt_full = _full_ckpt_sync(ckpt3, model3, s, "fullonly")
                full_ckpt_stats.append({"step": s, "sync_ms": dt_full})
                print(f"  FULL step {s:3d}/{steps}  step_dt={dt_ms:.1f}ms  "
                      f"FULL(sync)={dt_full:.0f}ms  "
                      f"loss={float(_loss_first(loss).asnumpy().flat[0]):.4f}")
            elif s % 10 == 0 or s == 1:
                print(f"  FULL step {s:3d}/{steps}  dt={dt_ms:.1f}ms  "
                      f"loss={float(_loss_first(loss).asnumpy().flat[0]):.4f}")

            full_steps.append({"step": s, "dt_ms": dt_ms})

        ft_times = [x["dt_ms"] for x in full_steps]
        results["full_ckpt_bench"] = {
            "steps": full_steps,
            "step_mean_ms": float(np.mean(ft_times)),
            "step_std_ms": float(np.std(ft_times)),
            "total_ms": (time.perf_counter() - t_start3) * 1000,
            "ckpt_stats": full_ckpt_stats,
        }
        if full_ckpt_stats:
            avg_sync = float(np.mean([x["sync_ms"] for x in full_ckpt_stats]))
            results["full_ckpt_bench"]["avg_sync_ms"] = avg_sync
            print(f"  FULL ckpt sync: avg {avg_sync:.0f}ms x "
                  f"{len(full_ckpt_stats)} (every {ckpt_every} steps)")
        print(f"  Step time: {results['full_ckpt_bench']['step_mean_ms']:.1f}ms +- "
              f"{results['full_ckpt_bench']['step_std_ms']:.1f}ms")

        ckpt3.cleanup()
    else:
        print("[Stage 3] FULL ckpt only — SKIPPED")

    # -- Summary --
    summary = {}
    if "baseline" in results:
        summary["baseline_mean_ms"] = results["baseline"]["mean_ms"]
    if "delta" in results:
        summary["delta_mean_ms"] = results["delta"]["mean_ms"]
        if "overhead_ms" in results["delta"]:
            summary["overhead_ms"] = results["delta"]["overhead_ms"]
            summary["overhead_pct"] = results["delta"]["overhead_pct"]
        summary["n_full_ckpts"] = len(results["delta"].get("full_ckpt_times", []))
        summary["p_old_nonzero"] = bool(
            results["delta"].get("final_p_old_abs_sum", 0) > 0)
        summary["quant_nonzero"] = bool(
            results["delta"].get("final_quant_abs_sum", 0) > 0)
    results["summary"] = summary

    return results


# -- Main --

def main():
    parser = argparse.ArgumentParser(
        description="Full delta-checkpoint benchmark for GPT-2 XL")
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=10,
                        help="FULL checkpoint interval in steps (default 10)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip Stage 1 (baseline step time)")
    parser.add_argument("--skip-delta", action="store_true",
                        help="Skip Stage 2 (delta pipeline + FaF)")
    parser.add_argument("--skip-full", action="store_true",
                        help="Skip Stage 3 (FULL ckpt only)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = run_benchmark(
        device_id=args.device_id,
        steps=args.steps,
        ckpt_every=args.ckpt_every,
        skip_baseline=args.skip_baseline,
        skip_delta=args.skip_delta,
        skip_full=args.skip_full,
    )

    out = os.path.join(OUTPUT_DIR, "bench_full.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")

    s = results.get("summary", {})
    parts = []
    if "baseline_mean_ms" in s:
        parts.append(f"baseline={s['baseline_mean_ms']:.1f}ms")
    if "delta_mean_ms" in s:
        parts.append(f"delta={s['delta_mean_ms']:.1f}ms")
    if "overhead_pct" in s:
        parts.append(f"overhead={s['overhead_ms']:+.1f}ms ({s['overhead_pct']:+.1f}%)")
    if "n_full_ckpts" in s:
        parts.append(f"FULL={s['n_full_ckpts']}")
    print(f"\n{'='*60}")
    print("FINAL: " + "  ".join(parts))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
