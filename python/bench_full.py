#!/usr/bin/env python3
"""Full delta-checkpoint benchmark for GPT-2 XL.

Measures three configurations in sequence:

  Stage 1 — Baseline: ProbeTrainOneStepCell (no delta ops, no FaF).
  Stage 2 — Delta pipeline: DeltaTrainCell + FaF listener, with
            FULL checkpoint at epoch boundary (sync SPDK write).
  Stage 3 — FULL ckpt only: ProbeTrainOneStepCell with registered
            model params, sync SPDK write, no delta.

Output: experiments/output/bench_full.json

Usage:
  sudo python python/bench_full.py --device-id 1 --steps 50 --ckpt-every 10
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
    """Save a FULL checkpoint and block until the SPDK write completes.

    Args:
        ckpt:  DirectCheckpoint instance.
        model: MindSpore nn.Cell.
        step:  training step number.
        tag:   label embedded in the temporary file path.

    Returns:
        wall-clock time in milliseconds.
    """
    t0 = time.perf_counter()
    ckpt.save(model, step=step,
              meta_path=f"/tmp/bench_full_{tag}_step{step}.pkl")
    ckpt.wait_for_io_completion()
    return (time.perf_counter() - t0) * 1000


# -- Benchmark --

def run_benchmark(device_id=1, steps=50, ckpt_every=10):
    """Run all three stages and return a results dict.

    Args:
        device_id:   Ascend NPU device ID (default 1).
        steps:       total training steps (default 50).
        ckpt_every:  save a FULL checkpoint every N steps (default 10).

    Returns:
        dict with keys "baseline", "delta", "full_ckpt_bench", "summary".
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
        },
    }

    # ==================================================================
    # Stage 1 — Baseline (no delta ops)
    # ==================================================================
    print("=" * 60)
    print("[Stage 1] Baseline — ProbeTrainOneStepCell (no delta)")
    print("=" * 60)

    bl_model, bl_ds, bl_opt = make_gpt2xl_training(
        total_steps=steps, device_id=device_id)
    bl_cell = ProbeTrainOneStepCell(bl_model, bl_opt, enable_probe=False,
                                     ckpt_interval=9999)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    it = bl_ds.create_tuple_iterator()
    _ = bl_cell(*next(it))  # compile

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
    bl_total = (time.perf_counter() - t_start) * 1000

    bl_times = [x["dt_ms"] for x in bl_steps]
    results["baseline"] = {
        "steps": bl_steps,
        "mean_ms": float(np.mean(bl_times)),
        "std_ms": float(np.std(bl_times)),
        "min_ms": float(np.min(bl_times)),
        "max_ms": float(np.max(bl_times)),
        "total_ms": bl_total,
    }
    print(f"  Baseline: {results['baseline']['mean_ms']:.1f}ms +- "
          f"{results['baseline']['std_ms']:.1f}ms  (total {bl_total:.0f}ms)")

    # ==================================================================
    # Stage 2 — Delta pipeline (DeltaTrainCell + FaF listener)
    # ==================================================================
    print()
    print("=" * 60)
    print("[Stage 2] Delta pipeline — DeltaTrainCell + FaF listener")
    print("=" * 60)

    model, ds, opt = make_gpt2xl_training(
        total_steps=steps, device_id=device_id)
    cell = DeltaTrainCell(model, opt, block_size=BLOCK_SIZE,
                           top_k_frac=TOP_K_FRAC)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    ckpt = make_ckpt(device_id=device_id, pipeline_depth=8)

    # Compile + wire FaF
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
        "mean_ms": float(np.mean(dt_times)),
        "std_ms": float(np.std(dt_times)),
        "min_ms": float(np.min(dt_times)),
        "max_ms": float(np.max(dt_times)),
        "total_ms": dt_total2,
        "full_ckpt_times": full_ckpt_times,
        "final_p_old_abs_sum": p_old_val,
        "final_quant_abs_sum": quant_val,
    }

    overhead_ms = results["delta"]["mean_ms"] - results["baseline"]["mean_ms"]
    overhead_pct = (overhead_ms / results["baseline"]["mean_ms"]) * 100
    results["delta"]["overhead_ms"] = overhead_ms
    results["delta"]["overhead_pct"] = overhead_pct

    print(f"\n  Delta:    {results['delta']['mean_ms']:.1f}ms +- "
          f"{results['delta']['std_ms']:.1f}ms")
    print(f"  Overhead: {overhead_ms:+.1f}ms ({overhead_pct:+.1f}%)")
    if full_ckpt_times:
        avg_full = float(np.mean([x["dt_ms"] for x in full_ckpt_times]))
        print(f"  FULL ckpt: avg {avg_full:.0f}ms x {len(full_ckpt_times)} "
              f"(every {ckpt_every} steps)")
    print(f"  P_old sum: {p_old_val:.1e}  quant sum: {quant_val:.1e}")

    ckpt.cleanup()

    # ==================================================================
    # Stage 3 — FULL ckpt only (registered params, sync write, no delta)
    # ==================================================================
    print()
    print("=" * 60)
    print("[Stage 3] FULL ckpt — registered params, sync SPDK write, no delta")
    print("=" * 60)

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

    dt_total3 = (time.perf_counter() - t_start3) * 1000
    ft_times = [x["dt_ms"] for x in full_steps]
    results["full_ckpt_bench"] = {
        "step_mean_ms": float(np.mean(ft_times)),
        "step_std_ms": float(np.std(ft_times)),
        "total_ms": dt_total3,
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

    # -- Summary --
    results["summary"] = {
        "baseline_mean_ms": results["baseline"]["mean_ms"],
        "delta_mean_ms": results["delta"]["mean_ms"],
        "overhead_ms": overhead_ms,
        "overhead_pct": overhead_pct,
        "n_full_ckpts": len(full_ckpt_times),
        "p_old_nonzero": bool(p_old_val > 0),
        "quant_nonzero": bool(quant_val > 0),
    }

    return results


# -- Main --

def main():
    parser = argparse.ArgumentParser(
        description="Full delta-checkpoint benchmark for GPT-2 XL")
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=10,
                        help="FULL checkpoint interval in steps (default 10)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = run_benchmark(
        device_id=args.device_id, steps=args.steps,
        ckpt_every=args.ckpt_every)

    out = os.path.join(OUTPUT_DIR, "bench_full.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")

    s = results["summary"]
    print(f"\n{'='*60}")
    print(f"FINAL: baseline={s['baseline_mean_ms']:.1f}ms  "
          f"delta={s['delta_mean_ms']:.1f}ms  "
          f"overhead={s['overhead_ms']:+.1f}ms ({s['overhead_pct']:+.1f}%)  "
          f"FULL={s['n_full_ckpts']}  "
          f"p_old={'OK' if s['p_old_nonzero'] else 'ZERO'}  "
          f"quant={'OK' if s['quant_nonzero'] else 'ZERO'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
