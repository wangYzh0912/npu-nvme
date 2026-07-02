#!/usr/bin/env python3
"""GPT-2 XL transfer and checkpoint benchmark example."""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, REPO)

import mindspore as ms

from delta_cell import DeltaTrainCell
from direct_checkpoint import ProbeTrainOneStepCell
from public_bench_helpers import init_env, make_ckpt, make_gpt2xl_training


BLOCK_SIZE = 524288
TOP_K_FRAC = 0.10
OUTPUT_DIR = os.path.join(REPO, "output")


def _loss_first(loss):
    return loss[0] if isinstance(loss, (tuple, list)) else loss


def _loss_float(loss):
    return float(_loss_first(loss).asnumpy().flat[0])


def _stats(times_ms):
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
    }


def _next_batch(iterator, dataset):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = dataset.create_tuple_iterator()
        return next(iterator), iterator


def _sync_full_checkpoint(ckpt, model, step, tag):
    t0 = time.perf_counter()
    ckpt.save(model, step=step,
              meta_path=os.path.join(OUTPUT_DIR, f"bench_{tag}_step{step}.pkl"))
    ckpt.wait_for_io_completion()
    return (time.perf_counter() - t0) * 1000


def _run_baseline(device_id, steps):
    print("=" * 60)
    print("[Stage 1] Baseline training")
    print("=" * 60)

    model, ds, opt = make_gpt2xl_training(total_steps=steps, device_id=device_id)
    cell = ProbeTrainOneStepCell(model, opt, enable_probe=False,
                                 ckpt_interval=9999)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    iterator = ds.create_tuple_iterator()
    data, iterator = _next_batch(iterator, ds)
    _ = cell(*data)

    rows = []
    t_start = time.perf_counter()
    for step in range(1, steps + 1):
        data, iterator = _next_batch(iterator, ds)
        t0 = time.perf_counter()
        loss = cell(*data)
        dt_ms = (time.perf_counter() - t0) * 1000
        row = {"step": step, "dt_ms": dt_ms, "loss": _loss_float(loss)}
        rows.append(row)
        if step == 1 or step % 10 == 0:
            print(f"  baseline step {step:3d}/{steps} "
                  f"dt={dt_ms:.1f}ms loss={row['loss']:.4f}")

    times = [row["dt_ms"] for row in rows]
    result = {
        "steps": rows,
        **_stats(times),
        "total_ms": (time.perf_counter() - t_start) * 1000,
    }
    print(f"  Baseline: {result['mean_ms']:.1f}ms +- "
          f"{result['std_ms']:.1f}ms")
    return result


def _run_delta_pipeline(device_id, steps, ckpt_every, baseline_mean):
    print()
    print("=" * 60)
    print("[Stage 2] Delta pipeline + periodic full checkpoint")
    print("=" * 60)

    model, ds, opt = make_gpt2xl_training(total_steps=steps, device_id=device_id)
    cell = DeltaTrainCell(model, opt, block_size=BLOCK_SIZE,
                          top_k_frac=TOP_K_FRAC)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    ckpt = make_ckpt(device_id=device_id, pipeline_depth=8)
    iterator = ds.create_tuple_iterator()
    data, iterator = _next_batch(iterator, ds)
    _ = cell(*data)

    ckpt.register_delta_tasks(cell, ckpt_interval=1)
    ckpt.delta_init(256, 128)

    rows = []
    full_ckpts = []
    t_start = time.perf_counter()
    try:
        for step in range(1, steps + 1):
            data, iterator = _next_batch(iterator, ds)
            t0 = time.perf_counter()
            loss = cell(*data)
            dt_ms = (time.perf_counter() - t0) * 1000

            try:
                flag = ckpt.read_probe_flag_dev()
            except Exception:
                flag = -1
            step_counter = int(cell.step_counter.value().asnumpy().flat[0])
            row = {
                "step": step,
                "dt_ms": dt_ms,
                "loss": _loss_float(loss),
                "flag": flag,
                "step_counter": step_counter,
            }
            rows.append(row)

            if step % ckpt_every == 0:
                full_ms = _sync_full_checkpoint(ckpt, model, step, "delta")
                full_ckpts.append({"step": step, "dt_ms": full_ms})
                print(f"  delta step {step:3d}/{steps} dt={dt_ms:.1f}ms "
                      f"loss={row['loss']:.4f} FULL={full_ms:.0f}ms")
            elif step == 1 or step % 10 == 0:
                print(f"  delta step {step:3d}/{steps} dt={dt_ms:.1f}ms "
                      f"loss={row['loss']:.4f} flag={flag}")

        times = [row["dt_ms"] for row in rows]
        result = {
            "steps": rows,
            **_stats(times),
            "total_ms": (time.perf_counter() - t_start) * 1000,
            "full_ckpt_times": full_ckpts,
            "final_p_old_abs_sum": float(
                np.abs(cell.delta_p_old.value().asnumpy()).sum()),
            "final_quant_abs_sum": float(
                np.abs(cell.delta_quant_buf.value().asnumpy()).sum()),
        }
        if baseline_mean is not None:
            result["overhead_ms"] = result["mean_ms"] - baseline_mean
            result["overhead_pct"] = result["overhead_ms"] / baseline_mean * 100
        print(f"  Delta: {result['mean_ms']:.1f}ms +- "
              f"{result['std_ms']:.1f}ms")
        return result
    finally:
        ckpt.cleanup()


def _run_full_checkpoint(device_id, steps, ckpt_every):
    print()
    print("=" * 60)
    print("[Stage 3] Full checkpoint only")
    print("=" * 60)

    model, ds, opt = make_gpt2xl_training(total_steps=steps, device_id=device_id)
    cell = ProbeTrainOneStepCell(model, opt, enable_probe=True,
                                 ckpt_interval=ckpt_every)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    ckpt = make_ckpt(device_id=device_id, pipeline_depth=8)
    iterator = ds.create_tuple_iterator()
    data, iterator = _next_batch(iterator, ds)
    _ = cell(*data)
    ckpt.register_tasks(model, step=0)

    rows = []
    full_ckpts = []
    t_start = time.perf_counter()
    try:
        for step in range(1, steps + 1):
            data, iterator = _next_batch(iterator, ds)
            t0 = time.perf_counter()
            loss = cell(*data)
            dt_ms = (time.perf_counter() - t0) * 1000
            rows.append({"step": step, "dt_ms": dt_ms})

            if step % ckpt_every == 0:
                full_ms = _sync_full_checkpoint(ckpt, model, step, "full")
                full_ckpts.append({"step": step, "sync_ms": full_ms})
                print(f"  full step {step:3d}/{steps} dt={dt_ms:.1f}ms "
                      f"FULL={full_ms:.0f}ms loss={_loss_float(loss):.4f}")
            elif step == 1 or step % 10 == 0:
                print(f"  full step {step:3d}/{steps} dt={dt_ms:.1f}ms "
                      f"loss={_loss_float(loss):.4f}")

        times = [row["dt_ms"] for row in rows]
        result = {
            "steps": rows,
            "step_mean_ms": float(np.mean(times)),
            "step_std_ms": float(np.std(times)),
            "total_ms": (time.perf_counter() - t_start) * 1000,
            "ckpt_stats": full_ckpts,
        }
        if full_ckpts:
            result["avg_sync_ms"] = float(
                np.mean([row["sync_ms"] for row in full_ckpts]))
        return result
    finally:
        ckpt.cleanup()


def run_benchmark(device_id=0, steps=50, ckpt_every=10,
                  skip_baseline=False, skip_delta=False, skip_full=False):
    init_env(device_id=device_id)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {
        "benchmark": "bench_full",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model": "gpt2_xl",
            "steps": steps,
            "ckpt_every": ckpt_every,
            "block_size": BLOCK_SIZE,
            "top_k_frac": TOP_K_FRAC,
            "device_id": device_id,
        },
    }

    baseline_mean = None
    if not skip_baseline:
        results["baseline"] = _run_baseline(device_id, steps)
        baseline_mean = results["baseline"]["mean_ms"]
    else:
        print("[Stage 1] Baseline training skipped")

    if not skip_delta:
        results["delta"] = _run_delta_pipeline(
            device_id, steps, ckpt_every, baseline_mean)
    else:
        print("[Stage 2] Delta pipeline skipped")

    if not skip_full:
        results["full_ckpt_bench"] = _run_full_checkpoint(
            device_id, steps, ckpt_every)
    else:
        print("[Stage 3] Full checkpoint skipped")

    summary = {}
    if "baseline" in results:
        summary["baseline_mean_ms"] = results["baseline"]["mean_ms"]
    if "delta" in results:
        summary["delta_mean_ms"] = results["delta"]["mean_ms"]
        if "overhead_ms" in results["delta"]:
            summary["overhead_ms"] = results["delta"]["overhead_ms"]
            summary["overhead_pct"] = results["delta"]["overhead_pct"]
        summary["n_full_ckpts"] = len(
            results["delta"].get("full_ckpt_times", []))
    if "full_ckpt_bench" in results:
        summary["full_ckpt_avg_sync_ms"] = results[
            "full_ckpt_bench"].get("avg_sync_ms")
    results["summary"] = summary
    return results


def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 XL transfer and checkpoint benchmark")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-delta", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    args = parser.parse_args()

    results = run_benchmark(
        device_id=args.device_id,
        steps=args.steps,
        ckpt_every=args.ckpt_every,
        skip_baseline=args.skip_baseline,
        skip_delta=args.skip_delta,
        skip_full=args.skip_full,
    )

    out = os.path.join(OUTPUT_DIR, "bench_full.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")

    parts = []
    summary = results.get("summary", {})
    if "baseline_mean_ms" in summary:
        parts.append(f"baseline={summary['baseline_mean_ms']:.1f}ms")
    if "delta_mean_ms" in summary:
        parts.append(f"delta={summary['delta_mean_ms']:.1f}ms")
    if "overhead_pct" in summary:
        parts.append(f"overhead={summary['overhead_ms']:+.1f}ms "
                     f"({summary['overhead_pct']:+.1f}%)")
    if summary.get("full_ckpt_avg_sync_ms") is not None:
        parts.append(f"full_ckpt={summary['full_ckpt_avg_sync_ms']:.0f}ms")
    print("FINAL: " + "  ".join(parts))


if __name__ == "__main__":
    main()
