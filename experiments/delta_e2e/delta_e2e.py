#!/usr/bin/env python3
"""Delta-checkpoint end-to-end verification — T4, T5, T6.

Validates the full incremental-checkpoint pipeline:

  T4 — multi-step FaF: listener triggers SPDK writes via direct iteration.
  T5 — step-time overhead vs baseline (ProbeTrainOneStepCell without delta ops).
  T6 — recovery: FULL ckpt + delta chain restore, NRMSE vs oracle.

Usage:
  bash _run.sh [DEVICE_ID] [--steps N]

Output: experiments/output/delta_e2e/delta_e2e.json
"""

import os, sys, time, json, ctypes, argparse

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, REPO)

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops
from mindspore.common.initializer import Normal

PCI_ADDR = "0000:83:00.0"
BLOCK_SIZE = 524288
TOP_K_FRAC = 0.10
OUTPUT_DIR = os.path.join(REPO, "experiments", "output", "delta_e2e")


# ---- Helpers ---------------------------------------------------------------

def compute_nrmse(recovered, oracle):
    """Per-parameter normalised RMSE."""
    all_n = []
    for nm in oracle:
        r = recovered[nm].astype(np.float64).flatten()
        o = oracle[nm].astype(np.float64).flatten()
        d = r - o
        std = float(np.std(o)) + 1e-12
        all_n.append(float(np.sqrt(np.mean(d ** 2))) / std)
    return {
        "mean": float(np.mean(all_n)),
        "median": float(np.median(all_n)),
        "max": float(np.max(all_n)),
        "n_params": len(all_n),
    }


def get_all_params_np(model):
    """Snapshot all trainable parameters as numpy dict."""
    return {p.name: p.value().asnumpy().copy()
            for p in model.trainable_params()}


def direct_train(cell, ds, steps, label="train"):
    """Run *steps* direct iterations, return step times in seconds."""
    times = []
    it = ds.create_tuple_iterator()
    for s in range(steps):
        try:
            data = next(it)
        except StopIteration:
            it = ds.create_tuple_iterator()
            data = next(it)
        t0 = time.perf_counter()
        loss = cell(*data)
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)
        if (s + 1) % 10 == 0 or s == 0:
            print(f"  [{label}] step {s + 1}/{steps}  "
                  f"loss={float(loss.asnumpy().flat[0]):.4f}  "
                  f"dt={dt * 1000:.1f}ms", flush=True)
    return times


# ---- T4: Multi-step FaF trigger -------------------------------------------

def test_faf_trigger(device_id, steps=10):
    """T4: Multi-step training with FaF listener — direct iteration.

    Uses direct iteration (cell(*data)) instead of ms.Model.train to
    avoid the attention-mask shape broadcasting issue in the framework
    wrapper layer.
    """
    from delta_cell import DeltaTrainCell
    from experiments.common import make_gpt2xl_training, make_ckpt

    print(f"[T4] Multi-step FaF trigger ({steps} steps, direct iteration)")
    model, ds, opt = make_gpt2xl_training(total_steps=steps, device_id=device_id)

    cell = DeltaTrainCell(model, opt, block_size=BLOCK_SIZE, top_k_frac=TOP_K_FRAC)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    ckpt = make_ckpt(device_id=device_id, pipeline_depth=8)

    # Compile with one dummy step
    it = ds.create_tuple_iterator()
    data = next(it)
    _ = cell(*data)

    # Wire FaF listener
    dev_flag, dev_step = ckpt.register_delta_tasks(cell, ckpt_interval=1)
    ckpt.delta_init(256, 128)

    # Train with direct iteration
    flags = []
    step_vals = []
    for s in range(1, steps + 1):
        try:
            data = next(it)
        except StopIteration:
            it = ds.create_tuple_iterator()
            data = next(it)
        loss = cell(*data)
        sc = int(cell.step_counter.value().asnumpy().flat[0])
        step_vals.append(sc)
        try:
            flag = ckpt.read_probe_flag_dev()
            flags.append(flag)
        except Exception:
            flags.append(-1)

    final_flag = flags[-1] if flags else -1
    final_step = step_vals[-1] if step_vals else -1

    results = {
        "status": "pass" if final_step >= steps else "fail",
        "probe_flags": flags,
        "final_flag": final_flag,
        "step_counter_values": step_vals,
        "final_step": final_step,
        "expected_min": steps,
    }
    print(f"  T4 result — step_counter={step_vals}  flags={flags}  "
          f"→ {results['status']}")
    ckpt.cleanup()
    return results


# ---- T5: Overhead comparison ----------------------------------------------

def test_overhead(device_id, steps=50):
    """T5: Step-time overhead of I3 delta pipeline vs baseline.

    Compare two configurations over *steps* iterations:
      Baseline — ProbeTrainOneStepCell (FaF step counter, NO delta ops)
      I3       — DeltaTrainCell (full 7-phase GE delta pipeline)

    Returns step-by-step times and aggregate statistics.
    """
    from delta_cell import DeltaTrainCell
    from experiments.common import make_gpt2xl_training, make_ckpt
    from direct_checkpoint import ProbeTrainOneStepCell

    print(f"[T5] Overhead benchmark ({steps} steps each)")

    # -- Baseline: ProbeTrainOneStepCell (no delta ops) --
    print("  [T5] Baseline — ProbeTrainOneStepCell...")
    model_b, ds_b, opt_b = make_gpt2xl_training(total_steps=steps,
                                                  device_id=device_id)
    cell_b = ProbeTrainOneStepCell(model_b, opt_b, enable_probe=False,
                                    ckpt_interval=9999)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)
    # Compile
    it_b = ds_b.create_tuple_iterator()
    _ = cell_b(*next(it_b))
    baseline_times = direct_train(cell_b, ds_b, steps, label="baseline")

    # -- I3: DeltaTrainCell (full delta pipeline) --
    print("  [T5] I3 — DeltaTrainCell...")
    model_i3, ds_i3, opt_i3 = make_gpt2xl_training(total_steps=steps,
                                                     device_id=device_id)
    cell_i3 = DeltaTrainCell(model_i3, opt_i3, block_size=BLOCK_SIZE,
                              top_k_frac=TOP_K_FRAC)
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)
    # Compile
    it_i3 = ds_i3.create_tuple_iterator()
    _ = cell_i3(*next(it_i3))
    i3_times = direct_train(cell_i3, ds_i3, steps, label="i3")

    bl_mean = float(np.mean(baseline_times))
    bl_std = float(np.std(baseline_times))
    i3_mean = float(np.mean(i3_times))
    i3_std = float(np.std(i3_times))
    overhead_ms = (i3_mean - bl_mean) * 1000
    overhead_pct = (i3_mean / bl_mean - 1.0) * 100 if bl_mean > 0 else 0

    results = {
        "status": "pass" if overhead_ms < 50 else "warn",
        "baseline": {"mean_s": bl_mean, "std_s": bl_std,
                      "times": [float(t) for t in baseline_times]},
        "i3": {"mean_s": i3_mean, "std_s": i3_std,
                "times": [float(t) for t in i3_times]},
        "overhead_ms": overhead_ms,
        "overhead_pct": overhead_pct,
    }
    print(f"  T5 result — baseline={bl_mean * 1000:.1f}ms ± {bl_std * 1000:.1f}  "
          f"I3={i3_mean * 1000:.1f}ms ± {i3_std * 1000:.1f}  "
          f"overhead={overhead_ms:.1f}ms ({overhead_pct:+.1f}%)  "
          f"→ {results['status']}")
    return results


# ---- T6: Recovery verification --------------------------------------------

def test_recovery(device_id, steps=20):
    """T6: FULL ckpt + delta chain → recover → NRMSE vs oracle.

    Strategy:
      1. Train with DeltaTrainCell for N steps, saving FULL at step 0
         and delta frames via FaF each step.
      2. Record oracle (final weights).
      3. Create a fresh model, recover to step N via FULL + delta chain.
      4. Compare recovered weights with oracle.
    """
    from delta_cell import DeltaTrainCell
    from experiments.common import make_gpt2xl_training, make_ckpt

    print(f"[T6] Recovery verification ({steps} steps)")
    model, ds, opt = make_gpt2xl_training(total_steps=steps, device_id=device_id)

    cell = DeltaTrainCell(model, opt, block_size=BLOCK_SIZE, top_k_frac=TOP_K_FRAC)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    ckpt = make_ckpt(device_id=device_id, pipeline_depth=8)
    ckpt.delta_init(256, 128)

    # Compile
    it = ds.create_tuple_iterator()
    _ = cell(*next(it))

    # Save FULL checkpoint at step 0 (before any training)
    ckpt.save("/tmp/t6_full_step0.pkl", step=0)

    # Wire FaF listener for delta writes
    dev_flag, dev_step = ckpt.register_delta_tasks(cell, ckpt_interval=1)

    # Train with direct iteration, collect oracle
    for s in range(1, steps + 1):
        try:
            data = next(it)
        except StopIteration:
            it = ds.create_tuple_iterator()
            data = next(it)
        _ = cell(*data)

    oracle_weights = get_all_params_np(model)

    # Recover: fresh model + apply FULL step_0 + delta chain
    print("  [T6] Recovering from FULL step_0 + delta chain...")
    model2, _, _ = make_gpt2xl_training(total_steps=1, device_id=device_id)
    ckpt2 = make_ckpt(device_id=device_id, pipeline_depth=8)
    ckpt2.delta_init(256, 128)

    try:
        recovery = ckpt2.recover(model2, target_step=steps)
        recovered_weights = get_all_params_np(model2)

        # Filter to params that exist in both
        common = set(oracle_weights) & set(recovered_weights)
        oracle_filt = {k: oracle_weights[k] for k in common}
        rec_filt = {k: recovered_weights[k] for k in common}

        nrmse_result = compute_nrmse(rec_filt, oracle_filt)
        results = {
            "status": "pass" if nrmse_result["median"] < 0.05 else "warn",
            "recovery": recovery,
            "nrmse": nrmse_result,
            "n_common_params": len(common),
        }
        print(f"  T6 recovery — base={recovery['base_step']} "
              f"deltas={recovery['n_deltas']} "
              f"NRMSE median={nrmse_result['median']:.6f} "
              f"mean={nrmse_result['mean']:.6f} "
              f"→ {results['status']}")
    except Exception as e:
        results = {"status": "fail", "error": str(e)}
        import traceback
        results["traceback"] = traceback.format_exc()
        print(f"  T6 FAIL — {e}")
    finally:
        ckpt2.cleanup()
        ckpt.cleanup()
    return results


# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Delta-checkpoint E2E (T4-T6)")
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--tests", type=str, default="all",
                        help="comma-separated: T4,T5,T6 or 'all'")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Delta-Checkpoint E2E Verification (T4–T6)")
    print(f"  Device: {args.device_id}  |  Steps: {args.steps}")
    print("=" * 60)

    ms.set_recursion_limit(10000)
    from experiments.common import init_env
    init_env(device_id=args.device_id)

    results = {
        "experiment": "delta_e2e_t4t6",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "device_id": args.device_id,
            "steps": args.steps,
            "block_size": BLOCK_SIZE,
            "top_k_frac": TOP_K_FRAC,
        },
        "tests": {},
    }

    test_names = args.tests.split(",") if args.tests != "all" else \
                  ["T4", "T5", "T6"]

    for tn in test_names:
        tn = tn.strip()
        print(f"\n{'—' * 40}")
        print(f"[{tn}]")
        try:
            if tn == "T4":
                r = test_faf_trigger(args.device_id, args.steps)
            elif tn == "T5":
                r = test_overhead(args.device_id, steps=50)
            elif tn == "T6":
                r = test_recovery(args.device_id, args.steps)
            else:
                print(f"  Unknown test: {tn}, skipping")
                continue
        except Exception as e:
            r = {"status": "error", "error": str(e)}
            import traceback
            r["traceback"] = traceback.format_exc()
            print(f"  ERROR: {e}")

        results["tests"][tn] = r

    # Summary
    passed = sum(1 for v in results["tests"].values()
                 if v.get("status") == "pass")
    total = len(results["tests"])
    results["summary"] = {"passed": passed, "total": total}
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed")
    for tn, tr in results["tests"].items():
        print(f"  {tn}: {tr.get('status', '?')}")
    print(f"{'=' * 60}")

    out = os.path.join(OUTPUT_DIR, "delta_e2e.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
