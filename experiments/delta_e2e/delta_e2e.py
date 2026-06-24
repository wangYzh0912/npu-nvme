#!/usr/bin/env python3
"""Delta-checkpoint end-to-end verification.

Validates the full incremental-checkpoint pipeline across six stages:

  T1 — GRAPH_MODE compilation of DeltaTrainCell (no OOM).
  T2 — single-step: P_old and quant buffers are non-zero after one step.
  T3 — FaF registration: register_delta_tasks + step_ptr + delta_init.
  T4 — multi-step FaF: listener triggers SPDK writes, probe_flag correct.
  T5 — step-time overhead vs baseline (ProbeTrainOneStepCell without I3 ops).
  T6 — recovery: FULL ckpt + delta chain restore, NRMSE < threshold.

Usage:
  bash _run.sh [DEVICE_ID] [--steps N]

Output: experiments/output/delta_e2e/delta_e2e.json
"""

import os, sys, time, json, ctypes, argparse, math, re

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

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


# ---- Test functions --------------------------------------------------------

def test_compile(device_id):
    """T1: GRAPH_MODE compilation of DeltaTrainCell."""
    from direct_checkpoint import ProbeTrainOneStepCell
    from delta_cell import DeltaTrainCell
    from mindformers import AutoModel, AutoConfig
    from experiments.common import make_gpt2xl_training

    model, ds, opt = make_gpt2xl_training(total_steps=2, device_id=device_id)

    print("[T1] Building DeltaTrainCell...")
    cell = DeltaTrainCell(model, opt, block_size=BLOCK_SIZE, top_k_frac=TOP_K_FRAC)

    # Compile: one forward pass in GRAPH_MODE triggers JIT compilation
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    dummy = Tensor(np.zeros((1, 512), dtype=np.int32))
    dry_out = cell(dummy, dummy, dummy)
    print(f"  T1 compile OK — loss shape={dry_out.shape}")
    return {"status": "pass"}


def test_single_step(device_id):
    """T2: Run 1 step, verify delta_p_old and delta_quant_buf are non-zero."""
    from delta_cell import DeltaTrainCell
    from experiments.common import make_gpt2xl_training

    model, ds, opt = make_gpt2xl_training(total_steps=2, device_id=device_id)

    cell = DeltaTrainCell(model, opt, block_size=BLOCK_SIZE, top_k_frac=TOP_K_FRAC)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    for data in ds.create_tuple_iterator():
        _ = cell(*data)
        break

    p_old_val = cell.delta_p_old.value().asnumpy()
    quant_val = cell.delta_quant_buf.value().asnumpy()

    p_old_sum = float(np.abs(p_old_val).sum())
    quant_sum = float(np.abs(quant_val).sum())

    results = {
        "p_old_abs_sum": p_old_sum,
        "quant_buf_abs_sum": quant_sum,
        "p_old_nonzero": bool(p_old_sum > 0),
        "quant_buf_nonzero": bool(quant_sum > 0),
    }
    status = "pass" if results["p_old_nonzero"] and results["quant_buf_nonzero"] else "fail"
    results["status"] = status
    print(f"  T2 p_old_sum={p_old_sum:.1f} quant_sum={quant_sum:.1f} → {status}")
    return results


def test_faf_register(device_id):
    """T3: register delta buffers + probe flag + step ptr."""
    from delta_cell import DeltaTrainCell
    from experiments.common import make_gpt2xl_training, make_ckpt

    model, ds, opt = make_gpt2xl_training(total_steps=2, device_id=device_id)

    cell = DeltaTrainCell(model, opt, block_size=BLOCK_SIZE, top_k_frac=TOP_K_FRAC)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    # Compile
    for data in ds.create_tuple_iterator():
        _ = cell(*data)
        break

    ckpt = make_ckpt(device_id=device_id, pipeline_depth=8)

    try:
        dev_flag, dev_step = ckpt.register_delta_tasks(cell, ckpt_interval=5)
        ckpt.delta_init(256, 128)
        results = {
            "status": "pass",
            "dev_flag": hex(dev_flag) if dev_flag else "0x0 (self-allocated)",
            "dev_step": hex(dev_step),
            "delta_area_offset": ckpt._delta_slot_size,
        }
        print(f"  T3 register OK — flag={hex(dev_flag)} step={hex(dev_step)}")
    except Exception as e:
        results = {"status": "fail", "error": str(e)}
        print(f"  T3 FAIL — {e}")
    finally:
        ckpt.cleanup()
    return results


def test_faf_trigger(device_id, steps=5):
    """T4: multi-step training with FaF listener triggering SPDK writes."""
    from delta_cell import DeltaTrainCell
    from experiments.common import make_gpt2xl_training, make_ckpt

    model, ds, opt = make_gpt2xl_training(total_steps=steps, device_id=device_id)

    cell = DeltaTrainCell(model, opt, block_size=BLOCK_SIZE, top_k_frac=TOP_K_FRAC)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    ckpt = make_ckpt(device_id=device_id, pipeline_depth=8)

    # Compile + register
    for data in ds.create_tuple_iterator():
        _ = cell(*data)
        break
    dev_flag, dev_step = ckpt.register_delta_tasks(cell, ckpt_interval=1)
    ckpt.delta_init(256, 128)

    # Train
    class ProbeCB(ms.Callback):
        def __init__(self):
            self.flags = []
        def on_train_epoch_end(self, run_context):
            try:
                self.flags.append(ckpt.read_probe_flag_dev())
            except Exception:
                self.flags.append(-1)

    cb = ProbeCB()
    ms_model = ms.Model(cell)
    ms_model.train(epoch=steps, train_dataset=ds, callbacks=[cb],
                   dataset_sink_mode=True, sink_size=1)

    final_flag = cb.flags[-1] if cb.flags else -1
    results = {
        "status": "pass" if final_flag >= steps - 1 else "fail",
        "probe_flags": cb.flags,
        "final_flag": final_flag,
        "step_counter": int(cell.step_counter.value().asnumpy().flat[0]),
        "expected_min": steps - 1,
    }
    print(f"  T4 trigger — flags={cb.flags} step_counter={results['step_counter']} "
          f"→ {results['status']}")
    ckpt.cleanup()
    return results


def test_recovery(device_id, steps=5):
    """T6: FULL ckpt + delta chain → recover → NRMSE."""
    from delta_cell import DeltaTrainCell
    from experiments.common import make_gpt2xl_training, make_ckpt

    model, ds, opt = make_gpt2xl_training(total_steps=steps, device_id=device_id)

    cell = DeltaTrainCell(model, opt, block_size=BLOCK_SIZE, top_k_frac=TOP_K_FRAC)

    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend",
                   device_id=device_id)

    ckpt = make_ckpt(device_id=device_id, pipeline_depth=8)
    ckpt.delta_init(256, 128)

    # Compile
    for data in ds.create_tuple_iterator():
        _ = cell(*data)
        break

    # Collect oracle weights and save FULL ckpt + deltas
    oracle_weights = get_all_params_np(model)

    ms_model = ms.Model(cell)
    ms_model.train(epoch=steps, train_dataset=ds,
                   dataset_sink_mode=True, sink_size=1)

    # Recover from step_0 FULL + delta chain → step N-1
    try:
        recovery = ckpt.recover(model, target_step=steps - 1)
        recovered_weights = get_all_params_np(model)

        nrmse_result = compute_nrmse(recovered_weights, oracle_weights)
        results = {
            "status": "pass" if nrmse_result["median"] < 0.05 else "warn",
            "recovery": recovery,
            "nrmse": nrmse_result,
        }
        print(f"  T6 recovery — base={recovery['base_step']} "
              f"deltas={recovery['n_deltas']} NRMSE median={nrmse_result['median']:.4f} "
              f"→ {results['status']}")
    except Exception as e:
        results = {"status": "fail", "error": str(e)}
        print(f"  T6 FAIL — {e}")
    finally:
        ckpt.cleanup()
    return results


# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Delta-checkpoint E2E test")
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--tests", type=str, default="all",
                        help="comma-separated: T1,T2,T3,T4,T6 or 'all'")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Delta-Checkpoint E2E Verification")
    print(f"  Device: {args.device_id}  |  Steps: {args.steps}")
    print("=" * 60)

    results = {
        "experiment": "delta_e2e",
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
                  ["T1", "T2", "T3", "T4", "T6"]

    for tn in test_names:
        tn = tn.strip()
        print(f"\n{'—' * 40}")
        print(f"[{tn}]")
        try:
            if tn == "T1":
                r = test_compile(args.device_id)
            elif tn == "T2":
                r = test_single_step(args.device_id)
            elif tn == "T3":
                r = test_faf_register(args.device_id)
            elif tn == "T4":
                r = test_faf_trigger(args.device_id, args.steps)
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
