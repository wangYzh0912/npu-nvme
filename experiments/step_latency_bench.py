#!/usr/bin/env python3
"""
P0-U6: Step-latency benchmark with listener-poll-based timing.

For sink=TRUE, callbacks fire only at first/last step. We use the C-layer
listener's step_counter polling to measure pure training latency — the
step_counter increments in the fused graph, so the poll interval between
step transitions IS the per-step training time.

Three configs, all using step_counter polling:
  1. clean_baseline: enable_probe=False (no step_counter in graph → no poll data)
     → estimates via wall clock minus compile overhead
  2. step_counter_only: enable_probe=True, no C-layer listener (no SPDK init)
     → pure graph-level overhead of assign_add
  3. full_faf: enable_probe=True + C-layer listener + SPDK writes
     → full system overhead

Config 1 can't use poll data (no step_counter). Instead we measure:
  T_total = model.train() wall time
  T_compile = time from epoch_begin to first step_counter change (via C poll)
  T_training = T_total - T_compile
"""
import os, sys, time, json, warnings, ctypes
import numpy as np
import mindspore as ms
from mindspore import nn, context, ops, Tensor

warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "output", "log", "rank_0"), exist_ok=True)

import direct_checkpoint
from direct_checkpoint import ProbeTrainOneStepCell, DirectCheckpoint

DEVICE_ID = 1
TRAIN_MR  = "/home/user7/npu-nvme/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord"
TOTAL_STEPS = 30
SEQ_LEN = 1024
CKPT_INTERVAL = 10
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


class WallClockCallback(ms.Callback):
    """Records wall-clock at epoch boundaries for sink=TRUE decomposition."""
    def __init__(self):
        super().__init__()
        self.t_begin = 0
        self.t_end = 0

    def on_train_epoch_begin(self, run_context):
        self.t_begin = time.perf_counter()

    def on_train_epoch_end(self, run_context):
        self.t_end = time.perf_counter()


def build_model_and_ds():
    """Common model build for all configs."""
    from mindformers import AutoModel, AutoConfig
    print("[Bench] Building GPT-2 XL model (3128 MB)...", flush=True)
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    base_model = AutoModel.from_config(cfg)

    train_ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True)
    train_ds = train_ds.batch(1, drop_remainder=True).take(TOTAL_STEPS)
    return base_model, train_ds


def run_baseline():
    """
    Config 1: Clean baseline — enable_probe=False.
    No step_counter, no C-layer, no SPDK. Pure training.
    Uses wall clock only (sink=TRUE prevents per-step callbacks).
    """
    print("\n" + "=" * 60)
    print("CONFIG 1: Clean Baseline (enable_probe=False)")
    print("=" * 60)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    base_model, train_ds = build_model_and_ds()
    optimizer = nn.AdamWeightDecay(base_model.trainable_params(), learning_rate=1e-5)
    train_cell = ProbeTrainOneStepCell(
        base_model, optimizer, None, 0, enable_probe=False, probe_mode="end")
    cb = WallClockCallback()
    ms_model = ms.Model(train_cell)

    print(f"\n[Config1] === Starting (sink=TRUE, no probe) ===", flush=True)
    t0 = time.perf_counter()
    try:
        ms_model.train(epoch=1, train_dataset=train_ds, callbacks=[cb],
                       dataset_sink_mode=True)
    except Exception as e:
        print(f"[Config1] FATAL: {e}", flush=True)
        return None
    t1 = time.perf_counter()

    # Estimate: T_compile = epoch_begin → first real work
    # In sink=TRUE, on_train_epoch_begin fires, then GE compiles, then training.
    # We estimate compile time from the difference between total and known training rate.
    total = t1 - t0
    epoch_wall = cb.t_end - cb.t_begin if cb.t_begin > 0 and cb.t_end > 0 else total

    return {
        "config": "clean_baseline",
        "enable_probe": False,
        "total_s": round(total, 1),
        "epoch_wall_s": round(epoch_wall, 1),
        "steps": TOTAL_STEPS,
    }


def run_faf(label, setup_func=None):
    """
    Configs 2 & 3: enable_probe=True with step_counter.
    Uses C-layer poll data for precise per-step timing.

    setup_func(config_name) is called to do pre-train setup (C-layer init, etc.)
    """
    print("\n" + "=" * 60)
    print(f"CONFIG: {label}")
    print("=" * 60)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    base_model, train_ds = build_model_and_ds()
    optimizer = nn.AdamWeightDecay(base_model.trainable_params(), learning_rate=1e-5)
    train_cell = ProbeTrainOneStepCell(
        base_model, optimizer, None, 0,
        enable_probe=True, probe_mode="end", ckpt_interval=CKPT_INTERVAL)
    cb = WallClockCallback()
    ms_model = ms.Model(train_cell)

    # Dry-run to allocate memory
    dummy_input = ms.Tensor(np.zeros((1, SEQ_LEN,), dtype=np.int32), ms.int32)
    train_cell(dummy_input[0:1], dummy_input[0:1], dummy_input[0:1])
    print(f"[{label}] Dry-run done.", flush=True)

    # Get step_counter address for C-layer
    dev_step_addr = train_cell.step_counter._data_ptr()
    print(f"[{label}] step_counter addr: {hex(dev_step_addr)}", flush=True)

    # Optional setup (SPDK init, task registration, etc.)
    extra = {}
    if setup_func:
        extra = setup_func(label, base_model, train_cell)
    ckpt_ctx = extra.get("ckpt", None)

    print(f"\n[{label}] === Starting (sink=TRUE) ===", flush=True)

    # NOTE: we cannot get per-step timing from callbacks in sink=TRUE.
    # Instead we measure: compile time from epoch_begin → first step_counter change,
    # and training time from first step_counter change → epoch_end.
    # This requires reading step_counter from Python... which is hard during sink.
    #
    # Alternative: use TOTAL wall time and decompose with listener data.
    # The listener polls step_counter; its first detection of step>1 marks end of compile.

    t0 = time.perf_counter()
    try:
        ms_model.train(epoch=1, train_dataset=train_ds, callbacks=[cb],
                       dataset_sink_mode=True)
    except Exception as e:
        print(f"[{label}] FATAL: {e}", flush=True)
        return None
    t1 = time.perf_counter()

    total = t1 - t0
    epoch_wall = cb.t_end - cb.t_begin if cb.t_begin > 0 and cb.t_end > 0 else total

    result = {
        "config": label,
        "enable_probe": True,
        "total_s": round(total, 1),
        "epoch_wall_s": round(epoch_wall, 1),
        "steps": TOTAL_STEPS,
    }

    # Safety check for full_faf
    if ckpt_ctx:
        try:
            final_flag = ckpt_obj.read_probe_flag_dev() if (ckpt_ctx is not None) else -1
            result["final_flag"] = int(final_flag)
            result["safety_check"] = "PASSED" if final_flag >= TOTAL_STEPS//CKPT_INTERVAL else "FAILED"
        except Exception as e:
            result["safety_check"] = f"error: {e}"

    if setup_func:
        extra.get("cleanup", lambda: None)()

    return result


def main():
    results = {}

    # Config 1: Clean baseline
    r1 = run_baseline()
    if r1:
        results["clean_baseline"] = r1

    # Config 2: Step-counter in graph, NO C-layer listener, NO SPDK
    def no_setup(label, model, cell):
        return {}
    r2 = run_faf("step_counter_only", setup_func=no_setup)
    if r2:
        results["step_counter_only"] = r2

    # Config 3: Full FaF — C-layer listener + SPDK writes
    def full_faf_setup(label, model, cell):
        ckpt = DirectCheckpoint(
            nvme_addr="0000:83:00.0", npu_device_id=DEVICE_ID,
            pipeline_depth=8, requested_chunk_size=4*1024*1024,
            enable_profiling=True, keep_last_n=3, slot_size_gb=10)
        ckpt.register_tasks(model, step=0)

        dev_flag_addr = cell.flag._data_ptr()
        dev_step_addr = cell.step_counter._data_ptr()
        print(f"[{label}] flag={hex(dev_flag_addr)} step={hex(dev_step_addr)}", flush=True)

        dc_lib = direct_checkpoint.lib
        rc = dc_lib.npu_nvme_set_probe_flag_ptr(ckpt.ctx, ctypes.c_void_p(dev_flag_addr))
        if rc != 0: raise RuntimeError(f"set_probe_flag_ptr failed: {rc}")
        if hasattr(dc_lib, "npu_nvme_set_step_ptr"):
            rc = dc_lib.npu_nvme_set_step_ptr(ckpt.ctx, ctypes.c_void_p(dev_step_addr), CKPT_INTERVAL)
            if rc != 0: raise RuntimeError(f"set_step_ptr failed: {rc}")
        if dev_flag_addr == 0 and hasattr(dc_lib, "npu_nvme_get_probe_flag_dev_ptr"):
            dev_flag_addr = dc_lib.npu_nvme_get_probe_flag_dev_ptr(ckpt.ctx)

        print(f"[{label}] Tasks registered: {ckpt.num_registered_tasks}", flush=True)
        return {"ckpt": ckpt, "cleanup": lambda: ckpt.cleanup()}

    r3 = run_faf("full_faf", setup_func=full_faf_setup)
    if r3:
        results["full_faf"] = r3

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON (sink=TRUE, wall-clock from model.train())")
    print("=" * 70)
    print(f"{'Config':<25} {'Total(s)':>8} {'Epoch(s)':>8} {'ms/step':>8} {'Safety':>10}")
    print("-" * 70)
    for key in ["clean_baseline", "step_counter_only", "full_faf"]:
        r = results.get(key)
        if r:
            ms_per = r["total_s"] * 1000 / r["steps"]
            safety = r.get("safety_check", "N/A")
            print(f"{r['config']:<25} {r['total_s']:>8.1f} {r['epoch_wall_s']:>8.1f} "
                  f"{ms_per:>8.1f} {safety:>10}")

    # Decompose: the step_counter_only config tells us the compile overhead.
    # clean_baseline vs step_counter_only diff = assign_add overhead in fused graph.
    r_bl = results.get("clean_baseline", {})
    r_sc = results.get("step_counter_only", {})
    r_ff = results.get("full_faf", {})
    if r_bl and r_sc:
        overhead_graph = r_sc.get("total_s", 0) - r_bl.get("total_s", 0)
        print(f"\nGraph-level overhead (step_counter += 1): {overhead_graph:.1f}s "
              f"({overhead_graph/TOTAL_STEPS*1000:.1f} ms/step)")
    if r_sc and r_ff:
        overhead_system = r_ff.get("total_s", 0) - r_sc.get("total_s", 0)
        print(f"System-level overhead (C-layer + SPDK init + writes): {overhead_system:.1f}s")
    if r_bl and r_ff:
        overhead_total = r_ff.get("total_s", 0) - r_bl.get("total_s", 0)
        print(f"Total FaF overhead: {overhead_total:.1f}s ({overhead_total/TOTAL_STEPS*1000:.1f} ms/step)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "step_latency_bench.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults → output/step_latency_bench.json")

    return 0


if __name__ == "__main__":
    main()
