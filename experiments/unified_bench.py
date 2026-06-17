#!/usr/bin/env python3
"""
P0-U6: Unified benchmark — single script, all configs, clean labels.

Tests 4 configs sequentially, each as independent process (separate sudo invocation
to avoid OOM). All use identical GPT-2 XL, seq_len=1024, batch=1.

Config matrix:
  A: sink=FALSE, enable_probe=False  → pure training baseline
  B: sink=FALSE, enable_probe=True   → step_counter in graph (no SPDK, no C-layer)
  C: sink=TRUE,  enable_probe=False  → fused-graph baseline
  D: sink=TRUE,  enable_probe=True   → step_counter + C-layer listener + SPDK writes

Each config runs TOTAL_STEPS=20 steps. For sink=TRUE we use sink_size=10 so GE
only fuses 10 steps per epoch (reducing memory pressure).

Output: experiments/output/unified_bench.json
"""
import os, sys, time, json, ctypes
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO, "python"))
os.chdir(REPO)

DEVICE_ID = 1
TRAIN_MR = os.path.join(REPO, "dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord")
TOTAL_STEPS = 20
SEQ_LEN = 1024
CKPT_INTERVAL = 5  # trigger every 5 steps
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def build():
    """Common model/dataset/optimizer construction."""
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)

    ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(TOTAL_STEPS)

    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
    return model, ds, opt


def run_sink_false(enable_probe):
    """
    sink=FALSE: per-step callbacks work. Measure precise per-step timing.
    Skip first 2 steps (warmup).
    """
    label = f"sinkF_probe{1 if enable_probe else 0}"
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    model, ds, opt = build()
    from direct_checkpoint import ProbeTrainOneStepCell
    cell = ProbeTrainOneStepCell(
        model, opt, None, 0,
        enable_probe=enable_probe, probe_mode="end", ckpt_interval=CKPT_INTERVAL)

    step_times = []
    t_last = [0]

    class CB(ms.Callback):
        def on_train_step_begin(self, rc):
            t_last[0] = time.perf_counter()
        def on_train_step_end(self, rc):
            step_times.append(time.perf_counter() - t_last[0])

    ms_model = ms.Model(cell)
    t0 = time.perf_counter()
    ms_model.train(epoch=1, train_dataset=ds, callbacks=[CB()], dataset_sink_mode=False)
    t1 = time.perf_counter()

    arr = np.array(step_times[2:])  # skip warmup
    return {
        "label": label,
        "sink": False,
        "enable_probe": enable_probe,
        "total_s": round(t1 - t0, 2),
        "steps": len(step_times),
        "per_step": {
            "mean_ms": round(arr.mean() * 1000, 1),
            "std_ms": round(arr.std() * 1000, 1),
            "p99_ms": round(np.percentile(arr, 99) * 1000, 1) if len(arr) > 1 else -1,
        }
    }


def run_sink_true(enable_probe):
    """
    sink=TRUE: epoch callbacks only. Use sink_size=10 (2 epochs of 10 steps each).
    Measure epoch_begin→epoch_end, decompose compile vs training via epoch deltas.
    """
    label = f"sinkT_probe{1 if enable_probe else 0}"
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    model, ds, opt = build()
    from direct_checkpoint import ProbeTrainOneStepCell
    cell = ProbeTrainOneStepCell(
        model, opt, None, 0,
        enable_probe=enable_probe, probe_mode="end", ckpt_interval=CKPT_INTERVAL)

    epoch_times = []

    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc):
            self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times.append(time.perf_counter() - self.t0)

    ms_model = ms.Model(cell)
    t0 = time.perf_counter()
    # 2 epochs of 10 steps each = 20 total steps
    ms_model.train(epoch=2, train_dataset=ds, callbacks=[CB()],
                   dataset_sink_mode=True, sink_size=10)
    t1 = time.perf_counter()

    result = {
        "label": label,
        "sink": True,
        "enable_probe": enable_probe,
        "total_s": round(t1 - t0, 2),
        "epochs": 2,
        "sink_size": 10,
        "total_steps": TOTAL_STEPS,
        "epoch_s": [round(e, 2) for e in epoch_times],
    }

    # Epoch 1 = compile + 10 steps; Epoch 2 = 10 steps (no recompile)
    if len(epoch_times) >= 2:
        e2_per_step_ms = epoch_times[1] * 1000 / 10
        result["per_step_from_epoch2_ms"] = round(e2_per_step_ms, 1)
        if epoch_times[0] > epoch_times[1]:
            result["compile_overhead_s"] = round(epoch_times[0] - epoch_times[1], 1)

    return result


def run_sink_true_faf():
    """
    sink=TRUE, enable_probe=True, WITH C-layer listener + SPDK writes.
    Full Fire-and-Forget stack.
    """
    label = "sinkT_FaF"
    import direct_checkpoint
    from direct_checkpoint import ProbeTrainOneStepCell, DirectCheckpoint

    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    model, ds, opt = build()
    cell = ProbeTrainOneStepCell(
        model, opt, None, 0,
        enable_probe=True, probe_mode="end", ckpt_interval=CKPT_INTERVAL)

    # Dry-run to allocate
    dummy = Tensor(np.zeros((1, SEQ_LEN), dtype=np.int32), ms.int32)
    cell(dummy[0:1], dummy[0:1], dummy[0:1])

    # SPDK init + task registration
    ckpt = DirectCheckpoint(
        nvme_addr="0000:83:00.0", npu_device_id=DEVICE_ID,
        pipeline_depth=8, requested_chunk_size=4*1024*1024,
        enable_profiling=False, keep_last_n=3, slot_size_gb=10)
    ckpt.register_tasks(model, step=0)

    dev_flag = cell.flag._data_ptr()
    dev_step = cell.step_counter._data_ptr()

    dc_lib = direct_checkpoint.lib
    dc_lib.npu_nvme_set_probe_flag_ptr(ckpt.ctx, ctypes.c_void_p(dev_flag))
    if hasattr(dc_lib, "npu_nvme_set_step_ptr"):
        dc_lib.npu_nvme_set_step_ptr(ckpt.ctx, ctypes.c_void_p(dev_step), CKPT_INTERVAL)

    if dev_flag == 0 and hasattr(dc_lib, "npu_nvme_get_probe_flag_dev_ptr"):
        dev_flag = dc_lib.npu_nvme_get_probe_flag_dev_ptr(ckpt.ctx)

    epoch_times = []

    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc):
            self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times.append(time.perf_counter() - self.t0)

    ms_model = ms.Model(cell)
    t0 = time.perf_counter()
    ms_model.train(epoch=2, train_dataset=ds, callbacks=[CB()],
                   dataset_sink_mode=True, sink_size=10)
    t1 = time.perf_counter()

    result = {
        "label": label,
        "sink": True,
        "enable_probe": True,
        "faf_full_stack": True,
        "total_s": round(t1 - t0, 2),
        "epochs": 2,
        "sink_size": 10,
        "total_steps": TOTAL_STEPS,
        "epoch_s": [round(e, 2) for e in epoch_times],
    }

    if len(epoch_times) >= 2:
        e2_per_step_ms = epoch_times[1] * 1000 / 10
        result["per_step_from_epoch2_ms"] = round(e2_per_step_ms, 1)
        diff = epoch_times[0] - epoch_times[1]
        result["compile_plus_init_s"] = round(diff, 1) if diff > 0 else round(diff, 1)

    # Safety check
    try:
        final_flag = ckpt.read_probe_flag_dev()
        result["final_flag"] = int(final_flag)
        result["safety"] = "PASSED" if final_flag >= TOTAL_STEPS // CKPT_INTERVAL else "FAILED"
    except Exception as e:
        result["safety"] = f"error: {e}"

    ckpt.cleanup()
    return result


def main():
    results = {}

    # Config A: sink=FALSE, no probe
    print("\n" + "=" * 60)
    print("CONFIG A: sink=FALSE, enable_probe=False (pure baseline)")
    print("=" * 60)
    results["A_sinkF_noProbe"] = run_sink_false(enable_probe=False)

    # Config B: sink=FALSE, step_counter in graph
    print("\n" + "=" * 60)
    print("CONFIG B: sink=FALSE, enable_probe=True (step_counter, no SPDK)")
    print("=" * 60)
    results["B_sinkF_probe"] = run_sink_false(enable_probe=True)

    # Config C: sink=TRUE, no probe
    print("\n" + "=" * 60)
    print("CONFIG C: sink=TRUE, enable_probe=False (sink_size=10)")
    print("=" * 60)
    results["C_sinkT_noProbe"] = run_sink_true(enable_probe=False)

    # Config D: sink=TRUE, step_counter only (no C-layer)
    print("\n" + "=" * 60)
    print("CONFIG D: sink=TRUE, enable_probe=True (step_counter only)")
    print("=" * 60)
    results["D_sinkT_probeOnly"] = run_sink_true(enable_probe=True)

    # Config E: Full FaF (needs separate process to avoid OOM)
    # Run via os.system to get clean process
    print("\n" + "=" * 60)
    print("CONFIG E: sink=TRUE, Full FaF (C-layer + SPDK)")
    print("=" * 60)

    # Save intermediate results before E
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "unified_bench.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nConfigs A-D done. Config E must be run separately due to NPU memory.")
    print("Intermediate results saved to output/unified_bench.json")

    return 0


if __name__ == "__main__":
    main()
