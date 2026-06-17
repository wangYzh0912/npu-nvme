#!/usr/bin/env python3
"""
P0-U6: Clean sink=TRUE performance baseline (enable_probe=False).

Measures pure training wall-clock time with NO checkpointing overhead:
  - No step_counter in fused graph
  - No C layer listener polling
  - No SPDK writes
  - No DirectCheckpoint at all

Compared with fire_and_forget.py to determine:
  1. Baseline step latency (sink=TRUE)
  2. Overhead of step_counter increment (graph-level)
  3. Overhead of C layer listener polling (system-level)

Parameters match fire_and_forget.py exactly for comparability.
"""
import os, sys, time, json, warnings
import numpy as np
import mindspore as ms
from mindspore import nn, context, ops, Tensor

warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "output", "log", "rank_0"), exist_ok=True)

import direct_checkpoint
from direct_checkpoint import ProbeTrainOneStepCell

DEVICE_ID = 1
TRAIN_MR  = "/home/user7/npu-nvme/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord"
TOTAL_STEPS = 30
SEQ_LEN = 1024
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


class BaselineCallback(ms.Callback):
    """Minimal callback for baseline timing — no checkpoint logic at all."""
    def __init__(self):
        super().__init__()
        self.epoch_t0 = 0

    def on_train_epoch_begin(self, run_context):
        self.epoch_t0 = time.perf_counter()
        print(f"\n  [Baseline] Epoch started at t={self.epoch_t0:.1f}", flush=True)

    def on_train_epoch_end(self, run_context):
        elapsed = time.perf_counter() - self.epoch_t0
        avg_ms = elapsed * 1000 / TOTAL_STEPS
        print(f"  [Baseline] Epoch completed in {elapsed:.1f}s "
              f"({avg_ms:.1f} ms/step avg)", flush=True)

        results = {
            "test": "P0-U6 Clean sink=TRUE Baseline (enable_probe=False)",
            "dataset_sink_mode": True,
            "enable_probe": False,
            "total_steps": TOTAL_STEPS,
            "epoch_elapsed_s": round(elapsed, 1),
            "avg_step_ms": round(avg_ms, 1),
        }
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "sink_baseline.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to output/sink_baseline.json")


def main():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    print("[Baseline] Building GPT-2 XL model...", flush=True)
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    base_model = AutoModel.from_config(cfg)

    total_bytes = sum(
        int(np.prod(p.shape)) * ms.dtype_to_nptype(p.dtype)().itemsize
        for p in base_model.get_parameters())
    print(f"[Baseline] Model params: {total_bytes / 1024 / 1024:.1f} MB", flush=True)

    train_ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=True)
    train_ds = train_ds.batch(1, drop_remainder=True).take(TOTAL_STEPS)

    optimizer = nn.AdamWeightDecay(base_model.trainable_params(), learning_rate=1e-5)

    # enable_probe=False → construct() uses pure training branch
    # No step_counter, no C listener, no SPDK, no DirectCheckpoint
    train_cell = ProbeTrainOneStepCell(
        base_model, optimizer, None, 0, enable_probe=False, probe_mode="end")
    cb = BaselineCallback()
    ms_model = ms.Model(train_cell)

    print(f"\n[Baseline] === Starting Clean sink=TRUE Baseline ===\n", flush=True)
    t_total = time.perf_counter()
    try:
        ms_model.train(epoch=1, train_dataset=train_ds, callbacks=[cb],
                       dataset_sink_mode=True)
    except Exception as e:
        print(f"\n[Baseline] FATAL: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1

    elapsed = time.perf_counter() - t_total
    print(f"\n[Baseline] Total wall time: {elapsed:.1f}s for {TOTAL_STEPS} steps", flush=True)
    print("[Baseline] DONE.", flush=True)
    return 0


if __name__ == "__main__":
    main()
