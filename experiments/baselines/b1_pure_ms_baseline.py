#!/usr/bin/env python3
"""
B1: Pure MindSpore Baseline — No DirectCheckpoint, no SPDK, no probe.
Goal: measure native MS training step time to quantify our overhead.

Config:
  model: gpt2_xl, seq_len=1024, batch=1
  sink=TRUE, sink_size=10, 100 steps (10 epochs × 10)

IMPORTANT: With sink=TRUE + sink_size=10, MS batches 10 steps into a single
fused graph per epoch. on_train_step_end fires only at epoch boundaries, not
per logical step.  We therefore measure epoch wall-clock and divide by sink_size
to get per-step time.

Output: experiments/baselines/b1_pure_ms_baseline.json

Usage:
  sudo su - root -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && \
    /root/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/baselines/b1_pure_ms_baseline.py'
"""
import os, sys, time, json
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

DEVICE_ID = 1
SEQ_LEN = 1024
TOTAL_STEPS = 100
SINK_SIZE = 10
EPOCHS = TOTAL_STEPS // SINK_SIZE

print("=" * 60)
print("  B1: Pure MS baseline (no DirectCheckpoint, no SPDK)")
print(f"  sink=TRUE sink_size={SINK_SIZE}, {TOTAL_STEPS} logical steps")
print("=" * 60)

ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
ms.common.set_seed(42)

from mindformers import AutoModel, AutoConfig
cfg = AutoConfig.from_pretrained("gpt2_xl")
cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
model = AutoModel.from_config(cfg)

ds = ms.dataset.MindDataset(
    REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
    shuffle=True)
ds = ds.batch(1, drop_remainder=True).take(TOTAL_STEPS)

opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

class PlainTrainOneStep(nn.Cell):
    def __init__(self, network, optimizer):
        super().__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.grad_fn = ops.value_and_grad(self.network, grad_position=None,
                                           weights=self.optimizer.parameters)
        self.depend = ops.Depend()

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        opt_res = self.optimizer(grads)
        loss = self.depend(loss, opt_res)
        return loss

cell = PlainTrainOneStep(model, opt)

epoch_times_ms = []

class CB(ms.Callback):
    def on_train_epoch_begin(self, rc):
        self.t0 = time.perf_counter()
    def on_train_epoch_end(self, rc):
        et = (time.perf_counter() - self.t0) * 1000  # epoch wall-clock ms
        epoch_times_ms.append(et)
        epoch_num = len(epoch_times_ms)
        per_step = et / SINK_SIZE
        print(f"  [B1] Epoch {epoch_num:2d}/{EPOCHS} | {et:.0f}ms epoch | "
              f"~{per_step:.0f}ms/step", flush=True)

t_total = time.perf_counter()

ms_model = ms.Model(cell)
ms_model.train(epoch=EPOCHS, train_dataset=ds, callbacks=[CB()],
               dataset_sink_mode=True, sink_size=SINK_SIZE)

total_s = time.perf_counter() - t_total

# Epoch 1 includes compilation — exclude from warm avg
compile_epoch_ms = epoch_times_ms[0] if epoch_times_ms else 0
warm_epochs_ms = epoch_times_ms[1:] if len(epoch_times_ms) > 1 else []
warm_step_times = [et / SINK_SIZE for et in warm_epochs_ms]
avg_step_ms = sum(warm_step_times) / len(warm_step_times) if warm_step_times else 0

print(f"\n{'='*60}")
print(f"  B1 Results: Pure MS (no SPDK overhead)")
print(f"{'='*60}")
print(f"  Epoch 1 (compile): {compile_epoch_ms:.0f}ms")
print(f"  Warm epochs:       {[f'{et:.0f}ms' for et in warm_epochs_ms]}")
print(f"  Avg per-step:      {avg_step_ms:.0f}ms")
print(f"  P50 per-step:      {sorted(warm_step_times)[len(warm_step_times)//2]:.0f}ms" if warm_step_times else "  N/A")
print(f"  Total wall:        {total_s:.1f}s")
print(f"  Logical steps:     {len(warm_epochs_ms) * SINK_SIZE} (warm)")

os.makedirs(REPO + "/experiments/baselines", exist_ok=True)
result = {
    "test": "B1 Pure MS Baseline",
    "total_steps": TOTAL_STEPS,
    "sink_size": SINK_SIZE,
    "epochs": EPOCHS,
    "total_elapsed_s": round(total_s, 1),
    "compile_epoch_ms": round(compile_epoch_ms, 0),
    "warm_epochs_ms": [round(et, 0) for et in warm_epochs_ms],
    "avg_step_ms": round(avg_step_ms, 0),
    "p50_step_ms": round(sorted(warm_step_times)[len(warm_step_times)//2], 0) if warm_step_times else 0,
}
with open(REPO + "/experiments/baselines/b1_pure_ms_baseline.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\n  Results saved to experiments/baselines/b1_pure_ms_baseline.json")
print("[B1] DONE.", flush=True)
