#!/usr/bin/env python3
"""
B2: Probe Overhead Baseline — ProbeTrainOneStepCell + DirectCheckpoint init,
    but NO SPDK writes (listener mode=idle). Measure FaF infrastructure overhead.

This measures the cost of:
  - step_counter assign_add in the fused graph
  - C-layer listener thread polling step_counter every 10ms
  - registered tensor bookkeeping
  ...BUT WITHOUT the 3.1GB SPDK write pipeline.

Config:
  model: gpt2_xl, seq_len=1024, batch=1
  sink=TRUE, sink_size=10, 100 steps
  NPU_NVME_LISTENER_MODE=idle

Output: experiments/baselines/b2_probe_overhead_baseline.json

Usage:
  sudo su - root -c 'NPU_NVME_LISTENER_MODE=idle source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && \
    /root/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/baselines/b2_probe_overhead_baseline.py'
"""
import os, sys, time, json, ctypes
REPO = "/home/user7/npu-nvme"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

# SET BEFORE import: idle mode = listener spins but never triggers SPDK write
os.environ["NPU_NVME_LISTENER_MODE"] = "idle"

DEVICE_ID = 1
SEQ_LEN = 1024
TOTAL_STEPS = 100
CKPT_INTERVAL = 10
SINK_SIZE = 10
EPOCHS = TOTAL_STEPS // SINK_SIZE

print("=" * 60)
print("  B2: Probe overhead (listener idle, NO SPDK writes)")
print(f"  NPU_NVME_LISTENER_MODE=idle, sink=T s={SINK_SIZE}")
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

from direct_checkpoint import ProbeTrainOneStepCell, DirectCheckpoint
import direct_checkpoint as dc

cell = ProbeTrainOneStepCell(model, opt, enable_probe=True,
                             ckpt_interval=CKPT_INTERVAL)

dummy = Tensor(np.zeros((1, SEQ_LEN), dtype=np.int32), ms.int32)
warmup_fn = lambda: [cell(dummy[0:1], dummy[0:1], dummy[0:1]) for _ in range(2)]

print("[B2] SPDK init (warmup_fn + idle listener)...", flush=True)
ckpt = DirectCheckpoint(
    nvme_addr="0000:83:00.0", npu_device_id=DEVICE_ID,
    pipeline_depth=8, requested_chunk_size=4*1024*1024,
    enable_profiling=False, keep_last_n=3, slot_size_gb=10,
    warmup_fn=warmup_fn)

ckpt.register_tasks(model, step=0)

# Setup C-layer: flag (self-allocate) + step_counter
dev_step = cell.step_counter._data_ptr()
print(f"[B2] flag=0x0 (self-alloc), step_counter={hex(dev_step)}", flush=True)

dc_lib = dc.lib
# Pass NULL — C layer self-allocates probe_flag
rc = dc_lib.npu_nvme_set_probe_flag_ptr(ckpt.ctx, ctypes.c_void_p(0))
if rc != 0: raise RuntimeError(f"set_probe_flag_ptr failed: {rc}")
rc = dc_lib.npu_nvme_set_step_ptr(ckpt.ctx, ctypes.c_void_p(dev_step), CKPT_INTERVAL)
if rc != 0: raise RuntimeError(f"set_step_ptr failed: {rc}")

dev_flag = dc_lib.npu_nvme_get_probe_flag_dev_ptr(ckpt.ctx)
ckpt.probe_flag_ptr = dev_flag
print(f"[B2] Tasks registered. flag={hex(dev_flag) if dev_flag else '0x0'}", flush=True)

epoch_times_ms = []

class CB(ms.Callback):
    def on_train_epoch_begin(self, rc):
        self.t0 = time.perf_counter()
    def on_train_epoch_end(self, rc):
        et = (time.perf_counter() - self.t0) * 1000
        epoch_times_ms.append(et)
        epoch_num = len(epoch_times_ms)
        per_step = et / SINK_SIZE
        print(f"  [B2] Epoch {epoch_num:2d}/{EPOCHS} | {et:.0f}ms epoch | "
              f"~{per_step:.0f}ms/step", flush=True)

t_total = time.perf_counter()

ms_model = ms.Model(cell)
ms_model.train(epoch=EPOCHS, train_dataset=ds, callbacks=[CB()],
               dataset_sink_mode=True, sink_size=SINK_SIZE)

total_s = time.perf_counter() - t_total

compile_epoch_ms = epoch_times_ms[0] if epoch_times_ms else 0
warm_epochs_ms = epoch_times_ms[1:] if len(epoch_times_ms) > 1 else []
warm_step_times = [et / SINK_SIZE for et in warm_epochs_ms]
avg_step_ms = sum(warm_step_times) / len(warm_step_times) if warm_step_times else 0

# Safety: idle mode should produce zero SPDK writes
try:
    final_flag = ckpt.read_probe_flag_dev()
except Exception as e:
    final_flag = f"ERR:{e}"

print(f"\n{'='*60}")
print(f"  B2 Results: Probe infra overhead (no SPDK writes)")
print(f"{'='*60}")
print(f"  Epoch 1 (compile): {compile_epoch_ms:.0f}ms")
print(f"  Warm epochs:       {[f'{et:.0f}ms' for et in warm_epochs_ms]}")
print(f"  Avg per-step:      {avg_step_ms:.0f}ms")
print(f"  Probe flag:        {final_flag}")
print(f"  Total wall:        {total_s:.1f}s")

os.makedirs(REPO + "/experiments/baselines", exist_ok=True)
result = {
    "test": "B2 Probe Overhead Baseline",
    "listener_mode": "idle",
    "ckpt_interval": CKPT_INTERVAL,
    "total_steps": TOTAL_STEPS,
    "sink_size": SINK_SIZE,
    "epochs": EPOCHS,
    "total_elapsed_s": round(total_s, 1),
    "compile_epoch_ms": round(compile_epoch_ms, 0),
    "warm_epochs_ms": [round(et, 0) for et in warm_epochs_ms],
    "avg_step_ms": round(avg_step_ms, 0),
    "probe_flag": str(final_flag),
}
with open(REPO + "/experiments/baselines/b2_probe_overhead_baseline.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\n  Results saved to experiments/baselines/b2_probe_overhead_baseline.json")

ckpt.cleanup()
print("[B2] DONE.", flush=True)
