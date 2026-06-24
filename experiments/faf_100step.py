#!/usr/bin/env python3
"""
P1-3: FaF Full 100-Step End-to-End Test.
sink=TRUE, sink_size=10, CKPT_INTERVAL=10, 100 steps total.

Expected: 10 SPDK write triggers (steps 10/20/.../100), ~425ms/step
"""
import os, sys, time, json, ctypes
REPO = "/home/user7/npu-nvme"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

DEVICE_ID = 1
SEQ_LEN = 1024
TOTAL_STEPS = 100
CKPT_INTERVAL = 10
SINK_SIZE = 10

print("=" * 60)
print("  P1-3: FaF 100-Step E2E (sink=T s=10, ckpt_interval=10)")
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

from direct_checkpoint import ProbeTrainOneStepCell
cell = ProbeTrainOneStepCell(model, opt, enable_probe=True,
                             ckpt_interval=CKPT_INTERVAL)

# Warmup via DirectCheckpoint.__init__
dummy = Tensor(np.zeros((1, SEQ_LEN), dtype=np.int32), ms.int32)
warmup_fn = lambda: [cell(dummy[0:1], dummy[0:1], dummy[0:1]) for _ in range(2)]

from direct_checkpoint import DirectCheckpoint
import direct_checkpoint as dc

print("[P1-3] SPDK init (warmup_fn provided)...", flush=True)
ckpt = DirectCheckpoint(
    nvme_addr="0000:83:00.0", npu_device_id=DEVICE_ID,
    pipeline_depth=8, requested_chunk_size=4*1024*1024,
    enable_profiling=False, keep_last_n=3, slot_size_gb=10,
    warmup_fn=warmup_fn)

# Register tasks
ckpt.register_tasks(model, step=0)

# Setup C-layer listener
dev_flag = cell.flag._data_ptr()
dev_step = cell.step_counter._data_ptr()
print(f"[P1-3] flag={hex(dev_flag)} step_counter={hex(dev_step)}", flush=True)

dc_lib = dc.lib
rc = dc_lib.npu_nvme_set_probe_flag_ptr(ckpt.ctx, ctypes.c_void_p(dev_flag))
if rc != 0: raise RuntimeError(f"set_probe_flag_ptr failed: {rc}")
rc = dc_lib.npu_nvme_set_step_ptr(ckpt.ctx, ctypes.c_void_p(dev_step), CKPT_INTERVAL)
if rc != 0: raise RuntimeError(f"set_step_ptr failed: {rc}")

if dev_flag == 0 and hasattr(dc_lib, "npu_nvme_get_probe_flag_dev_ptr"):
    dev_flag = dc_lib.npu_nvme_get_probe_flag_dev_ptr(ckpt.ctx)
ckpt.probe_flag_ptr = dev_flag
print(f"[P1-3] Tasks registered. flag={hex(dev_flag)}", flush=True)

# sink=TRUE training: 10 epochs x 10 steps each = 100 steps
EPOCHS = TOTAL_STEPS // SINK_SIZE  # 10 epochs
epoch_times = []

class CB(ms.Callback):
    def on_train_epoch_begin(self, rc): self.t0 = time.perf_counter()
    def on_train_epoch_end(self, rc):
        et = time.perf_counter() - self.t0
        epoch_times.append(et)
        epoch_num = len(epoch_times)
        step_start = (epoch_num - 1) * SINK_SIZE + 1
        step_end = epoch_num * SINK_SIZE
        # Check flag: expected = floor(last_step / interval) = floor(max_step_in_epoch / CKPT_INTERVAL)
        # Each epoch covers steps [prev+1 .. epoch_num*SINK_SIZE]
        # Expected triggers by end of epoch = (epoch_num * SINK_SIZE) // CKPT_INTERVAL
        expected_triggers = (epoch_num * SINK_SIZE) // CKPT_INTERVAL
        try:
            cur_flag = ckpt.read_probe_flag_dev()
            safety = "OK" if cur_flag >= expected_triggers else f"WAIT({cur_flag}<{expected_triggers})"
        except Exception as e:
            safety = f"ERR:{e}"
        print(f"  [P1-3] Epoch {epoch_num:2d} | steps {step_start:3d}-{step_end:3d} | "
              f"{et:.1f}s ({et*1000/SINK_SIZE:.0f}ms/step) | flag={safety}", flush=True)

ms_model = ms.Model(cell)
print(f"\n[P1-3] Starting FaF 100-step training ({EPOCHS} epochs x {SINK_SIZE} steps)...\n", flush=True)
t_total = time.perf_counter()

ms_model.train(epoch=EPOCHS, train_dataset=ds, callbacks=[CB()],
               dataset_sink_mode=True, sink_size=SINK_SIZE)

total_elapsed = time.perf_counter() - t_total

# Final safety check
print("\n" + "=" * 60)
print("  P1-3 Results")
print("=" * 60)
try:
    final_flag = ckpt.read_probe_flag_dev()
    expected_ckpts = TOTAL_STEPS // CKPT_INTERVAL  # 10
    if final_flag >= expected_ckpts:
        print(f"  SAFETY CHECK: PASSED (flag={final_flag} >= {expected_ckpts})")
        safety_status = "PASSED"
    else:
        print(f"  SAFETY CHECK: FAILED (flag={final_flag} < {expected_ckpts}, "
              f"missing {expected_ckpts - final_flag} CKPTs)")
        safety_status = "FAILED"
except Exception as e:
    print(f"  SAFETY CHECK: ERROR ({e})")
    final_flag = -1
    expected_ckpts = TOTAL_STEPS // CKPT_INTERVAL
    safety_status = "ERROR"

# Epoch stats
if len(epoch_times) >= 2:
    warm_epochs = epoch_times[1:]  # skip first (compile)
    warm_avg_step_ms = sum(warm_epochs) * 1000 / (len(warm_epochs) * SINK_SIZE)
    print(f"  Total: {total_elapsed:.1f}s for {TOTAL_STEPS} steps")
    print(f"  Epoch times: {[f'{t:.1f}s' for t in epoch_times]}")
    print(f"  Avg per-step (excl epoch 1): {warm_avg_step_ms:.0f}ms")
else:
    warm_avg_step_ms = 0

# Save results
os.makedirs(REPO + "/experiments/output", exist_ok=True)
result = {
    "test": "P1-3 FaF 100-Step E2E",
    "total_steps": TOTAL_STEPS,
    "ckpt_interval": CKPT_INTERVAL,
    "sink_size": SINK_SIZE,
    "epochs": EPOCHS,
    "total_elapsed_s": round(total_elapsed, 1),
    "epoch_times_s": [round(t, 1) for t in epoch_times],
    "avg_per_step_ms": round(warm_avg_step_ms, 0),
    "final_flag": int(final_flag),
    "expected_ckpts": expected_ckpts,
    "safety": safety_status,
}
with open(REPO + "/experiments/output/faf_100step_result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\n  Results saved to experiments/output/faf_100step_result.json")

ckpt.cleanup()
print("[P1-3] DONE.", flush=True)
