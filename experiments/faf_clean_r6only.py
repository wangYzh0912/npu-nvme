import os, sys, time, json, ctypes
REPO = "/home/user7/npu-nvme"
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

DEVICE_ID = 1
SEQ_LEN = 1024
TOTAL_STEPS = 20
CKPT_INTERVAL = 5
SINK_SIZE = 10

print("=" * 60)
print("  R6: sink=TRUE sink_size=10, Full FaF (SPDK+listener+step_counter)")
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
cell = ProbeTrainOneStepCell(model, opt, None, 0,
                             enable_probe=True, probe_mode="end",
                             ckpt_interval=CKPT_INTERVAL)

# SPDK init with warmup_fn: DirectCheckpoint.__init__ runs warmup BEFORE spdk_env_init()
# so MS runtime starts in clean process state (avoids +304% overhead, see P0-U7).
from direct_checkpoint import DirectCheckpoint
import direct_checkpoint as dc

dummy = Tensor(np.zeros((1, SEQ_LEN), dtype=np.int32), ms.int32)
warmup_fn = lambda: [cell(dummy[0:1], dummy[0:1], dummy[0:1]) for _ in range(2)]

print("[R6] SPDK init (warmup_fn provided, warmup runs before SPDK)...", flush=True)
ckpt = DirectCheckpoint(
    nvme_addr="0000:83:00.0", npu_device_id=DEVICE_ID,
    pipeline_depth=8, requested_chunk_size=4*1024*1024,
    enable_profiling=False, keep_last_n=3, slot_size_gb=10,
    warmup_fn=warmup_fn)

# Register tasks
ckpt.register_tasks(model, step=0)

# Setup C-layer listener ptrs
dev_flag = cell.flag._data_ptr()
dev_step = cell.step_counter._data_ptr()
print(f"[R6] flag={hex(dev_flag)} step_counter={hex(dev_step)}", flush=True)

dc_lib = dc.lib
rc = dc_lib.npu_nvme_set_probe_flag_ptr(ckpt.ctx, ctypes.c_void_p(dev_flag))
if rc != 0: raise RuntimeError(f"set_probe_flag_ptr failed: {rc}")
rc = dc_lib.npu_nvme_set_step_ptr(ckpt.ctx, ctypes.c_void_p(dev_step), CKPT_INTERVAL)
if rc != 0: raise RuntimeError(f"set_step_ptr failed: {rc}")

if dev_flag == 0 and hasattr(dc_lib, "npu_nvme_get_probe_flag_dev_ptr"):
    dev_flag = dc_lib.npu_nvme_get_probe_flag_dev_ptr(ckpt.ctx)
# P0-2 fix: always store the actual probe_flag_ptr so read_probe_flag_dev() works
ckpt.probe_flag_ptr = dev_flag
print(f"[R6] Tasks registered. flag={hex(dev_flag)}", flush=True)

# sink=TRUE training
epoch_times = []
class CB(ms.Callback):
    def on_train_epoch_begin(self, rc): self.t0 = time.perf_counter()
    def on_train_epoch_end(self, rc):
        epoch_times.append(time.perf_counter() - self.t0)

ms_model = ms.Model(cell)
print("[R6] Starting sink=TRUE FaF training (2 epochs x 10 steps)...", flush=True)
ms_model.train(epoch=2, train_dataset=ds, callbacks=[CB()],
               dataset_sink_mode=True, sink_size=SINK_SIZE)

r = {
    "label": "R6_sinkT_FaF",
    "sink": True, "sink_size": SINK_SIZE,
    "e1_s": round(epoch_times[0], 1), "e2_s": round(epoch_times[1], 1),
    "e2_per_step_ms": round(epoch_times[1]*1000/SINK_SIZE, 0),
}

# Safety check
try:
    flag = ckpt.read_probe_flag_dev()
    r["final_flag"] = int(flag)
    expected = TOTAL_STEPS // CKPT_INTERVAL
    r["safety"] = "PASSED" if flag >= expected else "FAILED"
except Exception as e:
    r["safety"] = f"error: {e}"

print(f"RESULT_R6: e1={r['e1_s']}s e2={r['e2_s']}s e2_ps={r['e2_per_step_ms']}ms safety={r['safety']}", flush=True)

# Save
with open(REPO + "/experiments/output/faf_r6_result.json", "w") as f:
    json.dump(r, f, indent=2)

ckpt.cleanup()
print("DONE", flush=True)
