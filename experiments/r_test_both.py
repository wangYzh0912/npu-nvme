import os, sys, time
os.chdir("/home/user7/npu-nvme")
sys.path.insert(0, "python")
import numpy as np, mindspore as ms
from mindspore import nn, Tensor

# === Round 1: No SPDK (like H0) ===
print("=== ROUND 1: NO SPDK ===", flush=True)
ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=1)
ms.common.set_seed(42)
from mindformers import AutoModel, AutoConfig
cfg = AutoConfig.from_pretrained("gpt2_xl"); cfg.seq_length=1024; cfg.max_position_embeddings=1024
model1 = AutoModel.from_config(cfg)
ds1 = ms.dataset.MindDataset("dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
ds1 = ds1.batch(1, drop_remainder=True).take(5)
opt1 = nn.AdamWeightDecay(model1.trainable_params(), learning_rate=1e-5)
from direct_checkpoint import ProbeTrainOneStepCell
cell1 = ProbeTrainOneStepCell(model1, opt1, enable_probe=False)
times1 = []
class CB1(ms.Callback):
    def on_train_step_begin(self,rc): self.t0=time.perf_counter()
    def on_train_step_end(self,rc): times1.append(time.perf_counter()-self.t0)
ms_model1 = ms.Model(cell1)
ms_model1.train(epoch=1, train_dataset=ds1, callbacks=[CB1()], dataset_sink_mode=False)
arr1=np.array(times1[2:])
print("ROUND1_NO_SPDK: mean={:.0f}ms".format(arr1.mean()*1000), flush=True)

# === Round 2: WITH SPDK (same process!) ===
print("=== ROUND 2: WITH SPDK (same process) ===", flush=True)
ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=1)
ms.common.set_seed(42)
model2 = AutoModel.from_config(cfg)  # reuse cfg
ds2 = ms.dataset.MindDataset("dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
ds2 = ds2.batch(1, drop_remainder=True).take(5)
opt2 = nn.AdamWeightDecay(model2.trainable_params(), learning_rate=1e-5)
cell2 = ProbeTrainOneStepCell(model2, opt2, enable_probe=False)

# SPDK init AFTER first model.train()
from direct_checkpoint import DirectCheckpoint
ckpt = DirectCheckpoint(nvme_addr="0000:83:00.0", npu_device_id=1, pipeline_depth=8,
                        requested_chunk_size=4*1024*1024, enable_profiling=False, keep_last_n=3, slot_size_gb=10)

times2 = []
class CB2(ms.Callback):
    def on_train_step_begin(self,rc): self.t0=time.perf_counter()
    def on_train_step_end(self,rc): times2.append(time.perf_counter()-self.t0)
ms_model2 = ms.Model(cell2)
ms_model2.train(epoch=1, train_dataset=ds2, callbacks=[CB2()], dataset_sink_mode=False)
arr2=np.array(times2[2:])
print("ROUND2_WITH_SPDK_SAME_PROCESS: mean={:.0f}ms".format(arr2.mean()*1000), flush=True)
ckpt.cleanup()

print("\nOVERHEAD: +{:.0f}ms".format(arr2.mean()*1000 - arr1.mean()*1000), flush=True)
