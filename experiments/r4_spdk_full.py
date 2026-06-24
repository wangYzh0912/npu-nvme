import os, sys, time
os.chdir("/home/user7/npu-nvme")
sys.path.insert(0, "python")
import numpy as np, mindspore as ms
from mindspore import nn, Tensor
ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=1)
ms.common.set_seed(42)
from mindformers import AutoModel, AutoConfig
cfg = AutoConfig.from_pretrained("gpt2_xl"); cfg.seq_length=1024; cfg.max_position_embeddings=1024
model = AutoModel.from_config(cfg)
ds = ms.dataset.MindDataset("dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
ds = ds.batch(1, drop_remainder=True).take(20)
opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
from direct_checkpoint import ProbeTrainOneStepCell, DirectCheckpoint
cell = ProbeTrainOneStepCell(model, opt, enable_probe=False)
ckpt = DirectCheckpoint(nvme_addr="0000:83:00.0", npu_device_id=1, pipeline_depth=8,
                        requested_chunk_size=4*1024*1024, enable_profiling=False, keep_last_n=3, slot_size_gb=10)
times = []
class CB(ms.Callback):
    def on_train_step_begin(self,rc): self.t0=time.perf_counter()
    def on_train_step_end(self,rc): times.append(time.perf_counter()-self.t0)
ms_model = ms.Model(cell)
ms_model.train(epoch=1, train_dataset=ds, callbacks=[CB()], dataset_sink_mode=False)
arr=np.array(times[2:])
print("RESULT: label=R4_spdk_full, mean={:.0f}, std={:.0f}, p99={:.0f}, n={}".format(
    arr.mean()*1000, arr.std()*1000, np.percentile(arr,99)*1000, len(arr)), flush=True)
ckpt.cleanup()
