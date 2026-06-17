#!/usr/bin/env python3
"""Phase 3 GE Scalability Test v2 — test if GE can handle per-layer delta injection.

Key test: build ONE Cell with delta detection for N layers. Test N=1,2,4,6,8,12.
This is the SAME Cell, not rebuilds. If it compiles for N=12 layers,
it proves there is NO intrinsic GE node limit constraining I3.
"""
import os, sys, time, math, re, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=12, help="Number of layers with delta detection")
args = parser.parse_args()

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
DEVICE_ID = 1; SEQ_LEN = 1024; BLOCK_SIZE = 524288

NUM_LAYERS = args.num_layers

print(f"\n{'='*60}")
print(f"GE Scalability Test: {NUM_LAYERS} layers with block-level delta")
print(f"{'='*60}")

ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
ms.common.set_seed(42)

from mindformers import AutoModel, AutoConfig
cfg = AutoConfig.from_pretrained("gpt2")
cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
model = AutoModel.from_config(cfg)

params = list(model.trainable_params())
opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

# Build per-layer param lists
layer_map = {}
for pi, p in enumerate(params):
    m = re.search(r'backbone\.blocks\.(\d+)\.', p.name)
    if m: layer_map.setdefault(int(m.group(1)), []).append(p)

all_layers = sorted(layer_map.keys())
selected = all_layers[:NUM_LAYERS]
total_blocks = sum(math.ceil(sum(int(p.size) for p in layer_map[l]) / BLOCK_SIZE) for l in selected)

param_groups = [layer_map[l] for l in selected]
fp16_needed = [[p.dtype != ms.float16 for p in g] for g in param_groups]
n_params = sum(len(g) for g in param_groups)
n_blocks_per_group = [math.ceil(sum(int(p.size) for p in g) / BLOCK_SIZE) for g in param_groups]

est_ops = n_params * 2 + len(param_groups) + total_blocks * 5 + 3
print(f"  Layers: {NUM_LAYERS}  Params: {n_params}  Blocks: {total_blocks}  Est Ops: ~{est_ops}")

# Pre-build cell with all delta groups
class TestCell(nn.Cell):
    def __init__(self):
        super().__init__(auto_prefix=False)
        self.net = model; self.net.set_grad()
        self.opt = opt
        self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
        self.pg = param_groups; self.fn = fp16_needed
        self.nbp = n_blocks_per_group

    def construct(self, *inp):
        loss, grads = self.gf(*inp)
        acc = Tensor([0.0], dtype=ms.float32)
        for gi, group in enumerate(self.pg):
            flat_parts = []
            flags = self.fn[gi]
            for pi, p in enumerate(group):
                pv = ops.Cast()(p, ms.float16) if flags[pi] else p
                flat_parts.append(ops.Reshape()(pv, (-1,)))
            full = flat_parts[0] if len(flat_parts)==1 else ops.Concat()(tuple(flat_parts))
            nb = self.nbp[gi]
            for b in range(nb):
                blk = full[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]
                zero = ops.ZerosLike()(blk)
                delta = ops.Sub()(blk, zero)
                dsq = ops.Mul()(delta, delta)
                n = ops.ReduceSum()(dsq)
                acc = ops.Add()(acc, ops.Cast()(n, ms.float32))
        loss = ops.Depend()(loss, acc)
        return ops.Depend()(loss, self.opt(grads))

t_build = time.perf_counter()
cell = TestCell()
ms_model = ms.Model(cell)
build_s = time.perf_counter() - t_build
print(f"  Build: {build_s:.1f}s")

ds = ms.dataset.MindDataset(
    REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
ds = ds.batch(1, drop_remainder=True).take(8)

print(f"  Training 8 steps (sink=4)...", flush=True)
t0 = time.perf_counter()

epoch_times = []
class TCB(ms.Callback):
    def on_train_epoch_begin(self, rc):
        self.t0 = time.perf_counter()
    def on_train_epoch_end(self, rc):
        epoch_times.append((time.perf_counter() - self.t0) * 1000)

try:
    ms_model.train(epoch=2, train_dataset=ds, callbacks=[TCB()], dataset_sink_mode=True, sink_size=4)
    compile_epoch = epoch_times[0] if epoch_times else 0
    warm_epoch = epoch_times[1] if len(epoch_times) > 1 else 0
    avg_step = warm_epoch / 4 if warm_epoch else 0
    dt = time.perf_counter() - t0
    print(f"  ✅ COMPILE OK!")
    print(f"  compile={compile_epoch:.0f}ms  warm_epoch={warm_epoch:.0f}ms  avg_step={avg_step:.0f}ms  total={dt:.1f}s")
    print(f"\n  VERDICT: {NUM_LAYERS} layers ({total_blocks} blocks, ~{est_ops} ops) compiles in GRAPH_MODE")
    print(f"  → GE can handle {NUM_LAYERS}-layer delta injection. No intrinsic node limit at {total_blocks} blocks.")
except Exception as e:
    err = str(e)[:350]
    dt = time.perf_counter() - t0
    print(f"  ❌ FAIL ({dt:.1f}s): {err}")
    print(f"\n  VERDICT: {NUM_LAYERS} layers ({total_blocks} blocks, ~{est_ops} ops) FAILS")
