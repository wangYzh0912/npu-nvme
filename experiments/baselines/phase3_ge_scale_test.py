#!/usr/bin/env python3
"""Phase 3: GE Layer Scalability Test — how many layers can we inject delta for?

Tests: 1, 2, 4, 6, 8, 12 layers with block-level delta detection.
Finds the real GE node limit for per-layer block delta ops.
"""
import os, sys, time, math, re
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
DEVICE_ID = 1; SEQ_LEN = 1024; BLOCK_SIZE = 524288

ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
ms.common.set_seed(42)

from mindformers import AutoModel, AutoConfig
cfg = AutoConfig.from_pretrained("gpt2")
cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
model = AutoModel.from_config(cfg)

params = list(model.trainable_params())
opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

layer_map = {}
for pi, p in enumerate(params):
    m = re.search(r'backbone\.blocks\.(\d+)\.', p.name)
    if m: layer_map.setdefault(int(m.group(1)), []).append(p)

layer_ids = sorted(layer_map.keys())
print(f"Layers: {len(layer_ids)}  Params: {len(params)}")
for l in layer_ids[:3]:
    total = sum(int(p.size) for p in layer_map[l])
    nb = math.ceil(total / BLOCK_SIZE)
    print(f"  L{l}: {len(layer_map[l])} params, {total/1e6:.1f}M elems, {nb} blocks")
print()

for num_layers in [1, 2, 4, 6, 8, 12]:
    selected = layer_ids[:num_layers]
    total_blocks = sum(math.ceil(sum(int(p.size) for p in layer_map[l]) / BLOCK_SIZE) for l in selected)

    param_groups = [layer_map[l] for l in selected]
    fp16_needed = [[p.dtype != ms.float16 for p in g] for g in param_groups]
    n_params = sum(len(g) for g in param_groups)

    # Rebuild model each time (clean GE graph)
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)
    model = AutoModel.from_config(cfg)
    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

    class TestCell(nn.Cell):
        def __init__(self):
            super().__init__(auto_prefix=False)
            self.net = model; self.net.set_grad()
            self.opt = opt
            self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
            self.pg = param_groups; self.fn = fp16_needed
            self.n_blocks_per_group = [math.ceil(sum(int(p.size) for p in g) / BLOCK_SIZE) for g in param_groups]

        def construct(self, *inp):
            loss, grads = self.gf(*inp)
            acc = Tensor([0.0], dtype=ms.float32)
            for gi, group in enumerate(self.pg):
                flat_parts = [ops.Reshape()(ops.Cast()(p, ms.float16) if self.fn[gi][pi] else p, (-1,))
                             for pi, p in enumerate(group)]
                full = flat_parts[0] if len(flat_parts)==1 else ops.Concat()(tuple(flat_parts))
                nb = self.n_blocks_per_group[gi]
                for b in range(nb):
                    blk = full[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]
                    zero = ops.ZerosLike()(blk)
                    delta = ops.Sub()(blk, zero)
                    dsq = ops.Mul()(delta, delta)
                    n = ops.ReduceSum()(dsq)
                    acc = ops.Add()(acc, ops.Cast()(n, ms.float32))
            loss = ops.Depend()(loss, acc)
            return ops.Depend()(loss, self.opt(grads))

    cell = TestCell()
    ms_model = ms.Model(cell)

    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(4)

    est_ops = n_params * 2 + len(param_groups) + total_blocks * 5 + 3
    print(f"  Testing {num_layers:2d} layers ({n_params} params, {total_blocks} blocks, ~{est_ops} GE ops)...",
          end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        ms_model.train(epoch=1, train_dataset=ds, callbacks=[], dataset_sink_mode=True, sink_size=2)
        dt = time.perf_counter() - t0
        print(f"OK ({dt:.1f}s)", flush=True)
    except Exception as e:
        dt = time.perf_counter() - t0
        err = str(e)[:250].replace("\n", " ")
        print(f"FAIL ({dt:.1f}s): {err}", flush=True)

    del cell, ms_model

print("\n[DONE] GE Layer Scalability Test")
