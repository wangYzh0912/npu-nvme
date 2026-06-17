#!/usr/bin/env python3
"""
Phase 3: Batched Block Delta Test
===================================

Core insight: the bottleneck is NOT ReduceSum or compute — it's GE subgraph launch
overhead. Each Python `for b in range(N): ...` creates N independent subgraphs,
each costing ~0.14ms to launch → 168 blocks × 0.14ms = 23ms overhead.

Solution: Replace per-block loop with a SINGLE batched GE operation using Reshape:
  blocks = Reshape(flat, (N, block_size))   # [168, 512K]
  polds = Reshape(pold_flat, (N, block_size))
  deltas = Sub(blocks, polds)               # 1 op
  norms = ReduceSum(Mul(deltas, deltas), 1) # 1 op → [168] norms

Expected overhead: < 2ms (vs 23ms for per-block loop)
"""
import os, sys, time, math, re
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
DEVICE_ID = 1; SEQ_LEN = 1024; BLOCK_SIZE = 524288

print("=" * 70)
print("Phase 3: Batched Block Delta Test")
print("=" * 70)

# ── Config A: Loop-based (current approach) ──
ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
ms.common.set_seed(42)

from mindformers import AutoModel, AutoConfig
cfg = AutoConfig.from_pretrained("gpt2")
cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
model = AutoModel.from_config(cfg)
opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

params = list(model.trainable_params())
layer_map = {}
for pi, p in enumerate(params):
    m = re.search(r'backbone\.blocks\.(\d+)\.', p.name)
    if m: layer_map.setdefault(int(m.group(1)), []).append(p)

selected = sorted(layer_map.keys())[:12]
param_groups = [layer_map[l] for l in selected]
fp16_needed = [[p.dtype != ms.float16 for p in g] for g in param_groups]

# Pre-compute flat sizes for padding
layer_flat_sizes = [sum(int(p.size) for p in g) for g in param_groups]
layer_num_blocks = [math.ceil(s / BLOCK_SIZE) for s in layer_flat_sizes]
total_blocks = sum(layer_num_blocks)

print(f"\n  12 layers, {total_blocks} blocks total (per-layer: {layer_num_blocks})")
print(f"  Test A: Loop-based (current)")
print(f"  Test B: Batched (Reshape-based)")

# ── Test A: Loop-based ──
modelA = AutoModel.from_config(cfg)
optA = nn.AdamWeightDecay(modelA.trainable_params(), learning_rate=1e-5)

class LoopCell(nn.Cell):
    def __init__(self):
        super().__init__(auto_prefix=False)
        self.net = modelA; self.net.set_grad()
        self.opt = optA
        self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
        self.pg = param_groups; self.fn = fp16_needed
        self.nbp = layer_num_blocks

    def construct(self, *inp):
        loss, grads = self.gf(*inp)
        acc = Tensor([0.0], dtype=ms.float32)
        for gi, group in enumerate(self.pg):
            flat_parts = []
            for pi, p in enumerate(group):
                pv = ops.Cast()(p, ms.float16) if self.fn[gi][pi] else p
                flat_parts.append(ops.Reshape()(pv, (-1,)))
            full = flat_parts[0] if len(flat_parts)==1 else ops.Concat()(tuple(flat_parts))
            for b in range(self.nbp[gi]):
                blk = full[b*BLOCK_SIZE:(b+1)*BLOCK_SIZE]
                zero = ops.ZerosLike()(blk)
                delta = ops.Sub()(blk, zero)
                dsq = ops.Mul()(delta, delta)
                n = ops.ReduceSum()(dsq)
                acc = ops.Add()(acc, ops.Cast()(n, ms.float32))
        loss = ops.Depend()(loss, acc)
        return ops.Depend()(loss, self.opt(grads))

cellA = LoopCell()
msA = ms.Model(cellA)

dsA = ms.dataset.MindDataset(REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
dsA = dsA.batch(1, drop_remainder=True).take(8)

print("\n  [A] Loop-based...", end=" ", flush=True)
t0 = time.perf_counter()
try:
    epoch_times = []
    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc): self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc): epoch_times.append((time.perf_counter() - self.t0) * 1000)
    msA.train(epoch=2, train_dataset=dsA, callbacks=[CB()], dataset_sink_mode=True, sink_size=4)
    ce = epoch_times[0]; we = epoch_times[1] if len(epoch_times) > 1 else 0; avg = we/4
    dt = time.perf_counter() - t0
    print(f"OK: compile={ce:.0f}ms  warm={we:.0f}ms  avg_step={avg:.1f}ms")
    resultA = {"ok": True, "compile_ms": ce, "warm_epoch_ms": we, "avg_step_ms": avg}
except Exception as e:
    dt = time.perf_counter() - t0
    print(f"FAIL: {str(e)[:200]}")
    resultA = {"ok": False, "error": str(e)[:200]}

# ── Test B: Batched ──
ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
ms.common.set_seed(42)
modelB = AutoModel.from_config(cfg)
optB = nn.AdamWeightDecay(modelB.trainable_params(), learning_rate=1e-5)

class BatchCell(nn.Cell):
    def __init__(self):
        super().__init__(auto_prefix=False)
        self.net = modelB; self.net.set_grad()
        self.opt = optB
        self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
        self.pg = param_groups; self.fn = fp16_needed
        self.nbp = layer_num_blocks
        self.lfs = layer_flat_sizes

    def construct(self, *inp):
        loss, grads = self.gf(*inp)
        acc = Tensor([0.0], dtype=ms.float32)
        for gi, group in enumerate(self.pg):
            # Pad to exact multiple of BLOCK_SIZE
            padded_len = nb * BLOCK_SIZE
            pad_amt = padded_len - self.lfs[gi]
            # Use mindspore.ops.pad with explicit paddings tuple
            if pad_amt > 0:
                import mindspore.ops.functional as F
                padded = F.pad(full, ((0, pad_amt),), mode='constant', value=0.0)
            else:
                padded = full

            # BATCHED: Reshape to [nb, BLOCK_SIZE] → all blocks in ONE op
            blocks = ops.Reshape()(padded, (nb, BLOCK_SIZE))        # [nb, 512K]
            zeros = ops.ZerosLike()(blocks)                          # [nb, 512K]
            deltas = ops.Sub()(blocks, zeros)                        # batched
            dsqs = ops.Mul()(deltas, deltas)                        # batched
            norms = ops.ReduceSum()(dsqs, 1)                        # → [nb] norms
            # Sum all norms for this layer
            layer_acc = ops.ReduceSum()(ops.Cast()(norms, ms.float32))
            acc = ops.Add()(acc, layer_acc)
        loss = ops.Depend()(loss, acc)
        return ops.Depend()(loss, self.opt(grads))

cellB = BatchCell()
msB = ms.Model(cellB)

dsB = ms.dataset.MindDataset(REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
dsB = dsB.batch(1, drop_remainder=True).take(8)

print("  [B] Batched...", end=" ", flush=True)
t0 = time.perf_counter()
try:
    epoch_times = []
    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc): self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc): epoch_times.append((time.perf_counter() - self.t0) * 1000)
    msB.train(epoch=2, train_dataset=dsB, callbacks=[CB()], dataset_sink_mode=True, sink_size=4)
    ce = epoch_times[0]; we = epoch_times[1] if len(epoch_times) > 1 else 0; avg = we/4
    dt = time.perf_counter() - t0
    print(f"OK: compile={ce:.0f}ms  warm={we:.0f}ms  avg_step={avg:.1f}ms")
    resultB = {"ok": True, "compile_ms": ce, "warm_epoch_ms": we, "avg_step_ms": avg}
except Exception as e:
    dt = time.perf_counter() - t0
    print(f"FAIL: {str(e)[:200]}")
    resultB = {"ok": False, "error": str(e)[:200]}

# ── Comparison ──
print(f"\n{'='*70}")
print(f"RESULTS: Loop vs Batched (12 layers, {total_blocks} blocks)")
print(f"{'='*70}")
bl = 92.8  # baseline from earlier measurements
for label, r in [("Loop-based", resultA), ("Batched", resultB)]:
    if r["ok"]:
        delta = r["avg_step_ms"] - bl
        print(f"  {label:15s}: compile={r['compile_ms']:.0f}ms  warm_epoch={r['warm_epoch_ms']:.0f}ms  "
              f"avg_step={r['avg_step_ms']:.1f}ms  Δ={delta:+.1f}ms")
    else:
        print(f"  {label:15s}: FAILED — {r.get('error', 'N/A')[:120]}")

if resultA["ok"] and resultB["ok"]:
    reduction = (resultA["avg_step_ms"] - resultB["avg_step_ms"]) / (resultA["avg_step_ms"] - bl) * 100
    print(f"\n  Batched overhead reduction: {(resultA['avg_step_ms'] - resultB['avg_step_ms']):.1f}ms ({reduction:.0f}%)")

print("\n[DONE]")
