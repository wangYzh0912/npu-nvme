#!/usr/bin/env python3
"""
Phase 3: Batched Block Delta — Loop vs Batched comparison
===========================================================

Tests two implementations with completely separate GRAPH_MODE builds:
  A: Loop-based (current): for b in range(N): op(block[b])
  B: Batched: Reshape(flat,(N,BS)) → single batched op
"""
import os, sys, time, math, re
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
DEVICE_ID = 1; SEQ_LEN = 1024; BLOCK_SIZE = 524288

def run_test(label, use_batched, num_layers=12):
    """Run one test in a fresh GRAPH_MODE context."""
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

    selected = sorted(layer_map.keys())[:num_layers]
    # Use ParameterTuple to avoid name conflicts in GE
    pgroups = [tuple(layer_map[l]) for l in selected]
    fn_groups = [tuple(p.dtype != ms.float16 for p in g) for g in pgroups]
    flat_sizes = [sum(int(p.size) for p in g) for g in pgroups]
    nblks = [math.ceil(s / BLOCK_SIZE) for s in flat_sizes]
    total = sum(nblks)

    class CellA(nn.Cell):
        def __init__(self):
            super().__init__(auto_prefix=False)
            self.net = model; self.net.set_grad()
            self.opt = opt
            self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
            self.pg = pgroups; self.fn = fn_groups; self.nb = nblks; self.fs = flat_sizes
        def construct(self, *inp):
            loss, grads = self.gf(*inp)
            acc = Tensor([0.0], dtype=ms.float32)
            for gi in range(len(self.pg)):
                group = self.pg[gi]; flags = self.fn[gi]
                parts = []
                for pi in range(len(group)):
                    p = group[pi]
                    pv = ops.Cast()(p, ms.float16) if flags[pi] else p
                    parts.append(ops.Reshape()(pv, (-1,)))
                fd = parts[0] if len(parts)==1 else ops.Concat()(tuple(parts))
                for b in range(self.nb[gi]):
                    s = b * BLOCK_SIZE; e = s + BLOCK_SIZE
                    blk = fd[s:e]
                    z = ops.ZerosLike()(blk)
                    d = ops.Sub()(blk, z)
                    dsq = ops.Mul()(d, d)
                    n = ops.ReduceSum()(dsq)
                    acc = ops.Add()(acc, ops.Cast()(n, ms.float32))
            loss = ops.Depend()(loss, acc)
            return ops.Depend()(loss, self.opt(grads))

    class CellB(nn.Cell):
        def __init__(self):
            super().__init__(auto_prefix=False)
            self.net = model; self.net.set_grad()
            self.opt = opt
            self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
            self.pg = pgroups; self.fn = fn_groups; self.nb = nblks; self.fs = flat_sizes
        def construct(self, *inp):
            loss, grads = self.gf(*inp)
            acc = Tensor([0.0], dtype=ms.float32)
            for gi in range(len(self.pg)):
                group = self.pg[gi]; flags = self.fn[gi]
                parts = []
                for pi in range(len(group)):
                    p = group[pi]
                    pv = ops.Cast()(p, ms.float16) if flags[pi] else p
                    parts.append(ops.Reshape()(pv, (-1,)))
                fd = parts[0] if len(parts)==1 else ops.Concat()(tuple(parts))
                nb = self.nb[gi]
                padded_len = nb * BLOCK_SIZE
                pad_amt = padded_len - self.fs[gi]
                if pad_amt > 0:
                    # Use ops.pad from the initial import (already available)
                    padded = ops.pad(fd, (0, pad_amt), mode='constant', value=0.0)
                else:
                    padded = fd
                # BATCHED: [nb, BLOCK_SIZE] → all blocks in single GE invocation
                blocks = ops.Reshape()(padded, (nb, BLOCK_SIZE))
                zeros = ops.ZerosLike()(blocks)
                deltas = ops.Sub()(blocks, zeros)
                norms = ops.ReduceSum()(ops.Mul()(deltas, deltas), 1)
                layer_sum = ops.ReduceSum()(ops.Cast()(norms, ms.float32))
                acc = ops.Add()(acc, layer_sum)
            loss = ops.Depend()(loss, acc)
            return ops.Depend()(loss, self.opt(grads))

    Cell = CellA if not use_batched else CellB
    cell = Cell()
    msm = ms.Model(cell)

    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(8)

    print(f"  [{label}] Building...", end=" ", flush=True)
    et = []
    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc): self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc): et.append((time.perf_counter() - self.t0) * 1000)

    t0 = time.perf_counter()
    try:
        msm.train(epoch=2, train_dataset=ds, callbacks=[CB()], dataset_sink_mode=True, sink_size=4)
        ce = et[0] if et else 0; we = et[1] if len(et) > 1 else 0; av = we/4 if we else 0
        dt = time.perf_counter() - t0
        print(f"compile={ce:.0f}ms  warm={we:.0f}ms  avg_step={av:.1f}ms")
        return {"ok": True, "compile_ms": ce, "warm_epoch_ms": we, "avg_step_ms": av, "total_s": dt}
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"FAIL ({dt:.1f}s): {str(e)[:200]}")
        return {"ok": False, "error": str(e)[:200]}


print("=" * 70)
print("Phase 3: Loop vs Batched Block Delta")
print("=" * 70)

# Test A
rA = run_test("A: Loop (14 blocks/L × 12L)", False, 12)
# Test B
rB = run_test("B: Batched", True, 12)

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
bl = 92.8
for label, r in [("A: Loop-based", rA), ("B: Batched", rB)]:
    if r["ok"]:
        d = r["avg_step_ms"] - bl
        print(f"  {label:20s}: compile={r['compile_ms']:.0f}ms  avg_step={r['avg_step_ms']:.1f}ms  Δ={d:+.1f}ms")
    else:
        print(f"  {label:20s}: FAILED")

if rA["ok"] and rB["ok"]:
    saved = rA["avg_step_ms"] - rB["avg_step_ms"]
    overheadA = rA["avg_step_ms"] - bl
    reduction = saved / max(overheadA, 0.1) * 100
    print(f"\n  Batched saves {saved:.1f}ms vs loop ({reduction:.0f}% of I3 overhead)")
    print(f"  Remaining overhead: {rB['avg_step_ms']-bl:.1f}ms (vs baseline)")

print("[DONE]")
