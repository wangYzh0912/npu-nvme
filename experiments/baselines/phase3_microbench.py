#!/usr/bin/env python3
"""
Phase 3: Micro-benchmark — what is the real bottleneck in block delta?
========================================================================
Tests 4 configurations to isolate the bottleneck:
  A: 12L delta, block_size=512K  (current — ReduceSum on 512K)
  B: 12L delta, block_size=64K   (smaller ReduceSum, more blocks)
  C: 12L delta, block_size=512K, NO ReduceSum (only Sub+Mul, no reduction)
  D: Baseline (no I3)

This reveals whether ReduceSum or total element count is the bottleneck.
"""
import os, sys, time, math, re
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)
DEVICE_ID = 1; SEQ_LEN = 1024

def test_config(label, block_size, do_reduce, num_layers=12):
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

    all_layers = sorted(layer_map.keys())
    selected = all_layers[:num_layers]

    # Build param groups + block counts
    param_groups = [layer_map[l] for l in selected]
    fp16_needed = [[p.dtype != ms.float16 for p in g] for g in param_groups]
    n_blocks_per_group = [
        math.ceil(sum(int(p.size) for p in g) / block_size) for g in param_groups
    ]

    total_blocks = sum(n_blocks_per_group)
    total_elems = sum(sum(int(p.size) for p in g) for g in param_groups)
    print(f"  [{label}] {num_layers}L, block={block_size}, {total_blocks} blocks, "
          f"reduce={'Y' if do_reduce else 'N'}, {total_elems/1e6:.1f}M elems", flush=True)

    class TestCell(nn.Cell):
        def __init__(self):
            super().__init__(auto_prefix=False)
            self.net = model; self.net.set_grad()
            self.opt = opt
            self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
            self.pg = param_groups; self.fn = fp16_needed
            self.nbp = n_blocks_per_group; self.bs = block_size
            self.do_reduce = do_reduce

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
                    blk = full[b*self.bs:(b+1)*self.bs]
                    zero = ops.ZerosLike()(blk)
                    delta = ops.Sub()(blk, zero)
                    dsq = ops.Mul()(delta, delta)
                    if self.do_reduce:
                        n = ops.ReduceSum()(dsq)
                        acc = ops.Add()(acc, ops.Cast()(n, ms.float32))
                    else:
                        # Just accumulate element-wise sum (much cheaper)
                        s = ops.ReduceSum()(blk)  # one cheap sum
                        acc = ops.Add()(acc, ops.Cast()(s, ms.float32))
            loss = ops.Depend()(loss, acc)
            return ops.Depend()(loss, self.opt(grads))

    cell = TestCell()
    ms_model = ms.Model(cell)
    build_s = time.perf_counter() - time.perf_counter()  # ~0

    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(8)

    epoch_times = []
    class TCB(ms.Callback):
        def on_train_epoch_begin(self, rc): self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc): epoch_times.append((time.perf_counter() - self.t0) * 1000)

    t0 = time.perf_counter()
    try:
        ms_model.train(epoch=2, train_dataset=ds, callbacks=[TCB()], dataset_sink_mode=True, sink_size=4)
        compile_ep = epoch_times[0] if epoch_times else 0
        warm_ep = epoch_times[1] if len(epoch_times) > 1 else 0
        avg = warm_ep / 4 if warm_ep else 0
        dt = time.perf_counter() - t0
        ok = True
    except Exception as e:
        compile_ep = warm_ep = avg = 0
        dt = time.perf_counter() - t0
        ok = False
        print(f"    FAIL: {str(e)[:150]}")

    return {"label": label, "ok": ok, "compile_ms": compile_ep, "warm_epoch_ms": warm_ep, "avg_step_ms": avg,
            "total_s": dt, "block_size": block_size, "do_reduce": do_reduce,
            "total_blocks": total_blocks, "total_elems_M": total_elems/1e6}


print("=" * 70)
print("Phase 3: Delta Bottleneck Micro-benchmark")
print("=" * 70)

results = []

# Test A: 12L, 512K blocks, with ReduceSum
r = test_config("A: 512K + ReduceSum", 524288, True, 12)
results.append(r)

# Test B: 12L, 64K blocks, with ReduceSum
r = test_config("B: 64K + ReduceSum", 65536, True, 12)
results.append(r)

# Test C: 12L, 512K blocks, NO ReduceSum
r = test_config("C: 512K noReduce", 524288, False, 12)
results.append(r)

# Test D: 12L, 64K blocks, NO ReduceSum
r = test_config("D: 64K noReduce", 65536, False, 12)
results.append(r)

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"{'Test':<30s} {'Blocks':>7s} {'Elems(M)':>9s} {'Compile':>8s} {'WarmEp':>8s} {'AvgStep':>8s}")
print(f"{'-'*70}")
for r in results:
    if r["ok"]:
        print(f"{r['label']:<30s} {r['total_blocks']:>7d} {r['total_elems_M']:>8.1f} "
              f"{r['compile_ms']:>7.0f}ms {r['warm_epoch_ms']:>7.0f}ms {r['avg_step_ms']:>7.1f}ms")
    else:
        print(f"{r['label']:<30s} {'FAIL':>7s}")

print()
print("Interpretation:")
print("  A vs C: isolates ReduceSum cost for 512K blocks")
print("  A vs B: isolates block_size effect (same total elems, different ReduceSum size)")
print("  B vs D: isolates ReduceSum cost for 64K blocks")
print("[DONE]")
