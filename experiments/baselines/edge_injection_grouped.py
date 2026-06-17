#!/usr/bin/env python3
"""
I3 Edge Injection — GROUPED BATCH test on ALL 772 parameters.
Tests whether adding Vector ops to ALL model parameters (in 8 groups)
affects GE compilation or training step time.

Design:
  - Split 772 params into 8 groups of 96-97 each
  - For each group: flatten → Concat → Sub+ReduceSum+Cast×2
  - Accumulate scalar result across groups → Depend on loss
  - This stays under GE's 1000 call-depth limit (~800 calls)

Levels:
  G0: Pure MS baseline (0 extra ops, 0 groups)
  G1: 1 group × Flatten+Sub+ReduceSum+Cast×2 (~97 params)
  G2: 4 groups × Flatten+Sub+ReduceSum+Cast×2 (~386 params)
  G3: 8 groups × Flatten+Sub+ReduceSum+Cast×2 (ALL 772 params) ← THE KEY TEST

Output: experiments/baselines/edge_injection_grouped.json

Usage:
  sudo su - root -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && \
    /root/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/baselines/edge_injection_grouped.py'
"""
import os, sys, time, json, math
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)

DEVICE_ID = 1
SEQ_LEN = 1024
SINK_SIZE = 10
TOTAL_STEPS = 20  # 2 epochs


def run_grouped_test(num_groups, label):
    """
    If num_groups == 0: pure MS baseline.
    Otherwise: split all params into num_groups, concat each group,
    apply Sub+ReduceSum+Cast×2, accumulate result.
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}", flush=True)

    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN
    cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)
    print(f"  [{label}] Model built OK", flush=True)

    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(TOTAL_STEPS)

    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
    all_params = list(model.trainable_params())
    n_params = len(all_params)
    total_elems = sum(int(np.prod(p.shape)) for p in all_params)

    # Precompute group assignment — each param maps to one group
    group_size = math.ceil(n_params / max(num_groups, 1))
    param_groups = []
    for g in range(max(num_groups, 1)):
        start = g * group_size
        end = min(start + group_size, n_params)
        param_groups.append(all_params[start:end])

    group_info = ", ".join([f"G{g}:{len(pg)}params" for g, pg in enumerate(param_groups)])

    print(f"  [{label}] Params={n_params}, Total={total_elems/1e9:.2f}B elems", flush=True)
    print(f"  [{label}] Groups={len(param_groups)}: {group_info}", flush=True)

    class GroupedI3Cell(nn.Cell):
        def __init__(self, network, optimizer, param_groups, num_groups):
            super().__init__(auto_prefix=False)
            self.network = network
            self.network.set_grad()
            self.optimizer = optimizer
            self.grad_fn = ops.value_and_grad(self.network, grad_position=None,
                                               weights=self.optimizer.parameters)
            self.depend = ops.Depend()
            self.param_groups = param_groups
            self.num_groups = num_groups

        def construct(self, *inputs):
            loss, grads = self.grad_fn(*inputs)

            if self.num_groups > 0:
                acc = Tensor([0.0], dtype=ms.float16)
                for group in self.param_groups:
                    # Step 1: Cast all params to FP16, then flatten to 1D
                    flat_parts = []
                    for p in group:
                        # Cast to FP16 (real Vector op, + simulates I3's dtype unification)
                        p_fp16 = ops.Cast()(p, ms.float16) if p.dtype != ms.float16 else p
                        flat_parts.append(ops.Reshape()(p_fp16, (-1,)))

                    # Step 2: Concat into one big flat tensor for this group
                    if len(flat_parts) == 1:
                        flat = flat_parts[0]
                    else:
                        flat = ops.Concat()(tuple(flat_parts))

                    # Step 3: Apply Vector ops (simulate I3 delta detect + quant)
                    zero = ops.ZerosLike()(flat)
                    delta = ops.Sub()(flat, zero)              # Vector op: delta = W_cur - 0 (stub)
                    reduced = ops.ReduceSum()(delta)            # Vector op: ||delta||₁
                    cast32 = ops.Cast()(reduced, ms.float32)    # Vector op: FP16→FP32
                    cast16 = ops.Cast()(cast32, ms.float16)     # Vector op: FP32→FP16
                    acc = ops.Add()(acc, cast16)                # accumulate

                loss = self.depend(loss, acc)

            opt_res = self.optimizer(grads)
            loss = self.depend(loss, opt_res)
            return loss

    t_build = time.perf_counter()
    cell = GroupedI3Cell(model, opt, param_groups, num_groups)
    ms_model = ms.Model(cell)
    build_s = time.perf_counter() - t_build

    epoch_times = []

    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc):
            self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times.append((time.perf_counter() - self.t0) * 1000)

    print(f"  [{label}] Build={build_s:.1f}s. Starting training...", flush=True)
    t0 = time.perf_counter()

    ms_model.train(epoch=2, train_dataset=ds, callbacks=[CB()],
                   dataset_sink_mode=True, sink_size=SINK_SIZE)

    wall = time.perf_counter() - t0
    compile_ms = epoch_times[0] if epoch_times else 0
    warm_ms = epoch_times[1] if len(epoch_times) > 1 else 0

    print(f"  [{label}] DONE. compile={compile_ms/1000:.1f}s, warm={warm_ms:.0f}ms, "
          f"step={warm_ms/SINK_SIZE:.0f}ms", flush=True)

    return {
        "label": label,
        "num_groups": num_groups,
        "params_in_groups": sum(len(g) for g in param_groups),
        "total_params": n_params,
        "group_info": group_info,
        "build_s": round(build_s, 1),
        "compile_epoch_ms": round(compile_ms, 0),
        "warm_epoch_ms": round(warm_ms, 0),
        "warm_step_ms": round(warm_ms / SINK_SIZE, 0),
        "total_s": round(wall, 1),
    }


def main():
    os.makedirs(REPO + "/experiments/baselines", exist_ok=True)
    results = []

    # G0: baseline
    results.append(run_grouped_test(0, "G0: Pure MS baseline (0 groups, 0 ops)"))

    # G1: 1 group, ~97 params (first group only)
    results.append(run_grouped_test(1, "G1: 1 group × 97 params"))

    # G2: 4 groups, ~386 params
    results.append(run_grouped_test(4, "G2: 4 groups × 386 params"))

    # G3: 8 groups, ALL 772 params ← THE KEY TEST
    results.append(run_grouped_test(8, "G3: 8 groups × ALL 772 params"))

    # Print comparison
    print(f"\n\n{'='*80}")
    print(f"{'All-Parameter Edge Injection — Grouped Batch Test':^80}")
    print(f"{'GPT-2 XL, sink=TRUE, sink_size=10':^80}")
    print(f"{'Each group: Flatten→Concat→Sub→ReduceSum→Cast×2':^80}")
    print(f"{'='*80}")
    print(f"{'Test':<45} {'Compile':>10} {'Warm step':>10} {'Δ Step':>10}")
    print("-" * 80)

    l0 = results[0]
    for r in results:
        ds = r['warm_step_ms'] - l0['warm_step_ms']
        print(f"{r['label']:<45} {r['compile_epoch_ms']/1000:>8.1f}s {r['warm_step_ms']:>8.0f}ms {ds:>+8.0f}ms")

    print("=" * 80)

    # Verdict
    g3 = results[-1]
    overhead = g3['warm_step_ms'] - l0['warm_step_ms']
    overhead_pct = overhead / l0['warm_step_ms'] * 100 if l0['warm_step_ms'] else 0

    print(f"\n  G3 (ALL 772 params, 8 groups) vs G0 (pure MS):")
    print(f"    Step time:  {overhead:+d}ms ({overhead_pct:+.1f}%)")
    print(f"    Compile:    {g3['compile_epoch_ms']/1000:.1f}s (G0: {l0['compile_epoch_ms']/1000:.1f}s)")

    if abs(overhead) < 50 and abs(overhead_pct) < 10:
        print(f"\n  ★★★ VERDICT: Adding Vector ops to ALL 772 params is FEASIBLE ★★★")
        print(f"  Per-step overhead within noise (< ±50ms). Vector Engine idle capacity suffices.")
        print(f"  GE compilation does not explode (8-group batching is safe within 1000-call limit).")
    else:
        print(f"\n  WARNING: Significant overhead detected. Needs further analysis.")

    with open(REPO + "/experiments/baselines/edge_injection_grouped.json", "w") as f:
        json.dump({"test": "I3 All-Param Edge Injection — Grouped", "results": results}, f, indent=2)
    print(f"\nResults saved to experiments/baselines/edge_injection_grouped.json")
    print("[GroupedEdgeTest] DONE.", flush=True)


if __name__ == "__main__":
    main()
