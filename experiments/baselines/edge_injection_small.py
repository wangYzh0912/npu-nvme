#!/usr/bin/env python3
"""
I3 Edge Injection — SMALL-GROUP feasibility test.
Tests whether per-param Vector ops (Sub+ReduceSum+Cast) work within GE's
1000 call-depth limit. The real I3 implementation will use an Ascend C kernel
receiving packed metadata, so this test just needs to demonstrate that Vector
ops on parameters inside construct() DON'T slow down training.

Design:
  - Test groups of 50/100/200/400 params
  - Each param: Cast→FP16 → Reshape → Concat → Sub→ReduceSum→Cast×2
  - Find the largest group size that compiles successfully

Levels:
  S0: Pure MS baseline (0 extra ops)
  S1: 50 params in 1 group  (call depth ~300)
  S2: 100 params in 1 group (call depth ~600)
  S3: 200 params in 2 groups (call depth ~600 each)
  S4: 400 params in 4 groups (call depth ~600 each)

Config: GPT-2 XL, sink=TRUE, sink_size=10, 20 steps

Output: experiments/baselines/edge_injection_small.json

Usage:
  sudo /home/user7/npu-nvme/experiments/baselines/_run_small.sh
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


def run_small_group_test(num_params_to_cover, num_groups, label):
    """
    Cover the first num_params_to_cover model parameters, split into num_groups.
    Each group: Cast→FP16 → Flatten → Concat → Sub→ReduceSum→Cast×2
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
    all_params = list(model.trainable_params())[:num_params_to_cover]
    n_params = len(all_params)
    total_elems = sum(int(np.prod(p.shape)) for p in all_params)

    # Split covered params into groups
    group_size = math.ceil(n_params / max(num_groups, 1))
    param_groups = []
    for g in range(max(num_groups, 1)):
        start = g * group_size
        end = min(start + group_size, n_params)
        if start < n_params:
            param_groups.append(all_params[start:end])

    elems_per_group = [sum(int(np.prod(p.shape)) for p in pg) for pg in param_groups]
    group_info = ", ".join([f"G{g}:{len(pg)}p/{eg/1e6:.1f}M"
                           for g, (pg, eg) in enumerate(zip(param_groups, elems_per_group))])

    print(f"  [{label}] Covered={n_params} params, Total={total_elems/1e9:.2f}B elems", flush=True)
    print(f"  [{label}] Groups={len(param_groups)}: {group_info}", flush=True)

    # Pre-compute per-param dtype info for construct()
    param_fp16_needed = []
    for pg in param_groups:
        group_flags = []
        for p in pg:
            if hasattr(p, 'dtype'):
                group_flags.append(p.dtype != ms.float16)
            else:
                group_flags.append(False)
        param_fp16_needed.append(group_flags)

    class SmallGroupI3Cell(nn.Cell):
        def __init__(self, network, optimizer, param_groups, num_groups, fp16_needed):
            super().__init__(auto_prefix=False)
            self.network = network
            self.network.set_grad()
            self.optimizer = optimizer
            self.grad_fn = ops.value_and_grad(self.network, grad_position=None,
                                               weights=self.optimizer.parameters)
            self.depend = ops.Depend()
            self.param_groups = param_groups
            self.num_groups = num_groups
            self.fp16_needed = fp16_needed

        def construct(self, *inputs):
            loss, grads = self.grad_fn(*inputs)

            if self.num_groups > 0:
                acc = Tensor([0.0], dtype=ms.float16)
                for gi, group in enumerate(self.param_groups):
                    group_flags = self.fp16_needed[gi]

                    # Step 1: Cast to FP16 if needed, then flatten
                    flat_parts = []
                    for pi, p in enumerate(group):
                        if group_flags[pi]:
                            p_fp16 = ops.Cast()(p, ms.float16)  # Vector op
                        else:
                            p_fp16 = p
                        flat_parts.append(ops.Reshape()(p_fp16, (-1,)))

                    # Step 2: Concat
                    if len(flat_parts) == 1:
                        flat = flat_parts[0]
                    else:
                        flat = ops.Concat()(tuple(flat_parts))

                    # Step 3: Vector ops (I3 simulation)
                    zero = ops.ZerosLike()(flat)
                    delta = ops.Sub()(flat, zero)              # Vector: delta detect
                    reduced = ops.ReduceSum()(delta)            # Vector: norm
                    cast32 = ops.Cast()(reduced, ms.float32)    # Vector: FP16→FP32
                    cast16 = ops.Cast()(cast32, ms.float16)     # Vector: FP32→FP16
                    acc = ops.Add()(acc, cast16)

                loss = self.depend(loss, acc)

            opt_res = self.optimizer(grads)
            loss = self.depend(loss, opt_res)
            return loss

    t_build = time.perf_counter()
    cell = SmallGroupI3Cell(model, opt, param_groups, num_groups, param_fp16_needed)
    ms_model = ms.Model(cell)
    build_s = time.perf_counter() - t_build

    epoch_times = []

    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc):
            self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times.append((time.perf_counter() - self.t0) * 1000)

    compiled_ok = True
    error_msg = None
    print(f"  [{label}] Build={build_s:.1f}s. Starting training...", flush=True)
    t0 = time.perf_counter()

    try:
        ms_model.train(epoch=2, train_dataset=ds, callbacks=[CB()],
                       dataset_sink_mode=True, sink_size=SINK_SIZE)
    except RuntimeError as e:
        compiled_ok = False
        error_msg = str(e)[:200]
        print(f"  [{label}] GE COMPILE FAILED: {error_msg}", flush=True)

    wall = time.perf_counter() - t0

    if compiled_ok:
        compile_ms = epoch_times[0] if epoch_times else 0
        warm_ms = epoch_times[1] if len(epoch_times) > 1 else 0
    else:
        compile_ms = 0
        warm_ms = 0

    if compiled_ok:
        print(f"  [{label}] DONE. compile={compile_ms/1000:.1f}s, warm={warm_ms:.0f}ms, "
              f"step={warm_ms/SINK_SIZE:.0f}ms", flush=True)

    return {
        "label": label,
        "num_params": n_params,
        "num_groups": num_groups,
        "params_per_group": [len(pg) for pg in param_groups],
        "elems_per_group_M": [round(eg/1e6, 1) for eg in elems_per_group],
        "group_info": group_info,
        "compiled_ok": compiled_ok,
        "error": error_msg,
        "build_s": round(build_s, 1),
        "compile_epoch_ms": round(compile_ms, 0) if compiled_ok else None,
        "warm_epoch_ms": round(warm_ms, 0) if compiled_ok else None,
        "warm_step_ms": round(warm_ms / SINK_SIZE, 0) if compiled_ok else None,
        "total_s": round(wall, 1),
    }


def main():
    os.makedirs(REPO + "/experiments/baselines", exist_ok=True)
    results = []

    # S0: baseline
    results.append(run_small_group_test(0, 0, "S0: Pure MS baseline"))

    # S1: 50 params, 1 group — should work (~150 call depth)
    results.append(run_small_group_test(50, 1, "S1: 50 params × 1 group"))

    # S2: 100 params, 1 group — moderate (~300 call depth)
    results.append(run_small_group_test(100, 1, "S2: 100 params × 1 group"))

    # S3: 200 params, 2 groups — larger but distributed
    results.append(run_small_group_test(200, 2, "S3: 200 params × 2 groups"))

    # S4: 400 params, 4 groups — stress test
    results.append(run_small_group_test(400, 4, "S4: 400 params × 4 groups"))

    # S5: ALL 772 params, 8 groups — the big one (if we get this far)
    results.append(run_small_group_test(772, 8, "S5: ALL 772 params × 8 groups"))

    # Print comparison
    print(f"\n\n{'='*80}")
    print(f"{'I3 Edge Injection — Small-Group Feasibility':^80}")
    print(f"{'GPT-2 XL, sink=TRUE, sink_size=10':^80}")
    print(f"{'='*80}")
    print(f"{'Test':<40} {'OK?':>5} {'Compile':>9} {'Warm step':>9} {'Δ Step':>8}")
    print("-" * 80)

    l0 = results[0]
    for r in results:
        if r['compiled_ok']:
            ds = r['warm_step_ms'] - l0['warm_step_ms']
            print(f"{r['label']:<40} {'YES':>5} {r['compile_epoch_ms']/1000:>7.1f}s {r['warm_step_ms']:>7.0f}ms {ds:>+7.0f}ms")
        else:
            print(f"{r['label']:<40} {'NO':>5} {'N/A':>9} {'N/A':>9} {'N/A':>8}")

    print("=" * 80)

    # Verdict
    successful = [r for r in results[1:] if r['compiled_ok']]
    print(f"\n  Compiled OK: {len(successful)}/{len(results)-1} tests")
    if successful:
        max_ok = max(r['num_params'] for r in successful)
        print(f"  Max params covered: {max_ok}")
        last = successful[-1]
        overhead = last['warm_step_ms'] - l0['warm_step_ms']
        overhead_pct = overhead / l0['warm_step_ms'] * 100 if l0['warm_step_ms'] else 0
        print(f"  Largest test ({last['num_params']} params) overhead: {overhead:+d}ms ({overhead_pct:+.1f}%)")

    failed = [r for r in results[1:] if not r['compiled_ok']]
    if failed:
        print(f"\n  ⚠ GE call-depth limit reached at ~{failed[0]['num_params']} params.")
        print(f"  Error: {failed[0]['error'][:120]}")
        print(f"  → I3 real implementation will use Ascend C kernel w/ packed metadata")
        print(f"    to avoid per-param loop unrolling. This test confirms the limit.")

    with open(REPO + "/experiments/baselines/edge_injection_small.json", "w") as f:
        json.dump({"test": "I3 Small-Group Edge Injection", "results": results}, f, indent=2)
    print(f"\nResults saved to experiments/baselines/edge_injection_small.json")
    print("[SmallGroupEdgeTest] DONE.", flush=True)


if __name__ == "__main__":
    main()
