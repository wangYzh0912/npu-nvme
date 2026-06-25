#!/usr/bin/env python3
"""
Phase 1b Fast Multi-Model Profiler (standalone, no msprof)
===========================================================

Runs multiple models standalone to get compile time + step time.
Output: JSON with all model sizes for Vector Idle Budget curve.

For PMU data, we reuse Phase 1a A1/A2_50 data for GPT-2 XL,
and add msprof profiling on a few smaller models.

Usage: sudo /home/user7/miniconda3/envs/ms_2.5/bin/python phase1b_fast.py
"""
import os, sys, time, json, gc

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)

OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
SEQUENCE_LENGTH = 1024

# Model presets to profile
PRESETS = [
    ("V4b_gpt2_18L",   dict(num_layers=18, hidden_size=1024, num_heads=16)),
    ("V5_gpt2_large",  dict(num_layers=36, hidden_size=1280, num_heads=20)),
    ("V5b_gpt2_24L",   dict(num_layers=24, hidden_size=1536, num_heads=24)),
]

# Reuse existing data:
# V2  = GPT-2 6L  d=768  (already done)
# V3  = GPT-2 XL 48L d=1600 (Phase 1a A1 data)
# V4  = GPT-2 4L  d=768  (already done)
# V1  = Dense micro-net (already done)


def build_and_profile(label, model_cfg, steps=16, sink=4, epochs=2, inject=0):
    """Build a model, train, and return timing. No msprof."""
    from mindformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

    print(f"\n{'='*60}")
    print(f"  {label}: L={model_cfg['num_layers']} d={model_cfg['hidden_size']} "
          f"heads={model_cfg['num_heads']}  inject={inject}")
    print(f"{'='*60}", flush=True)

    # msprof is slow on large GE graphs — avoid.
    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=1)
    ms.common.set_seed(42)

    cfg = GPT2Config(**model_cfg, seq_length=SEQUENCE_LENGTH,
                     max_position_embeddings=SEQUENCE_LENGTH)
    model = GPT2LMHeadModel(cfg)
    params = list(model.trainable_params())
    n_params = len(params)
    fp16_mb = sum(int(np.prod(p.shape)) for p in params) * 2 / 1024 / 1024

    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(steps)

    opt = nn.AdamWeightDecay(params, learning_rate=1e-5)

    # Build cell with optional injection
    n_inject = min(inject, n_params) if inject else 0
    covered = params[:n_inject] if n_inject else []
    import math
    num_groups = max(1, min(math.ceil(n_inject / 100), 10)) if n_inject > 0 else 0
    param_groups, fp16_needed = [], []
    if n_inject > 0:
        gs = max(1, math.ceil(n_inject / num_groups))
        for g in range(num_groups):
            s = g * gs; e = min(s + gs, n_inject)
            if s < n_inject:
                pg = covered[s:e]
                param_groups.append(pg)
                fp16_needed.append([
                    (hasattr(p, 'dtype') and p.dtype != ms.float16) for p in pg
                ])

    class Cell(nn.Cell):
        def __init__(self):
            super().__init__(auto_prefix=False)
            self.net = model; self.net.set_grad()
            self.opt = opt
            self.gf = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
            self.dep = ops.Depend()
            self.pg = param_groups; self.fn = fp16_needed; self.inj = n_inject > 0
        def construct(self, *inputs):
            loss, grads = self.gf(*inputs)
            if self.inj:
                acc = Tensor([0.0], dtype=ms.float16)
                for gi, group in enumerate(self.pg):
                    flags = self.fn[gi]
                    flat_parts = []
                    for pi, p in enumerate(group):
                        pv = ops.Cast()(p, ms.float16) if flags[pi] else p
                        flat_parts.append(ops.Reshape()(pv, (-1,)))
                    flat = flat_parts[0] if len(flat_parts) == 1 else ops.Concat()(tuple(flat_parts))
                    delta = ops.Sub()(flat, ops.ZerosLike()(flat))
                    red   = ops.ReduceSum()(delta)
                    v32   = ops.Cast()(red, ms.float32)
                    v16   = ops.Cast()(v32, ms.float16)
                    acc   = ops.Add()(acc, v16)
                loss = self.dep(loss, acc)
            opt_res = self.opt(grads)
            return self.dep(loss, opt_res)

    cell = Cell()
    t_build = time.perf_counter()
    ms_model = ms.Model(cell)
    build_s = time.perf_counter() - t_build

    epoch_times_ms = []
    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc):
            self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times_ms.append((time.perf_counter() - self.t0) * 1000)

    print(f"  [{label}] Building... Build={build_s:.0f}s", flush=True)
    compiled_ok = True; error_msg = None
    t_total = time.perf_counter()
    try:
        ms_model.train(epoch=epochs, train_dataset=ds, callbacks=[CB()],
                       dataset_sink_mode=True, sink_size=sink)
    except Exception as e:
        compiled_ok = False; error_msg = str(e)[:300]
        print(f"  [{label}] FAILED: {error_msg}", flush=True)
    total_s = time.perf_counter() - t_total

    compile_epoch = epoch_times_ms[0] if epoch_times_ms else 0
    warm_epochs = epoch_times_ms[1:] if len(epoch_times_ms) > 1 else []
    avg_step = sum(warm_epochs) / len(warm_epochs) / sink if warm_epochs else 0

    result = {
        "label": label,
        "model": f"gpt2_L{model_cfg['num_layers']}_d{model_cfg['hidden_size']}",
        "n_params": n_params, "fp16_mb": round(fp16_mb, 1),
        "inject": inject, "build_s": round(build_s, 1),
        "compile_epoch_ms": round(compile_epoch, 0),
        "avg_step_ms": round(avg_step, 1),
        "warm_epochs_ms": [round(e, 0) for e in warm_epochs],
        "compiled_ok": compiled_ok,
    }

    out_path = os.path.join(OUTPUT_DIR, f"phase1b_{label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  [{label}] step={avg_step:.1f}ms compile={compile_epoch:.0f}ms", flush=True)

    # Cleanup
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    ms.reset_auto_parallel_context()
    gc.collect()
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    for label, cfg in PRESETS:
        r = build_and_profile(label, cfg, steps=16, sink=4, epochs=2, inject=0)
        all_results[label] = r

    # Summary
    print(f"\n{'='*70}")
    print("Phase 1b Fast Summary")
    print(f"{'Model':<20s} {'Params':>7s} {'FP16MB':>8s} {'Step(ms)':>9s} {'Compile(s)':>10s}")
    print("-"*60)
    for label, r in all_results.items():
        print(f"{label:<20s} {r['n_params']:>7d} {r['fp16_mb']:>8.0f} {r['avg_step_ms']:>9.1f} {r['compile_epoch_ms']/1000:>10.0f}")

    with open(os.path.join(OUTPUT_DIR, "phase1b_fast_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Results saved")

if __name__ == "__main__":
    main()
