#!/usr/bin/env python3
"""
Phase 1b: Multi-Model Vector Idle Budget Profiling
===================================================

Profiles N model sizes with msprof PMU data to build the
Vector idle budget vs model scale curve.

Each model gets TWO runs:
  - Baseline (no injection): pure training PMU
  - Inject=50: 50 params with Vector ops to verify GE scheduling

Usage (standalone, for timing):
  /root/miniconda3/envs/ms_2.5/bin/python phase1b_profile.py \
    --label V5_baseline --preset gpt2_large --inject 0

Usage (with msprof for PMU):
  msprof --output=<dir> -- \
    /root/miniconda3/envs/ms_2.5/bin/python phase1b_profile.py \
    --label V5_baseline --preset gpt2_large --inject 0
"""
import os, sys, time, json, math, glob, csv, argparse
from collections import defaultdict

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)

OUTPUT_DIR = os.path.join(REPO, "experiments", "output")

# ── Known good model presets ──
PRESETS = {
    "gpt2_6L":     dict(num_layers=6,  hidden_size=768,  num_heads=12),
    "gpt2_12L":    dict(num_layers=12, hidden_size=768,  num_heads=12),
    "gpt2_large":  dict(num_layers=36, hidden_size=1280, num_heads=20),
    "gpt2_xl":     dict(num_layers=48, hidden_size=1600, num_heads=25),
    "gpt2_2_5b":   dict(num_layers=48, hidden_size=2048, num_heads=32),
    "gpt2_3_3b":   dict(num_layers=64, hidden_size=2048, num_heads=32),
    "gpt2_1_2b":   dict(num_layers=24, hidden_size=1536, num_heads=24),
    "gpt2_9L":     dict(num_layers=9,  hidden_size=768,  num_heads=12),
    "gpt2_18L":    dict(num_layers=18, hidden_size=1024, num_heads=16),
}


def estimate_params(num_layers, hidden_size, num_heads, vocab_size=50257):
    """Rough GPT-2 param count. Returns (total_elements, size_GB_FP16)."""
    d = hidden_size
    # Rough: 12 * d^2 per layer + embeddings
    layer_params = 12 * num_layers * d * d
    embed_params = (vocab_size + 1024) * d
    total = layer_params + embed_params * 2
    return total, total * 2 / 1e9


def build_gpt2(num_layers, hidden_size, num_heads, seq_len=1024, **_kw):
    """Build a custom GPT-2 model using GPT2Config.
    Additional keyword args (like _display) are silently ignored."""
    from mindformers.models.gpt2 import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        seq_length=seq_len,
        max_position_embeddings=seq_len,
    )
    model = GPT2LMHeadModel(cfg)
    print(f"  Gpt2Model: {num_layers}L d={hidden_size} heads={num_heads}", flush=True)
    return model


def build_model(name_or_preset, seq_len=1024):
    """Build model from preset dict or MindFormers name."""
    if isinstance(name_or_preset, dict):
        return build_gpt2(**name_or_preset, seq_len=seq_len)
    preset = PRESETS.get(name_or_preset)
    if preset:
        return build_gpt2(**preset, seq_len=seq_len)
    from mindformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(name_or_preset)
    cfg.seq_length = seq_len
    cfg.max_position_embeddings = seq_len
    model = AutoModel.from_config(cfg)
    return model


def build_inject_cell(model, optimizer, inject_params_count):
    """Build the training cell with optional Vector ops injection."""
    all_params = list(model.trainable_params())
    n_total = len(all_params)
    n_inject = min(inject_params_count, n_total) if inject_params_count else 0
    covered = all_params[:n_inject] if n_inject else []

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
    total_elems = sum(int(np.prod(p.shape)) for p in covered) if n_inject else 0

    class ProfiledCell(nn.Cell):
        def __init__(self, network, optimizer, param_groups, fp16_needed, inject):
            super().__init__(auto_prefix=False)
            self.network = network; self.network.set_grad()
            self.optimizer = optimizer
            self.grad_fn = ops.value_and_grad(self.network, grad_position=None,
                                               weights=self.optimizer.parameters)
            self.depend = ops.Depend()
            self.param_groups = param_groups
            self.fp16_needed = fp16_needed
            self.inject = inject

        def construct(self, *inputs):
            loss, grads = self.grad_fn(*inputs)
            if self.inject:
                acc = Tensor([0.0], dtype=ms.float16)
                for gi, group in enumerate(self.param_groups):
                    flags = self.fp16_needed[gi]
                    flat_parts = []
                    for pi, p in enumerate(group):
                        pv = ops.Cast()(p, ms.float16) if flags[pi] else p
                        flat_parts.append(ops.Reshape()(pv, (-1,)))
                    flat = flat_parts[0] if len(flat_parts) == 1 else ops.Concat()(tuple(flat_parts))
                    delta = ops.Sub()(flat, ops.ZerosLike()(flat))
                    red   = ops.ReduceSum()(delta)
                    c32   = ops.Cast()(red, ms.float32)
                    c16   = ops.Cast()(c32, ms.float16)
                    acc   = ops.Add()(acc, c16)
                loss = self.depend(loss, acc)
            opt_res = self.optimizer(grads)
            return self.depend(loss, opt_res)

    cell = ProfiledCell(model, optimizer, param_groups, fp16_needed, n_inject > 0)
    return cell, n_total, n_inject, total_elems, num_groups


def run_experiment(label, model_name, inject_params, device_id=1,
                   total_steps=20, sink_size=4, epochs=2):
    """Run one experiment: build, train, measure. Saves JSON result."""

    print(f"\n{'='*70}")
    print(f"  Phase 1b — {label}: model={model_name}  inject={inject_params}")
    print(f"{'='*70}", flush=True)

    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    ms.common.set_seed(42)

    # Build model
    seq_len = 1024
    if isinstance(model_name, dict):
        model = build_model(model_name, seq_len=seq_len)
    else:
        model = build_model(model_name, seq_len=seq_len)

    tp = list(model.trainable_params())
    fp16_mb = sum(int(np.prod(p.shape)) for p in tp) * 2 / 1024 / 1024
    print(f"  [{label}] Model built OK — {len(tp)} trainable params, {fp16_mb:.1f}MB FP16", flush=True)

    # Dataset
    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(total_steps)

    # Optimizer
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

    # Build injection cell
    cell, n_total, n_inject, total_elems, num_groups = build_inject_cell(
        model, optimizer, inject_params)

    print(f"  [{label}] {n_total} params, {fp16_mb:.1f}MB FP16, "
          f"inject={n_inject}, {total_elems/1e6:.2f}M elems, {num_groups} groups", flush=True)

    # Compile
    t_build = time.perf_counter()
    ms_model = ms.Model(cell)
    build_s = time.perf_counter() - t_build
    print(f"  [{label}] Build={build_s:.1f}s", flush=True)

    # Train with timing
    epoch_times_ms = []

    class EpochCB(ms.Callback):
        def on_train_epoch_begin(self, rc):
            self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times_ms.append((time.perf_counter() - self.t0) * 1000)

    print(f"  [{label}] Training {total_steps} steps (sink_size={sink_size})...", flush=True)
    compiled_ok = True; error_msg = None
    t_total = time.perf_counter()

    try:
        ms_model.train(epoch=epochs, train_dataset=ds, callbacks=[EpochCB()],
                       dataset_sink_mode=True, sink_size=sink_size)
    except Exception as e:
        compiled_ok = False
        error_msg = str(e)[:400]
        print(f"  [{label}] FAILED: {error_msg}", flush=True)

    total_s = time.perf_counter() - t_total

    # Timing stats
    compile_epoch = epoch_times_ms[0] if epoch_times_ms else 0
    warm_epochs = epoch_times_ms[1:] if len(epoch_times_ms) > 1 else []
    avg_step = sum(warm_epochs) / len(warm_epochs) / sink_size if warm_epochs else 0

    print(f"  [{label}] compile_epoch={compile_epoch:.0f}ms  "
          f"warm_epochs={[f'{e:.0f}ms' for e in warm_epochs]}  "
          f"avg_step={avg_step:.1f}ms  total={total_s:.1f}s", flush=True)

    result = {
        "test": label, "label": label,
        "model_name": str(model_name) if not isinstance(model_name, dict)
                      else model_name.get("_display", "custom_gpt2"),
        "model_config": model_name if isinstance(model_name, dict) else None,
        "num_params": n_total,
        "fp16_mb": round(fp16_mb, 1),
        "inject_params": inject_params,
        "inject_elems_M": round(total_elems / 1e6, 3),
        "num_groups": num_groups,
        "sink_size": sink_size,
        "total_steps": total_steps,
        "epochs": epochs,
        "compiled_ok": compiled_ok,
        "error": error_msg,
        "build_s": round(build_s, 1),
        "total_wall_s": round(total_s, 1),
        "compile_epoch_ms": round(compile_epoch, 0),
        "warm_epochs_ms": [round(et, 0) for et in warm_epochs],
        "avg_step_ms": round(avg_step, 1),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"phase1b_{label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [{label}] → {os.path.basename(out_path)}", flush=True)

    # Cleanup between runs
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    ms.reset_auto_parallel_context()
    import gc; gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 1b model profiling")
    parser.add_argument("--label", required=True, help="Experiment label (e.g. V5_baseline)")
    parser.add_argument("--preset", default=None, help="Model preset name")
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--inject", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    # Determine model config
    if args.num_layers is not None:
        model_cfg = {
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size or 768,
            "num_heads": args.num_heads or 12,
            "_display": f"gpt2_{args.num_layers}L_d{args.hidden_size}",
        }
    elif args.preset and args.preset in PRESETS:
        model_cfg = PRESETS[args.preset].copy()
        model_cfg["_display"] = args.preset
    elif args.preset:
        model_cfg = args.preset
    else:
        model_cfg = PRESETS["gpt2_xl"].copy()
        model_cfg["_display"] = "gpt2_xl"

    result = run_experiment(
        args.label, model_cfg, args.inject, args.device_id,
        total_steps=args.steps, sink_size=args.sink, epochs=args.epochs)

    fp16_mb = result.get("fp16_mb", 0)
    step_ms = result["avg_step_ms"]
    print(f"\n[PHASE1B_RESULT] {result['label']}: "
          f"params={result['num_params']} fp16={fp16_mb:.0f}MB "
          f"step={step_ms:.1f}ms ok={result['compiled_ok']}", flush=True)


if __name__ == "__main__":
    main()
