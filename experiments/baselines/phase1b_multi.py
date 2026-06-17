#!/usr/bin/env python3
"""
Phase 1b: Multi-Model Family Profiling
=======================================

Profiles GPT-2 AND LLaMA models of different sizes with msprof PMU.
Each model is a COMPLETE full architecture (not truncated).

Models:
  GPT-2 Small  (12L/768d)    ~124M elems, 0.25GB FP16
  GPT-2 Medium (24L/1024d)   ~355M elems, 0.71GB FP16
  GPT-2 Large  (36L/1280d)   ~774M elems, 1.55GB FP16
  GPT-2 XL     (48L/1600d)   ~1.56B elems, 3.12GB FP16
  LLaMA-160M   (12L/768d)    ~134M elems, 0.27GB FP16
  LLaMA-410M   (24L/1024d)   ~374M elems, 0.75GB FP16
  LLaMA-1B     (16L/2048d)   ~953M elems, 1.91GB FP16
  LLaMA-2.7B   (32L/2560d)   ~2.7B elems, 5.40GB FP16

Usage (standalone timing):
  sudo /root/miniconda3/envs/ms_2.5/bin/python phase1b_multi.py \
    --label GPT2_Small_baseline --family gpt2 --size small --inject 0

Usage (with msprof for PMU):
  sudo msprof --output=<dir> --application="/root/miniconda3/envs/ms_2.5/bin/python phase1b_multi.py ..."
"""
import os, sys, time, json, math, gc, argparse

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)

OUTPUT_DIR = os.path.join(REPO, "experiments", "output")
DEVICE_ID = 1
SEQ_LEN = 1024

# ── Model definitions ──
MODELS = {
    # (family, size_label, num_layers, hidden_size, num_heads)
    "GPT2_Small":  ("gpt2",  "GPT-2 Small",   12, 768,  12),
    "GPT2_Medium": ("gpt2",  "GPT-2 Medium",   24, 1024, 16),
    "GPT2_Large":  ("gpt2",  "GPT-2 Large",    36, 1280, 20),
    "GPT2_XL":     ("gpt2",  "GPT-2 XL",       48, 1600, 25),
    "LLaMA_160M":  ("llama", "LLaMA-160M",     12, 768,  12),
    "LLaMA_410M":  ("llama", "LLaMA-410M",     24, 1024, 16),
    "LLaMA_1B":    ("llama", "LLaMA-1B",       16, 2048, 32),
    "LLaMA_2_7B":  ("llama", "LLaMA-2.7B",     32, 2560, 32),
}


def build_model(family, num_layers, hidden_size, num_heads):
    """Build the specified model with random weights."""
    if family == "gpt2":
        from mindformers.models.gpt2 import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config(num_layers=num_layers, hidden_size=hidden_size,
                         num_heads=num_heads, seq_length=SEQ_LEN,
                         max_position_embeddings=SEQ_LEN)
        return GPT2LMHeadModel(cfg)
    elif family == "llama":
        from mindformers.models.llama import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(num_layers=num_layers, hidden_size=hidden_size,
                          num_heads=num_heads, seq_length=SEQ_LEN,
                          max_position_embeddings=SEQ_LEN)
        return LlamaForCausalLM(cfg)
    raise ValueError(f"Unknown family: {family}")


def build_inject_cell(model, optimizer, inject_params_count):
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
                fp16_needed.append([(hasattr(p, 'dtype') and p.dtype != ms.float16) for p in pg])
    total_elems = sum(int(np.prod(p.shape)) for p in covered) if n_inject else 0

    class ProfiledCell(nn.Cell):
        def __init__(self):
            super().__init__(auto_prefix=False)
            self.net = model; self.net.set_grad()
            self.opt = optimizer
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

    return ProfiledCell(), n_total, n_inject, total_elems, num_groups


def run_experiment(label, family, num_layers, hidden_size, num_heads,
                   inject_params, total_steps=16, sink_size=4, epochs=2):
    """Run one profiling experiment."""
    print(f"\n{'='*70}")
    print(f"  Phase 1b Multi — {label}: {family} L={num_layers} d={hidden_size} h={num_heads}")
    print(f"{'='*70}", flush=True)

    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    model = build_model(family, num_layers, hidden_size, num_heads)
    params = list(model.trainable_params())
    n_total = len(params)
    elems = sum(int(p.size) for p in params)
    fp16_mb = elems * 2 / 1024 / 1024

    print(f"  [{label}] Model built — {n_total} params, {elems/1e6:.1f}M elems, {fp16_mb:.1f}MB FP16", flush=True)

    # Use GPT-2 dataset for all models (tokenizer compatible enough for graph profiling)
    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(total_steps)

    optimizer = nn.AdamWeightDecay(params, learning_rate=1e-5)
    cell, n_total, n_inject, total_elems, num_groups = build_inject_cell(
        model, optimizer, inject_params)

    if n_inject:
        print(f"  [{label}] Inject {n_inject} params, {total_elems/1e6:.2f}M elems, {num_groups} groups", flush=True)

    t_build = time.perf_counter()
    ms_model = ms.Model(cell)
    build_s = time.perf_counter() - t_build
    print(f"  [{label}] Build={build_s:.0f}s", flush=True)

    epoch_times_ms = []
    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc):
            self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times_ms.append((time.perf_counter() - self.t0) * 1000)

    print(f"  [{label}] Training {total_steps} steps (sink_size={sink_size})...", flush=True)
    compiled_ok = True; error_msg = None
    t_total = time.perf_counter()
    try:
        ms_model.train(epoch=epochs, train_dataset=ds, callbacks=[CB()],
                       dataset_sink_mode=True, sink_size=sink_size)
    except Exception as e:
        compiled_ok = False; error_msg = str(e)[:400]
        print(f"  [{label}] FAILED: {error_msg}", flush=True)
    total_s = time.perf_counter() - t_total

    compile_epoch = epoch_times_ms[0] if epoch_times_ms else 0
    warm_epochs = epoch_times_ms[1:] if len(epoch_times_ms) > 1 else []
    avg_step = sum(warm_epochs) / len(warm_epochs) / sink_size if warm_epochs else 0

    print(f"  [{label}] compile={compile_epoch:.0f}ms warm={[f'{e:.0f}ms' for e in warm_epochs]} "
          f"avg_step={avg_step:.1f}ms total={total_s:.0f}s", flush=True)

    result = {
        "label": label, "family": family,
        "num_layers": num_layers, "hidden_size": hidden_size, "num_heads": num_heads,
        "n_params": n_total, "elems_M": round(elems / 1e6, 1),
        "fp16_mb": round(fp16_mb, 1), "fp16_gb": round(fp16_mb / 1024, 3),
        "inject_params": inject_params,
        "inject_elems_M": round(total_elems / 1e6, 3),
        "num_groups": num_groups,
        "sink_size": sink_size, "total_steps": total_steps, "epochs": epochs,
        "compiled_ok": compiled_ok, "error": error_msg,
        "build_s": round(build_s, 1), "total_wall_s": round(total_s, 1),
        "compile_epoch_ms": round(compile_epoch, 0),
        "warm_epochs_ms": [round(et, 0) for et in warm_epochs],
        "avg_step_ms": round(avg_step, 1),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"phase1b_{label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [{label}] → {os.path.basename(out_path)}", flush=True)

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    ms.reset_auto_parallel_context()
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 1b Multi-Model Profiling")
    parser.add_argument("--label", required=True)
    parser.add_argument("--family", required=True, choices=["gpt2", "llama"])
    parser.add_argument("--size", required=True, help="Model size key in MODELS dict")
    parser.add_argument("--inject", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    if args.size in MODELS:
        family, desc, L, D, H = MODELS[args.size]
    else:
        print(f"Unknown size: {args.size}. Available: {list(MODELS.keys())}", file=sys.stderr)
        sys.exit(1)

    run_experiment(args.label, family, L, D, H, args.inject,
                   total_steps=args.steps, sink_size=args.sink, epochs=args.epochs)


if __name__ == "__main__":
    main()
