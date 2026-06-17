#!/usr/bin/env python3
"""
Phase 1a: GE Scheduling Verification — single training run, wrapped by msprof CLI.
This script does NOT use the Python Profiler API; it just trains and reports step times.
msprof collects PMU data externally.

Supports --inject N to control how many params get Vector ops.

Usage (standalone, for timing only):
  /root/miniconda3/envs/ms_2.5/bin/python phase1a_train.py --inject 0

Usage (with msprof for PMU):
  msprof --output=<dir> -- /root/miniconda3/envs/ms_2.5/bin/python phase1a_train.py --inject 0
"""
import os, sys, time, json, math, glob, csv, argparse
from collections import defaultdict

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops

ms.set_recursion_limit(10000)

DEVICE_ID = 1; SEQ_LEN = 1024
SINK_SIZE = 4; TOTAL_STEPS = 12; EPOCHS = 3
OUTPUT_DIR = os.path.join(REPO, "experiments", "output")


def run(inject_params, label):
    print(f"\n{'='*60}")
    print(f"  {label}: inject={inject_params}")
    print(f"{'='*60}", flush=True)

    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)

    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(TOTAL_STEPS)

    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
    all_params = list(model.trainable_params())
    n_total = len(all_params)

    covered = all_params[:inject_params] if inject_params > 0 else []
    n_inject = len(covered)
    num_groups = max(1, min(math.ceil(n_inject / 100), 10)) if n_inject > 0 else 0
    param_groups = []; fp16_needed = []
    if n_inject > 0:
        gs = max(1, math.ceil(n_inject / num_groups))
        for g in range(num_groups):
            s = g * gs; e = min(s + gs, n_inject)
            if s < n_inject:
                pg = covered[s:e]
                param_groups.append(pg)
                fp16_needed.append([
                    hasattr(p, 'dtype') and p.dtype != ms.float16 for p in pg
                ])

    total_elems = sum(int(np.prod(p.shape)) for p in covered) if inject_params else 0
    print(f"  [{label}] {n_total} total params, inject={n_inject}, "
          f"{total_elems/1e9:.2f}B elems, {num_groups} groups", flush=True)

    class ProfiledCell(nn.Cell):
        def __init__(self, network, optimizer, param_groups, fp16_needed, inject):
            super().__init__(auto_prefix=False)
            self.network = network; self.network.set_grad()
            self.optimizer = optimizer
            self.grad_fn = ops.value_and_grad(self.network, grad_position=None,
                                               weights=self.optimizer.parameters)
            self.depend = ops.Depend()
            self.param_groups = param_groups; self.fp16_needed = fp16_needed
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

    # Build
    t_build = time.perf_counter()
    cell = ProfiledCell(model, opt, param_groups, fp16_needed, inject_params > 0)
    ms_model = ms.Model(cell)
    build_s = time.perf_counter() - t_build
    print(f"  [{label}] Build={build_s:.1f}s", flush=True)

    # Timing
    epoch_times_ms = []

    class EpochCB(ms.Callback):
        def on_train_epoch_begin(self, rc):
            self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times_ms.append((time.perf_counter() - self.t0) * 1000)

    print(f"  [{label}] Training {TOTAL_STEPS} steps...", flush=True)
    compiled_ok = True; error_msg = None
    t_total = time.perf_counter()

    try:
        ms_model.train(epoch=EPOCHS, train_dataset=ds, callbacks=[EpochCB()],
                       dataset_sink_mode=True, sink_size=SINK_SIZE)
    except Exception as e:
        compiled_ok = False; error_msg = str(e)[:300]
        print(f"  [{label}] FAILED: {error_msg}", flush=True)

    total_s = time.perf_counter() - t_total

    compile_epoch = epoch_times_ms[0] if epoch_times_ms else 0
    warm_epochs = epoch_times_ms[1:] if len(epoch_times_ms) > 1 else []
    avg_step = sum(warm_epochs) / len(warm_epochs) / SINK_SIZE if warm_epochs else 0

    print(f"  [{label}] compile={compile_epoch:.0f}ms  warm_epochs={[f'{e:.0f}ms' for e in warm_epochs]}  "
          f"avg_step={avg_step:.0f}ms", flush=True)

    result = {
        "test": label, "total_params": n_total, "inject_params": inject_params,
        "inject_elems_B": round(total_elems/1e9, 3), "num_groups": num_groups,
        "sink_size": SINK_SIZE, "total_steps": TOTAL_STEPS, "epochs": EPOCHS,
        "compiled_ok": compiled_ok, "error": error_msg,
        "build_s": round(build_s, 1), "total_wall_s": round(total_s, 1),
        "compile_epoch_ms": round(compile_epoch, 0),
        "warm_epochs_ms": [round(et, 0) for et in warm_epochs],
        "avg_step_ms": round(avg_step, 1),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_json = os.path.join(OUTPUT_DIR, f"phase1a_{label.lower()}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [{label}] -> {os.path.basename(out_json)}", flush=True)

    return json.dumps(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject", type=int, default=0)
    args = parser.parse_args()
    label = "A1" if args.inject == 0 else (f"A2_{args.inject}" if args.inject else "A1")
    run(args.inject, label)
