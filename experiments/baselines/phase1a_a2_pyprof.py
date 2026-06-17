#!/usr/bin/env python3
"""Phase 1a A2 PMU via Python Profiler API — minimal injection (10 params) test.
Purpose: See if MindSpore's own profiler can collect PMU data without msprof CLI overhead.
"""
import os, sys, time, json, math

REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics

ms.set_recursion_limit(10000)

DEVICE_ID = 1
SEQ_LEN = 1024
INJECT_PARAMS = 10  # Tiny injection to test profiling stability
LABEL = "A2_PYPROF"

PROF_DIR = os.path.join(REPO, "output/profiling_vec/A2_PYPROF")
os.makedirs(PROF_DIR, exist_ok=True)

ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
ms.common.set_seed(42)

# Environment for profiler
os.environ["MS_DEV_RUNTIME_CONF"] = "profiling_option:true"
os.environ["ASCEND_PROFILING_MODE"] = "1"

print(f"[{LABEL}] Initializing MindSpore Profiler...", flush=True)

profiler = ms.Profiler(
    level=ProfilerLevel.Level1,
    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
    aicore_metrics=AicoreMetrics.ArithmeticUtilization,
    data_sink_specified_steps=[0, 1],
    output_path=PROF_DIR,
)

from mindformers import AutoModel, AutoConfig

cfg = AutoConfig.from_pretrained("gpt2_xl")
cfg.seq_length = SEQ_LEN
cfg.max_position_embeddings = SEQ_LEN
model = AutoModel.from_config(cfg)

ds = ms.dataset.MindDataset(
    REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
    shuffle=True,
)
ds = ds.batch(1, drop_remainder=True).take(8)

opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
all_params = list(model.trainable_params())
n_total = len(all_params)
covered = all_params[:INJECT_PARAMS] if INJECT_PARAMS > 0 else []
n_inject = len(covered)

num_groups = max(1, min(math.ceil(n_inject / 100), 10)) if n_inject > 0 else 0
param_groups = []
fp16_needed = []
if n_inject > 0:
    gs = max(1, math.ceil(n_inject / max(num_groups, 1)))
    for g in range(num_groups):
        s = g * gs
        e = min(s + gs, n_inject)
        if s < n_inject:
            pg = covered[s:e]
            param_groups.append(pg)
            fp16_needed.append([hasattr(p, "dtype") and p.dtype != ms.float16 for p in pg])

total_elems = sum(int(np.prod(p.shape)) for p in covered) if INJECT_PARAMS else 0
print(
    f"[{LABEL}] Total={n_total}, Inject={n_inject}, {total_elems/1e9:.2f}B elems, {num_groups} groups",
    flush=True,
)


class ProfiledCell(nn.Cell):
    def __init__(self, network, optimizer, pg, fn, inj):
        super().__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.grad_fn = ops.value_and_grad(
            self.network, grad_position=None, weights=self.optimizer.parameters
        )
        self.depend = ops.Depend()
        self.pg = pg
        self.fn = fn
        self.inj = inj

    def construct(self, *inputs):
        loss, grads = self.grad_fn(*inputs)
        if self.inj:
            acc = Tensor([0.0], dtype=ms.float16)
            for gi, group in enumerate(self.pg):
                flags = self.fn[gi]
                flat_parts = []
                for pi, p in enumerate(group):
                    pv = ops.Cast()(p, ms.float16) if flags[pi] else p
                    flat_parts.append(ops.Reshape()(pv, (-1,)))
                flat = (
                    flat_parts[0]
                    if len(flat_parts) == 1
                    else ops.Concat()(tuple(flat_parts))
                )
                delta = ops.Sub()(flat, ops.ZerosLike()(flat))
                red = ops.ReduceSum()(delta)
                c32 = ops.Cast()(red, ms.float32)
                c16 = ops.Cast()(c32, ms.float16)
                acc = ops.Add()(acc, c16)
            loss = self.depend(loss, acc)
        opt_res = self.optimizer(grads)
        return self.depend(loss, opt_res)


t_build = time.perf_counter()
cell = ProfiledCell(model, opt, param_groups, fp16_needed, INJECT_PARAMS > 0)
ms_model = ms.Model(cell)
build_s = time.perf_counter() - t_build
print(f"[{LABEL}] Build={build_s:.1f}s", flush=True)

epoch_times_ms = []


class CB(ms.Callback):
    def on_train_epoch_begin(self, rc):
        self.t0 = time.perf_counter()

    def on_train_epoch_end(self, rc):
        epoch_times_ms.append((time.perf_counter() - self.t0) * 1000)


print(f"[{LABEL}] Starting training (8 steps, sink=4, 2 epochs)...", flush=True)
t_total = time.perf_counter()
compiled_ok = True
error_msg = None

profiler.start()

try:
    ms_model.train(
        epoch=2,
        train_dataset=ds,
        callbacks=[CB()],
        dataset_sink_mode=True,
        sink_size=4,
    )
except Exception as e:
    compiled_ok = False
    error_msg = str(e)[:500]
    print(f"[{LABEL}] FAILED: {error_msg}", flush=True)

profiler.stop()
profiler.analyze_and_save()

total_s = time.perf_counter() - t_total
compile_epoch = epoch_times_ms[0] if epoch_times_ms else 0
warm_epochs = epoch_times_ms[1:] if len(epoch_times_ms) > 1 else []
avg_step = sum(warm_epochs) / len(warm_epochs) / 4 if warm_epochs else 0

print(
    f"[{LABEL}] compile={compile_epoch:.0f}ms  warm={[round(e,0) for e in warm_epochs]}  avg_step={avg_step:.0f}ms",
    flush=True,
)

result = {
    "test": LABEL,
    "total_params": n_total,
    "inject_params": INJECT_PARAMS,
    "inject_elems_B": round(total_elems / 1e9, 3),
    "num_groups": num_groups,
    "compiled_ok": compiled_ok,
    "error": error_msg,
    "build_s": round(build_s, 1),
    "total_wall_s": round(total_s, 1),
    "compile_epoch_ms": round(compile_epoch, 0),
    "warm_epochs_ms": [round(et, 0) for et in warm_epochs],
    "avg_step_ms": round(avg_step, 1),
    "prof_dir": PROF_DIR,
}

os.makedirs(REPO + "/experiments/output", exist_ok=True)
out_path = REPO + f"/experiments/output/phase1a_{LABEL.lower()}.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"[{LABEL}] -> {os.path.basename(out_path)}", flush=True)
