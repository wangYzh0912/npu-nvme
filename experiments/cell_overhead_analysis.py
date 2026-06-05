#!/usr/bin/env python3
"""
Cell Overhead Isolation Experiment — Quantifies HyperMap vs AICPU vs Control-Dependency overhead.

Three Cell variants measured on a mini-network + GPT-2 XL:
  (a) Minimal TrainOneStep — bare grad_fn + optimizer (baseline)
  (b) ProbeTrainOneStepCell, enable_probe=False — HyperMap only, no AICPU call
  (c) ProbeTrainOneStepCell, enable_probe=True — full WaitProbe injection

Output: experiments/output/cell_overhead.json

Usage:
  sudo su - root -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && /root/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/cell_overhead_analysis.py'
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import time, json, argparse, warnings
import numpy as np
import mindspore as ms
from mindspore import nn, context, ops, Tensor

from direct_checkpoint import wait_op_info, bind_depend_op, ProbeTrainOneStepCell

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cell_overhead.json")

WARMUP = 10
MEASURE = 50


def make_mini_model(device_id):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    ms.common.set_seed(42)
    d1 = nn.Dense(128, 64); relu = nn.ReLU(); d2 = nn.Dense(64, 32)
    opt = nn.AdamWeightDecay(d1.trainable_params() + relu.trainable_params() + d2.trainable_params(), learning_rate=1e-3)
    x = Tensor(np.random.randn(4, 128).astype(np.float32))
    return d1, relu, d2, opt, x


def make_gpt2_xl(device_id):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=device_id)
    ms.common.set_seed(42)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = 1024; cfg.max_position_embeddings = 1024
    base_model = AutoModel.from_config(cfg)
    optimizer = nn.AdamWeightDecay(base_model.trainable_params(), learning_rate=1e-5)
    x = Tensor(np.random.randint(0, 50257, (1, 1024)).astype(np.int32))
    return base_model, optimizer, x


# --- Cell (a): Minimal TrainOneStep ---
class CellMinimal(nn.Cell):
    def __init__(self, net, opt):
        super().__init__(auto_prefix=False)
        self.net = net; self.net.set_grad()
        self.opt = opt
        self.grad_fn = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
    def construct(self, *inputs):
        if len(inputs) == 1: inputs = inputs[0]
        loss, grads = self.grad_fn(*inputs)
        opt_res = self.opt(grads)
        return ops.depend(loss, opt_res)


# --- Cell (b): Probe cell with enable_probe=False ---
def CellNoProbe(net, opt):
    return ProbeTrainOneStepCell(net, opt, None, 0, enable_probe=False, probe_mode="end")


# --- Cell (c): Full WaitProbe ---
def CellFullProbe(net, opt):
    return ProbeTrainOneStepCell(net, opt, None, 0, enable_probe=True, probe_mode="end")


def measure(cell, x, label):
    # warmup
    for _ in range(WARMUP):
        _ = cell(x)
    times = []
    for _ in range(MEASURE):
        t0 = time.perf_counter()
        _ = cell(x)
        times.append((time.perf_counter() - t0) * 1000)
    avg = float(np.mean(times[-40:]))
    p99 = float(np.percentile(times[-40:], 99))
    std = float(np.std(times[-40:]))
    print(f"  {label:<30s}: avg={avg:.4f}ms  p99={p99:.4f}ms  std={std:.4f}ms")
    gc.collect()
    return {"avg_ms": round(avg, 4), "p99_ms": round(p99, 4), "std_ms": round(std, 4), "n_meas": MEASURE}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--scale", type=str, default="mini,gpt2xl", help="comma-separated: mini,gpt2xl")
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}

    if "mini" in args.scale:
        print("=== Mini Network ===")
        d1, relu, d2, opt, x = make_mini_model(args.device_id)
        net = nn.SequentialCell([d1, relu, d2])
        # Wrap sum for value_and_grad compatibility
        class MiniNet(nn.Cell):
            def __init__(self):
                super().__init__()
                self.seq = net
            def construct(self, x):
                return self.seq(x).sum()
        mn = MiniNet()
        results["mini"] = {
            "model": "Dense(128->64->32)+ReLU",
            "num_params": 14,
            "a_minimal":        measure(CellMinimal(mn, opt), x, "a_minimal"),
        }
        # For (b) and (c), reconstruct with fresh instances to avoid compile conflicts
        d1b, relub, d2b, optb, xb = make_mini_model(args.device_id)
        nb = nn.SequentialCell([d1b, relub, d2b])
        class MiniNetB(nn.Cell):
            def __init__(self):
                super().__init__(); self.seq = nb
            def construct(self, x): return self.seq(x).sum()
        mnb = MiniNetB()
        results["mini"]["b_no_probe"] = measure(CellNoProbe(mnb, optb), xb, "b_no_probe")

        d1c, reluc, d2c, optc, xc = make_mini_model(args.device_id)
        nc = nn.SequentialCell([d1c, reluc, d2c])
        class MiniNetC(nn.Cell):
            def __init__(self):
                super().__init__(); self.seq = nc
            def construct(self, x): return self.seq(x).sum()
        mnc = MiniNetC()
        results["mini"]["c_full_probe"] = measure(CellFullProbe(mnc, optc), xc, "c_full_probe")

        # Compute overheads
        a = results["mini"]["a_minimal"]["avg_ms"]
        b = results["mini"]["b_no_probe"]["avg_ms"]
        c = results["mini"]["c_full_probe"]["avg_ms"]
        results["mini"]["overhead_hypermap_pct"] = round((b-a)/a*100, 2)
        results["mini"]["overhead_aicpu_pct"] = round((c-b)/b*100, 2)
        results["mini"]["overhead_total_pct"] = round((c-a)/a*100, 2)
        results["mini"]["analysis"] = {
            "hypermap_overhead_pct": f"{((b-a)/a*100):.2f}%",
            "aicpu_overhead_pct": f"{((c-b)/b*100):.2f}%",
            "total_overhead_pct": f"{((c-a)/a*100):.2f}%",
        }

    if "gpt2xl" in args.scale:
        print("\n=== GPT-2 XL ===")
        net, opt, x = make_gpt2_xl(args.device_id)
        results["gpt2xl"] = {
            "model": "GPT-2 XL, 1.56B params",
            "a_minimal": measure(CellMinimal(net, opt), x, "a_minimal"),
        }
        # Reconstruct for each variant
        netb, optb, xb = make_gpt2_xl(args.device_id)
        results["gpt2xl"]["b_no_probe"] = measure(CellNoProbe(netb, optb), xb, "b_no_probe")
        netc, optc, xc = make_gpt2_xl(args.device_id)
        results["gpt2xl"]["c_full_probe"] = measure(CellFullProbe(netc, optc), xc, "c_full_probe")

        a = results["gpt2xl"]["a_minimal"]["avg_ms"]
        b = results["gpt2xl"]["b_no_probe"]["avg_ms"]
        c = results["gpt2xl"]["c_full_probe"]["avg_ms"]
        results["gpt2xl"]["overhead_hypermap_pct"] = round((b-a)/a*100, 2)
        results["gpt2xl"]["overhead_aicpu_pct"] = round((c-b)/b*100, 2)
        results["gpt2xl"]["overhead_total_pct"] = round((c-a)/a*100, 2)
        results["gpt2xl"]["analysis"] = {
            "hypermap_overhead_pct": f"{((b-a)/a*100):.2f}%",
            "aicpu_overhead_pct": f"{((c-b)/b*100):.2f}%",
            "total_overhead_pct": f"{((c-a)/a*100):.2f}%",
        }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results -> {OUTPUT_FILE}")


if __name__ == "__main__":
    import gc
    main()
