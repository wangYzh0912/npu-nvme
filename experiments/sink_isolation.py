#!/usr/bin/env python3
"""
Minimal sink isolation — CellMinimal with real GPT-2 dataset, sink=True vs sink=False.
Output: experiments/output/sink_isolation.json
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import time, json, warnings, gc
import numpy as np
import mindspore as ms
from mindspore import nn, context, ops, Tensor

warnings.filterwarnings("ignore")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DEVICE_ID = 1; WARMUP = 5; MEASURE = 30

class CellMinimal(nn.Cell):
    def __init__(self, net, opt):
        super().__init__(auto_prefix=False)
        self.net = net; self.net.set_grad()
        self.opt = opt
        self.grad_fn = ops.value_and_grad(self.net, grad_position=None, weights=self.opt.parameters)
    def construct(self, x):
        loss, grads = self.grad_fn(x)
        return ops.depend(loss, self.opt(grads))

def run(label, ds_sink):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)
    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = 1024; cfg.max_position_embeddings = 1024
    net = AutoModel.from_config(cfg)
    opt = nn.AdamWeightDecay(net.trainable_params(), learning_rate=1e-5)
    cell = CellMinimal(net, opt)
    ds = ms.dataset.MindDataset(
        "/home/user7/npu-nvme/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=False).batch(1, drop_remainder=True).take(WARMUP+MEASURE+5)
    step_times = []
    class Cb(ms.Callback):
        def on_train_step_begin(self, rc): self._t0 = time.perf_counter()
        def on_train_step_end(self, rc): step_times.append((time.perf_counter()-self._t0)*1000)
    model = ms.Model(cell)
    model.train(epoch=1, train_dataset=ds, callbacks=[Cb()], dataset_sink_mode=ds_sink)
    steady = step_times[WARMUP:]
    avg = round(float(np.mean(steady)), 1)
    print(f"  {label}: avg={avg}ms (n={len(steady)})")
    return {"avg_ms": avg, "p99_ms": round(float(np.percentile(steady,99)),1), "n": len(steady)}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}
    print("=== (A) CellMinimal + sink=TRUE ===")
    results["A_minimal_sink_TRUE"] = run("A", ds_sink=True)
    print("\n=== (B) CellMinimal + sink=FALSE ===")
    results["B_minimal_sink_FALSE"] = run("B", ds_sink=False)
    if results["A_minimal_sink_TRUE"]["avg_ms"] > 0 and results["B_minimal_sink_FALSE"]["avg_ms"] > 0:
        a=results["A_minimal_sink_TRUE"]["avg_ms"]; b=results["B_minimal_sink_FALSE"]["avg_ms"]
        results["analysis"] = {"sink_FALSE_overhead_ms": round(b-a,1), "sink_FALSE_slowdown": round(b/a,2)}
        print(f"\n  sink=FALSE overhead: +{b-a:.0f}ms (×{b/a:.1f})")
    with open(os.path.join(OUTPUT_DIR, "sink_isolation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("[OK]")

if __name__ == "__main__":
    main()
