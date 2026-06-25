#!/usr/bin/env python3
"""
P2-3 Prerequisite: GE Graph Edge Injection Impact Test.
Directly answers: Can we add hundreds of Vector ops to the fused graph
without GE compilation explosion or training slowdown?

Design:
  - Incrementally add dummy Vector operations (Sub + ReduceSum + Cast) to the
    fused graph, simulating the per-parameter delta-detection load of I3.
  - Measure: compile time, per-step time, GE graph size (if available).
  - The added ops do REAL computation on parameter tensors but don't change
    loss or optimizer — they are purely diagnostic.

Levels tested:
  L0: 0 extra ops (pure MS baseline, re-measured)
  L1: 1 extra op on 1 param (step_counter only, current FaF)
  L2: 1 extra op on ALL 772 params
  L3: Sub+ReduceSum on ALL 772 params (I3 delta detect)
  L4: Sub+ReduceSum+Cast on ALL 772 params (I3 delta detect + quant)

Config: GPT-2 XL, SEQ=1024, sink=TRUE, sink_size=10, 20 steps (2 epochs)

Output: experiments/baselines/edge_injection_test.json

Usage:
  sudo su - root -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && \
    /home/user7/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/baselines/edge_injection_test.py'
"""
import os, sys, time, json
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops, Parameter

DEVICE_ID = 1
SEQ_LEN = 1024
SINK_SIZE = 10
TOTAL_STEPS = 20  # 2 epochs × 10 steps
EPOCHS = TOTAL_STEPS // SINK_SIZE

def build_cell_with_ops(model, optimizer, ops_per_param, test_label):
    """
    Build TrainOneStepCell with configurable extra graph edges.

    ops_per_param:
      0 = no extra ops
      1 = assign_add on all params (per-param step_counter analog)
      2 = Sub + ReduceSum on all params (I3 delta detect)
      3 = Sub + ReduceSum + Cast on all params (I3 delta detect + quant)
    """
    class EdgeTestCell(nn.Cell):
        def __init__(self, network, optimizer, ops_per_param, label):
            super().__init__(auto_prefix=False)
            self.network = network
            self.network.set_grad()
            self.optimizer = optimizer
            self.grad_fn = ops.value_and_grad(self.network, grad_position=None,
                                               weights=self.optimizer.parameters)
            self.depend = ops.Depend()
            self.ops_per_param = ops_per_param
            self.label = label

            if ops_per_param > 0:
                # We need references to all parameters — use trainable_params
                self.all_params = list(self.network.trainable_params())
                print(f"  [{label}] Ops_per_param={ops_per_param}, params={len(self.all_params)}",
                      flush=True)
            else:
                self.all_params = []

        def construct(self, *inputs):
            loss, grads = self.grad_fn(*inputs)

            if self.ops_per_param == 0:
                # L0: no extra ops
                pass

            elif self.ops_per_param == 1:
                # L1: assign_add on first param only (current FaF step_counter analog)
                step = ops.assign_add(self.all_params[0], Tensor([0.0], dtype=ms.float16))
                loss = self.depend(loss, step)

            elif self.ops_per_param >= 2:
                # L2/L3: per-param Vector ops (Sub + ReduceSum + optional Cast)
                # We accumulate a scalar that is then added to loss via depend
                # to force GE to keep these ops in the fused graph.
                acc = Tensor([0.0], dtype=ms.float16)
                for p in self.all_params:
                    # Sub: param - param = 0 (we want p_cur - p_old; here use zero stub)
                    delta = ops.sub(p, p)  # real Vector op, result = zeros
                    # ReduceSum: ||delta|| (L1 norm stub)
                    reduced = ops.ReduceSum()(delta)  # real Vector op
                    acc = ops.add(acc, reduced)

                    if self.ops_per_param >= 3:
                        # Cast: FP16 -> FP32 -> FP16 (simulate quantization)
                        cast_up = ops.cast(reduced, ms.float32)
                        cast_down = ops.cast(cast_up, ms.float16)
                        acc = ops.add(acc, cast_down)

                loss = self.depend(loss, acc)

            opt_res = self.optimizer(grads)
            loss = self.depend(loss, opt_res)
            return loss

    return EdgeTestCell(model, optimizer, ops_per_param, test_label)


def run_test(ops_per_param, label):
    """Run one test level: init, compile (epoch 1), warm (epoch 2)."""
    print(f"\n{'='*60}")
    print(f"  {label}: ops_per_param={ops_per_param}")
    print(f"{'='*60}")

    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)
    print(f"  [{label}] Model built OK", flush=True)

    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(TOTAL_STEPS)

    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

    print(f"  [{label}] Building cell with ops_per_param={ops_per_param}...", flush=True)
    t_build = time.perf_counter()
    cell = build_cell_with_ops(model, opt, ops_per_param, label)
    build_s = time.perf_counter() - t_build

    epoch_times_ms = []

    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc):
            self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            et = (time.perf_counter() - self.t0) * 1000
            epoch_times_ms.append(et)

    ms_model = ms.Model(cell)

    print(f"  [{label}] Starting training ({EPOCHS} epochs × {SINK_SIZE} steps)...", flush=True)
    t_train = time.perf_counter()

    ms_model.train(epoch=EPOCHS, train_dataset=ds, callbacks=[CB()],
                   dataset_sink_mode=True, sink_size=SINK_SIZE)

    train_s = time.perf_counter() - t_train

    compile_ms = epoch_times_ms[0] if epoch_times_ms else 0
    warm_ms = epoch_times_ms[1] if len(epoch_times_ms) > 1 else 0
    warm_step_ms = warm_ms / SINK_SIZE

    return {
        "label": label,
        "ops_per_param": ops_per_param,
        "total_params": len(list(cell.all_params)) if hasattr(cell, 'all_params') else 0,
        "cell_build_s": round(build_s, 2),
        "total_train_s": round(train_s, 1),
        "compile_epoch_ms": round(compile_ms, 0),
        "warm_epoch_ms": round(warm_ms, 0),
        "warm_step_ms": round(warm_step_ms, 0),
    }


def main():
    os.makedirs(REPO + "/experiments/baselines", exist_ok=True)

    # Test levels
    tests = [
        (0, "L0: Zero extra ops"),
        (1, "L1: assign_add on 1 param (FaF step_counter)"),
        (2, "L2: Sub+ReduceSum on ALL params (I3 delta detect)"),
        (3, "L3: Sub+ReduceSum+Cast on ALL params (I3 delta+quant)"),
    ]

    results = []
    for ops_per_param, label in tests:
        r = run_test(ops_per_param, label)
        results.append(r)

    # Print comparison table
    print(f"\n\n{'='*72}")
    print(f"{'GE Edge Injection Impact Test — Results':^72}")
    print(f"{'='*72}")
    print(f"{'Test':<30} {'Compile':>10} {'Warm/step':>10} {'Total':>10}")
    print("-" * 72)
    for r in results:
        print(f"{r['label']:<30} {r['compile_epoch_ms']:>8.0f}ms {r['warm_step_ms']:>8.0f}ms {r['total_train_s']:>8.1f}s")

    # Compute overheads vs L0
    if results:
        l0 = results[0]
        print("-" * 72)
        for r in results[1:]:
            compile_delta = r['compile_epoch_ms'] - l0['compile_epoch_ms']
            step_delta = r['warm_step_ms'] - l0['warm_step_ms']
            step_pct = (r['warm_step_ms'] - l0['warm_step_ms']) / l0['warm_step_ms'] * 100 if l0['warm_step_ms'] else 0
            print(f"{r['label']} vs L0: compile={compile_delta:+.0f}ms, step={step_delta:+.0f}ms ({step_pct:+.1f}%)")

    print("=" * 72)

    # Save results
    report = {
        "test": "GE Edge Injection Impact",
        "model": "GPT-2 XL",
        "sink_size": SINK_SIZE,
        "total_steps": TOTAL_STEPS,
        "epochs": EPOCHS,
        "results": results,
    }
    outpath = REPO + "/experiments/baselines/edge_injection_test.json"
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {outpath}")
    print("[EdgeInjectionTest] DONE.", flush=True)


if __name__ == "__main__":
    main()
