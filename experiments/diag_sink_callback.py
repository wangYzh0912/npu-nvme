#!/usr/bin/env python3
"""
Minimal diagnostic: verify callback firing behavior under dataset_sink_mode=True.
Prints EVERY callback event to see what actually fires and when.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import time, numpy as np
import mindspore as ms
from mindspore import nn, context

DEVICE_ID = 1
TRAIN_MR = "/home/user7/npu-nvme/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord"
TOTAL_STEPS = 15


class DiagnosticCallback(ms.Callback):
    def __init__(self):
        super().__init__()
        self.events = []

    def on_train_epoch_begin(self, rc):
        print(f"  [DIAG] on_train_epoch_begin", flush=True)
        self.events.append(("epoch_begin", time.perf_counter()))

    def on_train_epoch_end(self, rc):
        print(f"  [DIAG] on_train_epoch_end", flush=True)
        self.events.append(("epoch_end", time.perf_counter()))

    def on_train_step_begin(self, rc):
        cb = rc.original_args()
        step = cb.cur_step_num
        print(f"  [DIAG] on_train_step_begin  step={step}", flush=True)
        self.events.append((f"step_{step}_begin", time.perf_counter()))

    def on_train_step_end(self, rc):
        cb = rc.original_args()
        step = cb.cur_step_num
        print(f"  [DIAG] on_train_step_end    step={step}", flush=True)
        self.events.append((f"step_{step}_end", time.perf_counter()))


def main():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    print("[Diag] Building model...", flush=True)
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = 1024
    cfg.max_position_embeddings = 1024
    net = AutoModel.from_config(cfg)
    opt = nn.AdamWeightDecay(net.trainable_params(), learning_rate=1e-5)

    # Simplest possible Cell: fwd + bwd + opt, NO probe
    class SimpleCell(nn.Cell):
        def __init__(self, network, optimizer):
            super().__init__(auto_prefix=False)
            self.net = network
            self.net.set_grad()
            self.opt = optimizer
            self.grad_fn = ops.value_and_grad(self.net, grad_position=None,
                                               weights=self.opt.parameters)
        def construct(self, *inputs):
            loss, grads = self.grad_fn(*inputs)
            return ops.depend(loss, self.opt(grads))

    ms.common.set_seed(42)
    from mindspore import ops
    cell = SimpleCell(net, opt)

    ds = ms.dataset.MindDataset(TRAIN_MR, shuffle=False)
    ds = ds.batch(1, drop_remainder=True).take(TOTAL_STEPS)

    cb = DiagnosticCallback()
    model = ms.Model(cell)

    print(f"\n[Diag] === Training with sink=TRUE, {TOTAL_STEPS} steps ===\n", flush=True)
    t0 = time.perf_counter()
    model.train(epoch=1, train_dataset=ds, callbacks=[cb], dataset_sink_mode=True)
    elapsed = time.perf_counter() - t0

    print(f"\n[Diag] Total: {elapsed:.1f}s, events={len(cb.events)}", flush=True)
    for evt in cb.events:
        print(f"  {evt}", flush=True)


if __name__ == "__main__":
    main()
