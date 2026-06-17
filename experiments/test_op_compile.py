#!/usr/bin/env python3
"""Minimal test to verify GE loads WaitProbe + TriggerProbe without Dlsym errors."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import numpy as np
import mindspore as ms
from mindspore import context, Tensor, nn, ops

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=1)
ms.common.set_seed(42)

from direct_checkpoint import wait_op_info, trigger_op_info

class TestCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.flag = ms.Parameter(Tensor([0], dtype=ms.uint32), name="flag")
        self.expected = ms.Parameter(Tensor([0], dtype=ms.uint32), name="expected")
        self.step_counter = ms.Parameter(Tensor([0], dtype=ms.int32), name="step")
        self.one_i32 = Tensor([1], dtype=ms.int32)
        self.interval_i32 = Tensor([10], dtype=ms.int32)
        self.trigger_buf = ms.Parameter(Tensor([0], dtype=ms.uint32), name="trig")
        self.wait_probe = ops.Custom("WaitProbe", out_shape=[1], out_dtype=ms.uint32,
                                       func_type="aicpu", reg_info=wait_op_info)
        self.trigger_probe = ops.Custom("TriggerProbe", out_shape=[1], out_dtype=ms.int32,
                                          func_type="aicpu", reg_info=trigger_op_info)

    def construct(self, x):
        s = ops.assign_add(self.step_counter, self.one_i32)
        _ = self.trigger_probe(s, self.interval_i32, self.trigger_buf, self.expected)
        w = self.wait_probe(self.flag, self.expected)
        return x + 1.0

cell = TestCell()
print("Cell created, compiling graph...", flush=True)
x = Tensor(np.random.randn(2, 4).astype(np.float32), ms.float32)
y = cell(x)
print(f"Graph compiled! output={y}", flush=True)
