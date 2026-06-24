"""Training cell with Fire-and-Forget step_counter injection for SPDK.

Provides ProbeTrainOneStepCell — a MindSpore nn.Cell wrapper that injects
a per-step counter into the GE graph.  The C-layer background listener
polls this counter via aclrtMemcpy and triggers SPDK writes autonomously.
"""

import mindspore as ms
from mindspore import ops, nn, Tensor
from mindspore.ops import HyperMap


class ProbeTrainOneStepCell(nn.Cell):
    """Training cell with optional FaF step_counter injection.

    When enable_probe=True, the GE graph increments a step_counter Parameter
    each training step.  The C-layer background listener polls step_counter
    via aclrtMemcpy and triggers SPDK writes autonomously.
    """

    def __init__(self, network, optimizer, enable_probe=True, ckpt_interval=10):
        super().__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.grad_fn = ops.value_and_grad(
            self.network, grad_position=None, weights=self.optimizer.parameters)

        self.enable_probe = enable_probe
        self.depend = ops.Depend()
        self.hyper_map = HyperMap()
        self.ckpt_interval = ckpt_interval

        if self.enable_probe:
            self.flag = ms.Parameter(
                ms.Tensor([0], dtype=ms.uint32),
                requires_grad=False, name="probe_flag")
            self.step_counter = ms.Parameter(
                ms.Tensor([0], dtype=ms.int32),
                requires_grad=False, name="step_counter")
            self.one_i32 = Tensor([1], dtype=ms.int32)

    def construct(self, *inputs):
        if not self.enable_probe:
            loss, grads = self.grad_fn(*inputs)
            opt_res = self.optimizer(grads)
            loss = self.depend(loss, opt_res)
            return loss

        loss, grads = self.grad_fn(*inputs)

        # Fire-and-Forget: step_counter auto-increments each step.
        # C layer listener polls step_counter directly (no AICPU kernel
        # needed).  This avoids the GE aclnn wrapper issue that prevents
        # custom AICPU kernels from launching in sink=TRUE fused graphs.
        step = ops.assign_add(self.step_counter, self.one_i32)
        loss = self.depend(loss, step)

        opt_res = self.optimizer(grads)
        loss = self.depend(loss, opt_res)

        return loss
