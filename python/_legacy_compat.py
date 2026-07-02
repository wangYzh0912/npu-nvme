"""Legacy compatibility — DEPRECATED WaitProbe-era symbols.

These are kept ONLY for backward compatibility with two legacy experiment
scripts: cell_overhead_analysis.py and operator_microbenchmarks.py.

New code MUST use the FaF step_counter path (ProbeTrainOneStepCell with
enable_probe=True) instead.
"""

from mindspore import ops
from mindspore.ops import MultitypeFuncGraph, CustomRegOp, DataType


# -- Legacy bind_depend_op --
bind_depend_op = MultitypeFuncGraph("bind_depend_op")
@bind_depend_op.register("Tensor", "Tensor")
def _bind_depend_op(sig, grad):
    return ops.depend(grad, sig)


# -- Legacy WaitProbe AICPU custom-op registration --
# The GE compiler cannot load custom AICPU kernels in sink=TRUE mode,
# so the WaitProbe path was replaced by the FaF step_counter listener.
wait_op_info = CustomRegOp("WaitProbe") \
    .input(0, "flag") \
    .input(1, "expected") \
    .output(0, "y") \
    .dtype_format(DataType.U32_Default, DataType.U32_Default,
                  DataType.U32_Default) \
    .target("Ascend") \
    .get_op_info()

trigger_op_info = CustomRegOp("TriggerProbe") \
    .input(0, "step") \
    .input(1, "interval") \
    .input(2, "trigger_buf") \
    .input(3, "expected") \
    .output(0, "y") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default,
                  DataType.U32_Default, DataType.U32_Default,
                  DataType.I32_Default) \
    .target("Ascend") \
    .get_op_info()
