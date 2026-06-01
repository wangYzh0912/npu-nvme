"""AICPU WaitProbe operator smoke test.

Usage:
- python python/test_aicpu.py

Inputs:
- Uses installed custom OPP and AICPU kernel.
Outputs:
- Prints operator execution results and errors if any.
"""
import mindspore as ms
from mindspore import ops, Tensor
from mindspore.ops import CustomRegOp, DataType
import numpy as np

# 强制使用静态图模式和 Ascend 硬件
ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

print("[Test] Generating Operator Registration Info...")

# =========================================================
# 【核心修改 1】：算子名称必须与 JSON 定义中的 "op": "WaitProbe" 完全一致
# =========================================================
wait_op_info = CustomRegOp("WaitProbe") \
    .input(0, "flag") \
    .input(1, "expected") \
    .output(0, "y") \
    .dtype_format(DataType.U32_Default, DataType.U32_Default, DataType.U32_Default) \
    .target("Ascend") \
    .get_op_info()

print("[Test] Loading Native NPU Operator...")

# =========================================================
# 【核心修改 2】：彻底抛弃 .so 路径，直接传入算子名称 "WaitProbe"
# func_type 依然保持 "aicpu"，框架会自动去底层 OPP 库中匹配
# =========================================================
wait_op = ops.Custom("WaitProbe", 
                     out_shape=[1], 
                     out_dtype=ms.uint32, 
                     func_type="aicpu",
                     reg_info=wait_op_info)

# 构造 flag 与 expected
flag_tensor = ms.Tensor([0], dtype=ms.uint32)
expected_tensor = ms.Tensor([0], dtype=ms.uint32)

print("[Test] Executing wait_op with flag=1...")

class TestNet(ms.nn.Cell):
    def __init__(self, op):
        super().__init__()
        self.op = op
        
    def construct(self, flag, expected):
        return self.op(flag, expected)

net = TestNet(wait_op)

# 预期：图引擎 GE 将在 NPU 的 AICPU 核心上原生拉起这个算子，并瞬间返回！
res = net(flag_tensor, expected_tensor)
print(f"[Test] Success! Operator returned. Result: {res.asnumpy()}")