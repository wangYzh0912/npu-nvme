#!/usr/bin/env python3
"""Short device probe for AIC/AIV profiling export validation."""
import json
import os
from pathlib import Path
import time

import mindspore as ms
from mindspore import Tensor, nn, ops


class MatrixWork(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul()

    def construct(self, left, right):
        return self.matmul(left, right)


class VectorWork(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = ops.Mul()
        self.add = ops.Add()

    def construct(self, value):
        return self.add(self.mul(value, 1.001), 0.001)


def main():
    out = Path(os.environ['AIC_AIV_PROBE_OUTPUT'])
    out.mkdir(parents=True, exist_ok=True)
    ms.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', device_id=0)
    left = Tensor([[0.25] * 512] * 512, ms.float16)
    right = Tensor([[0.5] * 512] * 512, ms.float16)
    vector = Tensor([0.25] * (1 << 20), ms.float16)
    matrix_work, vector_work = MatrixWork(), VectorWork()
    matrix_work(left, right)
    vector_work(vector)
    ms.hal.synchronize()
    records = []
    for name, operation, args in (
        ('matrix', matrix_work, (left, right)),
        ('vector', vector_work, (vector,)),
        ('mixed', None, None),
    ):
        started = time.monotonic_ns()
        deadline = time.monotonic() + 2.0
        count = 0
        while time.monotonic() < deadline:
            if name == 'mixed':
                matrix_work(left, right)
                vector_work(vector)
            else:
                operation(*args)
            count += 1
        ms.hal.synchronize()
        records.append(dict(stage=name, begin_ns=started, end_ns=time.monotonic_ns(),
                            iterations=count))
    (out / 'probe-stages.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(dict(status='pass', stages=records)))


if __name__ == '__main__':
    main()
