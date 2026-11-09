"""Device-side block scoring with bounded, per-parameter intermediates."""
from __future__ import annotations
from contextlib import contextmanager


@contextmanager
def local_graph(ms):
    """Local shards are already partitioned; never shard score inputs again."""
    mode=ms.get_auto_parallel_context('parallel_mode')
    full_batch=ms.get_auto_parallel_context('full_batch')
    ms.set_auto_parallel_context(parallel_mode='stand_alone',full_batch=False)
    try:yield
    finally:ms.set_auto_parallel_context(parallel_mode=mode,full_batch=full_batch)


class LocalScore:
    def __init__(self,ms,cell):self.ms=ms;self.cell=cell
    def __call__(self,*args):
        with local_graph(self.ms):return self.cell(*args)



def make_score_cell(ms, element_count, block_elements):
    if type(element_count) is not int or element_count <= 0 or block_elements <= 0:
        raise ValueError("invalid score geometry")
    padded = ((element_count + block_elements - 1) // block_elements) * block_elements
    tail = padded - element_count

    class BlockScore(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.tail = tail
            self.blocks = padded // block_elements
            self.flatten = ms.ops.Reshape()
            self.reduce = ms.ops.ReduceSum(keep_dims=False)
            self.zeros = ms.Tensor([0.0] * tail, ms.float32) if tail else None

        def construct(self, current, reference):
            difference = self.flatten(current, (element_count,)) - self.flatten(reference, (element_count,))
            if self.tail:
                difference = ms.ops.concat((difference, self.zeros), axis=0)
            difference = self.flatten(difference, (self.blocks, block_elements))
            return self.reduce(difference * difference, 1)

    return LocalScore(ms,BlockScore())


def score_parameter(ms, current, reference, *, block_elements=65536):
    cell = make_score_cell(ms, int(current.size), block_elements)
    result = cell(current, reference)
    return result, cell

