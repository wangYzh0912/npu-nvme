"""Verify all eight devices and HCCL after a collective device reset."""
import argparse
import json
import os
from pathlib import Path

import mindspore as ms
import numpy as np
from mindspore.communication import init,get_rank,get_group_size
from mindspore.communication.comm_func import all_reduce,barrier


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    device=int(os.environ.get('DEVICE_ID',os.environ.get('RANK_ID','0')))
    ms.set_context(mode=ms.PYNATIVE_MODE,device_target='Ascend',device_id=device)
    init();rank=get_rank();world=get_group_size()
    if world!=8 or rank!=device:raise ValueError('expected eight physical ranks')
    value=ms.Tensor(np.full(1024,rank+1,np.float32))
    result=all_reduce(value)
    if isinstance(result,tuple):result=result[0]
    ms.hal.synchronize()
    if not np.array_equal(result.asnumpy(),np.full(1024,36,np.float32)):
        raise ValueError('eight-rank allreduce mismatch')
    barrier();args.out.mkdir(parents=True,exist_ok=True)
    (args.out/f'rank-{rank}.json').write_text(json.dumps(dict(status='pass',rank=rank,device=device,
        world_size=world,allreduce_sum=36,mindspore=ms.__version__))+'\n')


if __name__=='__main__':main()
