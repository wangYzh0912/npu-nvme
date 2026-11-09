#!/usr/bin/env python3
"""Bound scoring and selection work without allocating a full model clone."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',type=int,default=7)
    parser.add_argument('--elements',type=int,default=64*1024*1024)
    parser.add_argument('--repeats',type=int,required=True)
    parser.add_argument('--select',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    import mindspore as ms
    from npu_nvme.experiments.device_score import make_score_cell
    ms.set_context(mode=ms.GRAPH_MODE,device_target='Ascend',device_id=args.device)
    rng=np.random.default_rng(42);current=rng.normal(size=args.elements).astype(np.float32)
    reference=current.copy();current[::1024]+=np.float32(.125)
    left=ms.Tensor(current);right=ms.Tensor(reference);cell=make_score_cell(ms,args.elements,65536)
    cell(left,right).asnumpy();ms.hal.synchronize();ms.runtime.reset_peak_memory_stats()
    begin=time.monotonic_ns();digest=0.0;select_ns=0
    for _ in range(args.repeats):
        scores=cell(left,right).asnumpy();digest+=float(scores.sum(dtype=np.float64))
        if args.select:
            before=time.monotonic_ns();np.argpartition(scores,-max(1,len(scores)//10));select_ns+=time.monotonic_ns()-before
    ms.hal.synchronize();elapsed=time.monotonic_ns()-begin
    result=dict(status='pass',device=args.device,elements=args.elements,repeats=args.repeats,
        select=args.select,elapsed_ns=elapsed,select_ns=select_ns,digest=digest,
        score_scan_bytes=args.repeats*args.elements*8,score_scalar_ops=args.repeats*args.elements*3,
        peak_bytes=ms.runtime.max_memory_allocated())
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
