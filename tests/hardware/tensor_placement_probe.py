#!/usr/bin/env python3
"""Real allocation metadata and direct hugepage-copy regression probe."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--npu',type=int,default=0)
    parser.add_argument('--shm-id',type=int,default=78100)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',npu=None if args.cpu_only else args.npu)
    def record(): (args.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    record()
    import mindspore as ms
    import numpy as np
    from npu_nvme.framework.parameters import describe_tensor,UnknownTensorAddress
    ms.set_context(device_target='CPU' if args.cpu_only else 'Ascend',device_id=args.npu)
    scalar=ms.Parameter(ms.Tensor(np.array(0,np.int64)),name='offset')
    raw=int(scalar._data_ptr())
    host=describe_tensor(scalar,expected_device=args.npu)
    if raw==0 or host.placement!='host' or host.pointer!=0:
        raise AssertionError('Host nonzero pointer was not classified as Host')
    report['host']=dict(raw_pointer=raw,allocation=asdict(host))
    if not args.cpu_only:
        from npu_nvme.d2.rank_copy_executor import RankCopyExecutor
        from npu_nvme.storage.bindings import load_backend
        import os
        report['copies']=[]
        executor=None
        try:
            expected=np.arange((4<<20)//4,dtype=np.float32)
            tensor=ms.ops.add(ms.Tensor(expected),ms.Tensor(np.zeros_like(expected)))
            ms.hal.synchronize()
            address=describe_tensor(tensor,expected_device=args.npu)
            if address.placement!='device':raise AssertionError('NPU output was not classified as device')
            try:describe_tensor(tensor,expected_device=(args.npu+1)%8)
            except UnknownTensorAddress:pass
            else:raise AssertionError('wrong-device address accepted')
            executor=RankCopyExecutor(load_backend(os.environ['NPU_NVME_LIBRARY_PATH']),
                dict(chunk_bytes=4<<20,shm_id=args.shm_id,profiling_dir=str(args.out)),
                npu_id=args.npu,depth=4,timeout_ms=120000)
            for size in (8,4<<20):
                with executor.d2h(address.pointer,size,owners=(tensor,)) as data:
                    if bytes(data)!=expected.tobytes()[:size]:raise AssertionError('D2H bytes differ')
                payload=bytes(size)
                executor.h2d(address.pointer,payload,expected_sha256=hashlib.sha256(payload).hexdigest(),owners=(tensor,))
                expected.view(np.uint8)[:size]=0
                ms.hal.synchronize()
                if tensor.asnumpy().tobytes()!=expected.tobytes():raise AssertionError('H2D bytes differ')
                report['copies'].append(dict(bytes=size,allocation=asdict(address),status='pass'))
                record()
        finally:
            if executor and not executor.close():
                report['status']='retained';record()
                import threading
                threading.Event().wait()
    report['status']='pass';report['mindspore_version']=ms.__version__;record()


if __name__=='__main__':main()
