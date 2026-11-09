#!/usr/bin/env python3
"""Hardware bit-pattern capture proof, including signed zero and NaN payloads."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import mindspore as ms
    ms.set_context(mode=ms.PYNATIVE_MODE,device_target='Ascend',device_id=7)
    from npu_nvme.storage.bindings import load_backend
    from incremental_manifest import build_training_state_manifest
    from npu_nvme.framework.byte_capture import ByteCapture
    from npu_nvme.framework.parameters import get_dev_ptr
    value=ms.Parameter(ms.Tensor(np.asarray([0,0x7fc00001,0x3f800000,0],np.uint32).view(np.float32)),name='bits')
    ms.ops.assign(value,ms.ops.zeros((4,),ms.float32));ms.hal.synchronize()
    class Network:
        def parameters_and_names(self):return [('bits',value)]
    manifest=build_training_state_manifest({'model':Network()},block_size=1,small_threshold=0)
    backend=load_backend()
    initial=np.asarray([0,0x7fc00001,0x3f800000,0],np.uint32)
    backend.acl_lib.aclrtMemcpy.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
    assert backend.acl_lib.aclrtMemcpy(get_dev_ptr(value),initial.nbytes,initial.ctypes.data,initial.nbytes,1)==0
    capture=ByteCapture(ms,backend.acl_lib,manifest,{'model/bits':value},hbm_budget_bytes=1<<20)
    capture.capture();capture.commit_ack()
    bits=np.asarray([0x80000000,0x7fc00002,0x3f800000,1],np.uint32)
    rc=backend.acl_lib.aclrtMemcpy(get_dev_ptr(value),bits.nbytes,bits.ctypes.data,bits.nbytes,1)
    if rc:raise RuntimeError('probe H2D failed')
    flags=capture.capture();assert flags.tolist()==[True,True,False,True],flags
    capture.commit_ack();assert not capture.capture().any()
    (a.out/'result.json').write_text(json.dumps(dict(status='pass',flags=flags.tolist(),scope='NPU uint8 bit comparison and byte-preserving D2D ACK')))
if __name__=='__main__':main()
