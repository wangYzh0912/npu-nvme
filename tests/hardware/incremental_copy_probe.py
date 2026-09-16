#!/usr/bin/env python3
"""Measure D2D-pack and D2H copy boundaries with consumed output."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time

from npu_nvme.storage.bindings import load_backend


def check(rc,name):
    if rc:raise RuntimeError(f'{name}: {rc}')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--device',type=int,default=7);parser.add_argument('--bytes',type=int,required=True);parser.add_argument('--granularity',type=int,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    acl=load_backend('/models/npu_nvme_exp/user7-stack/qwen-release-runtime-20260916-001/build/libnpu_nvme.so.2.0').acl_lib
    check(acl.aclrtSetDevice(args.device),'set device');stream=C.c_void_p();source=C.c_void_p();staging=C.c_void_p();host=C.c_void_p()
    check(acl.aclrtCreateStream(C.byref(stream)),'stream');size=256<<20
    check(acl.aclrtMalloc(C.byref(source),size,0),'source');check(acl.aclrtMalloc(C.byref(staging),size,0),'staging');check(acl.aclrtMallocHost(C.byref(host),size),'host')
    try:
        C.memset(host,0x5a,size)
        check(acl.aclrtMemcpyAsync(source,size,host,size,1,stream),'initialize source')
        check(acl.aclrtSynchronizeStream(stream),'initialize sync')
        check(acl.aclrtMemcpyAsync(staging,size,source,size,3,stream),'warm pack')
        check(acl.aclrtSynchronizeStream(stream),'warm sync')
        C.memset(host,0,size)
        iterations=(args.bytes+args.granularity-1)//args.granularity;begin=time.monotonic_ns()
        for index in range(iterations):
            count=min(args.granularity,args.bytes-index*args.granularity);offset=(index*args.granularity)%size
            check(acl.aclrtMemcpyAsync(C.c_void_p(staging.value+offset),count,C.c_void_p(source.value+offset),count,3,stream),'D2D')
        check(acl.aclrtSynchronizeStream(stream),'D2D sync');d2d=time.monotonic_ns()-begin;begin=time.monotonic_ns()
        for index in range(iterations):
            count=min(args.granularity,args.bytes-index*args.granularity);offset=(index*args.granularity)%size
            check(acl.aclrtMemcpyAsync(C.c_void_p(host.value+offset),count,C.c_void_p(staging.value+offset),count,2,stream),'D2H')
        check(acl.aclrtSynchronizeStream(stream),'D2H sync');d2h=time.monotonic_ns()-begin
        import numpy as np
        value=np.ctypeslib.as_array((C.c_ubyte*min(size,args.bytes)).from_address(host.value))
        if not np.all(value==0x5a):raise ValueError('copied payload differs')
        digest=hashlib.sha256(memoryview(value)).hexdigest()
        result=dict(status='pass',bytes=args.bytes,granularity=args.granularity,tasks=iterations,d2d_ns=d2d,d2h_ns=d2h,consumed_sha256=digest)
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:
        check(acl.aclrtFreeHost(host),'free host');check(acl.aclrtFree(staging),'free staging');check(acl.aclrtFree(source),'free source');check(acl.aclrtDestroyStream(stream),'destroy stream')


if __name__=='__main__':main()
