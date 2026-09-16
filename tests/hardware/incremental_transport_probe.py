#!/usr/bin/env python3
"""Isolate native per-tick copy budget using fixed bytes and media readback."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from npu_nvme.storage.bindings import load_backend
from npu_nvme.storage.full_transport import FullTransport


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--copy-budget',type=int,required=True);parser.add_argument('--checksum-budget',type=int,default=65536);args=parser.parse_args()
    lock_path=Path('/models/npu_nvme_exp/user7-stack/hardware-campaign.lock')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if lock_path.with_suffix('.lease.json').exists():raise RuntimeError('hardware lease still present')
        args.out.mkdir(parents=True,exist_ok=False)
        os.environ['NPU_NVME_COPY_BYTES']=str(args.copy_budget)
        os.environ['NPU_NVME_CHECKSUM_BYTES']=str(args.checksum_budget)
        backend=load_backend('/models/npu_nvme_exp/user7-stack/qwen-release-runtime-20260916-001/build/libnpu_nvme.so.2.0')
        transport=FullTransport(backend,pci='0000:83:00.0',npu=-1,depth=4,chunk_size=4<<20,profiling_dir=args.out,role='host_owner')
        try:
            payload=bytes([0x59])*(4<<20);count=32;offset=4<<20
            start=time.monotonic_ns()
            for i in range(count):transport.write(offset+i*len(payload),payload)
            transport.flush();write_ns=time.monotonic_ns()-start
            start=time.monotonic_ns();digest=hashlib.sha256()
            for i in range(count):
                value=transport.read(offset+i*len(payload),len(payload))
                if value!=payload:raise ValueError('transport readback mismatch')
                digest.update(value)
            read_ns=time.monotonic_ns()-start
            result=dict(status='pass',bytes=count*len(payload),copy_budget=transport.capabilities.copy_bytes_per_tick,
                        checksum_budget=transport.capabilities.checksum_bytes_per_tick,
                        write_ns=write_ns,read_ns=read_ns,payload_sha256=digest.hexdigest())
        finally:transport.close(300)
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
