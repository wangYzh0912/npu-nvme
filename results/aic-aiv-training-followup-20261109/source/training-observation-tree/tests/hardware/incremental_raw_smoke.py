#!/usr/bin/env python3
"""Four-client real-media smoke test for the experiment owner."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

from npu_nvme.d2 import wire


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);sock='/tmp/incremental-raw-smoke.sock'
    config=dict(library='/models/npu_nvme_exp/user7-stack/qwen-release-runtime-20260916-001/build/libnpu_nvme.so.2.0',
        pci_addr='0000:83:00.0',chunk_bytes=4<<20,timeout_seconds=300,
        config_sha256='1'*64,campaign_id='incremental-raw-smoke',catalog=str(args.out/'catalog.json'),
        socket=sock,maximum_run_bytes=1<<30)
    connection=args.out/'connection.json';connection.write_text(json.dumps(config)+'\n')
    root=Path(__file__).resolve().parents[2]
    manifest='/models/npu_nvme_exp/user7-stack/qwen-release-runtime-20260916-001/environments.json'
    process=subprocess.Popen([sys.executable,'scripts/run_user_environment.py','--manifest',manifest,
        '--profile','candidate','--','python','tools/incremental_raw_owner.py','--connection',str(connection),
        '--out',str(args.out/'owner')],cwd=root)
    deadline=time.monotonic()+180
    while not (args.out/'owner/ready.json').exists():
        if process.poll() is not None:raise RuntimeError('owner exited before ready')
        if time.monotonic()>deadline:raise TimeoutError('owner ready')
        time.sleep(.05)
    errors=[]
    def client(rank):
        try:
            s=socket.socket(socket.AF_UNIX);s.connect(sock);payload=bytes([rank+1])*4096
            wire.send(s,dict(kind='hello',rank=rank),b'',deadline=deadline,max_payload=0)
            descriptor=dict(schema_version=1,rank=rank,logical_step=1,records=[])
            digest=hashlib.sha256(payload).hexdigest()
            wire.send(s,dict(kind='begin',rank=rank,operation=0,run_id='smoke',step=1,
                payload_bytes=len(payload),payload_sha256=digest),json.dumps(descriptor).encode(),deadline=deadline,max_payload=256<<20)
            assigned,extra=wire.receive(s,deadline=deadline,max_payload=0)
            if extra or assigned['payload_bytes']!=len(payload):raise ValueError('assignment')
            wire.send(s,dict(kind='chunk',offset=0,bytes=len(payload),sha256=digest),payload,deadline=deadline,max_payload=4<<20)
            committed,extra=wire.receive(s,deadline=deadline,max_payload=0)
            if extra or committed['receipt']['step']!=1:raise ValueError('commit')
            wire.send(s,dict(kind='close',rank=rank,operation=1),b'',deadline=deadline,max_payload=0)
            closed,extra=wire.receive(s,deadline=deadline,max_payload=0)
            if extra or closed!=dict(kind='closed',rank=rank):raise ValueError('close')
            s.close()
        except BaseException as error:errors.append(repr(error))
    threads=[threading.Thread(target=client,args=(rank,)) for rank in range(4)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    process.wait(timeout=300)
    owner=json.loads((args.out/'owner/result.json').read_text())
    result=dict(status='pass' if not errors and process.returncode==0 and owner==dict(status='pass',closed=True,commits=1) else 'fail',
                errors=errors,owner=owner,returncode=process.returncode)
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return int(result['status']!='pass')


if __name__=='__main__':raise SystemExit(main())
