#!/usr/bin/env python3
"""Real 2/4-rank private SPDK pools, ACL copies and one Host-only NVMe owner.

Small deterministic fixture only; not Qwen, training or crash acceptance.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from npu_nvme.storage.bindings import load_backend
from npu_nvme.storage.full_transport import FullTransport
from npu_nvme.d2.rank_copy_executor import RankCopyExecutor
from npu_nvme.d2 import wire
from npu_nvme.d2.backend import RegisteredBackend,registration_from_config
from npu_nvme.d2.format import Region
from npu_nvme.d2.rank import Collective
from npu_nvme.d2.socket_service import RankService
from npu_nvme.d2.ready_protocol import ReadyDecision,coordinate
CHUNK=4<<20
SIZE=8197

def pattern(rank):return bytes((i*131+17+rank)&255 for i in range(SIZE))
def record(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def preserve(path,error):
    record(path,dict(status='retained',pid=os.getpid(),error=str(error)))
    threading.Event().wait()

def rank_main(a):
    sock=socket.socket(fileno=a.fd);deadline=time.monotonic()+300
    backend=load_backend(a.library)
    executor=None;device=C.c_void_p();safe=True
    try:
        executor=RankCopyExecutor(backend,dict(chunk_bytes=CHUNK,shm_id=a.shm_id+a.rank+1),
                                  npu_id=a.rank,depth=4,timeout_ms=120000)
        assert executor.transport.total_bytes==0
        assert executor.transport.capabilities.operation_mask==96
        raw=pattern(a.rank);host=C.create_string_buffer(raw,len(raw))
        assert backend.acl_lib.aclrtMalloc(C.byref(device),SIZE,0)==0
        assert backend.acl_lib.aclrtMemcpy(device,SIZE,host,SIZE,1)==0
        with executor.d2h(device.value,SIZE,owners=(device,)) as data:
            assert data.tobytes()==raw
            wire.send(sock,dict(kind='chunk',epoch=a.epoch,request_id=a.epoch,rank=a.rank,
                name='probe',offset=0,length=SIZE,sha256=hashlib.sha256(data).hexdigest(),lease=a.rank),data,
                deadline=deadline,max_payload=CHUNK)
            ack,_=wire.receive(sock,deadline=deadline,max_payload=0)
            assert ack==dict(kind='source_safe',lease=a.rank)
        wire.send(sock,dict(kind='complete',epoch=a.epoch,request_id=a.epoch,rank=a.rank,
            step=1,controls=dict(cursor=1,rank=a.rank)),b'',deadline=deadline,max_payload=0)
        committed,_=wire.receive(sock,deadline=deadline,max_payload=0)
        assert committed['kind']=='committed'
        control,data=wire.receive(sock,deadline=deadline,max_payload=CHUNK)
        assert control['kind']=='restore'
        executor.h2d(device.value,data,expected_sha256=hashlib.sha256(raw).hexdigest(),owners=(device,))
        with executor.d2h(device.value,SIZE,owners=(device,)) as observed:assert observed.tobytes()==raw
        wire.send(sock,dict(kind='prepared',rank=a.rank,epoch=a.epoch,
            generation=control['generation'],manifest_sha256=control['manifest_sha256'],
            transport_safe=True,schema_verified=True,controls_verified=True),b'',deadline=deadline,max_payload=0)
        release,_=wire.receive(sock,deadline=deadline,max_payload=0)
        assert release['kind']=='release' and release['ranks']==list(range(a.world_size))
        record(a.out/f'rank-{a.rank}.json',dict(status='pass',pid=os.getpid(),rank=a.rank,
            namespace_bytes=executor.transport.total_bytes,operation_mask=96,bytes=SIZE,
            status_memory=Path('/proc/self/status').read_text(),scope='small fixture, same-process target'))
    finally:
        if executor and not executor.close():preserve(a.out/f'rank-{a.rank}-retained.json','copy close unsafe')
        if device.value:backend.acl_lib.aclrtFree(device)
        sock.close()

def owner_main(a):
    a.out.mkdir(parents=True,exist_ok=False)
    os.environ['SPDK_SHM_ID']=str(a.shm_id)
    before=subprocess.check_output(['npu-smi','info'],text=True)
    (a.out/'npu-before.txt').write_text(before)
    for rank in range(a.world_size):
        if f'No running processes found in NPU {rank}' not in before:raise RuntimeError('rank device occupied')
    cfg=json.loads((ROOT/'config/d2_qwen_region.json').read_text())
    registration=registration_from_config(cfg)
    for pci,driver in ((registration['pci_addr'],'uio_pci_generic'),(registration['protected_pci_addr'],'nvme')):
        assert (Path('/sys/bus/pci/devices')/pci/'driver').resolve().name==driver
    transport=None;children=[];connections={};logs=[]
    try:
        transport=FullTransport(load_backend(a.library),pci=registration['pci_addr'],npu=-1,
            depth=4,chunk_size=CHUNK,profiling_dir=a.out,role='host_owner')
        assert transport.capabilities.operation_mask==31
        backend=RegisteredBackend(transport,registration,region_id=registration['region_id'])
        region=Region(backend,offset=backend.base,length=backend.end-backend.base,retention=3)
        region.mount();assert region.header['region_id']==registration['region_id']
        region.begin(a.epoch,dict(scope='role probe',world_size=a.world_size),reserve_bytes=4<<20)
        collective=Collective(world_size=a.world_size,epoch=a.epoch,
            schema=[dict(name='probe',partition='sharded',bytes_per_rank=[SIZE]*a.world_size)],
            chunk_bytes=CHUNK,credits=4,timeout_seconds=300)
        collective.begin(a.epoch,1)
        schema=[dict(rank=r,name='probe',shape=[SIZE],dtype='uint8',partition='sharded',bytes=SIZE) for r in range(a.world_size)]
        deadline=time.monotonic()+300
        for rank in range(a.world_size):
            parent,child=socket.socketpair();connections[rank]=parent
            log=(a.out/f'rank-{rank}.log').open('w');logs.append(log)
            cmd=[sys.executable,__file__,'--rank',str(rank),'--fd',str(child.fileno()),'--world-size',str(a.world_size),
                 '--library',str(a.library),'--out',str(a.out),'--shm-id',str(a.shm_id),'--epoch',a.epoch]
            children.append(subprocess.Popen(cmd,pass_fds=(child.fileno(),),stdout=log,stderr=subprocess.STDOUT))
            child.close()
        receipt=RankService(collective,region,wire,max_payload=CHUNK,deadline=deadline,tensor_schema=schema).run(
            connections,step=1,topology=dict(world_size=a.world_size))
        (a.out/'npu-during.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))
        selected=region.current['state']['generations'][0]
        rows=list(region.codec.read(backend,selected['pages'],region.data_base,region.end))
        digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
        decision=ReadyDecision(world_size=a.world_size,epoch=a.epoch,generation=selected['generation'],manifest_sha256=digest)
        for row in rows:
            ref=row['payload'];data=backend.read(ref['offset'],ref['length'])[:ref['logical_bytes']]
            wire.send(connections[row['rank']],dict(kind='restore',generation=selected['generation'],manifest_sha256=digest),
                data,deadline=deadline,max_payload=CHUNK)
        ready=coordinate(connections,decision,wire,deadline=deadline)
        for child in children:assert child.wait(timeout=120)==0
        record(a.out/'result.json',dict(status='pass',owner_pid=os.getpid(),owner_npu=None,
            world_size=a.world_size,receipt=receipt,ready=ready,operation_mask=31,
            library=str(a.library),library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest(),
            memory=Path('/proc/self/status').read_text(),scope='small fixture; same-process restore; no Qwen claim'))
    finally:
        for sock in connections.values():sock.close()
        for child in children:
            try:child.wait(timeout=130)
            except subprocess.TimeoutExpired:preserve(a.out/'retained-owner.json',f'child retained: {child.pid}')
        if transport:
            try:transport.close(120)
            except BaseException as error:preserve(a.out/'retained-owner.json',error)
        for log in logs:log.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--world-size',type=int,choices=[2,4],default=4);p.add_argument('--shm-id',type=int,required=True)
    p.add_argument('--rank',type=int);p.add_argument('--fd',type=int);p.add_argument('--epoch',default=uuid.uuid4().hex)
    a=p.parse_args()
    if a.rank is None:owner_main(a)
    else:rank_main(a)
if __name__=='__main__':main()
