#!/usr/bin/env python3
"""Small fresh-process fixture exercising the same sessions as training."""
import argparse
from contextlib import contextmanager
import ctypes as C
import hashlib
import json
from pathlib import Path
import socket
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'python'))
from npu_nvme.d2 import wire
from npu_nvme.d2.session import RankSession
from npu_nvme.d2.rank_copy_executor import RankCopyExecutor
from npu_nvme.storage.bindings import load_backend


def main():
    p=argparse.ArgumentParser();p.add_argument('--operation',choices=['save','restore'],required=True)
    p.add_argument('--socket',required=True);p.add_argument('--epoch',required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--shm-id',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+300;backend=load_backend(a.library);executor=None;device=C.c_void_p();sock=socket.socket(socket.AF_UNIX)
    size=(4<<20)+8197;expected=bytes((i*131+a.rank)&255 for i in range(size))
    try:
        executor=RankCopyExecutor(backend,dict(chunk_bytes=4<<20,shm_id=a.shm_id),npu_id=a.rank,depth=4,timeout_ms=120000)
        assert backend.acl_lib.aclrtMalloc(C.byref(device),size,0)==0
        initial=C.create_string_buffer(expected if a.operation=='save' else bytes(size),size)
        assert backend.acl_lib.aclrtMemcpy(device,size,initial,size,1)==0
        sock.connect(a.socket);wire.send(sock,dict(kind='connect',rank=a.rank,epoch=a.epoch),b'',deadline=deadline,max_payload=0)
        session=RankSession(sock,rank=a.rank,epoch=a.epoch,identity=json.loads(a.identity.read_text()),
            schema=[dict(rank=a.rank,name='tensor',shape=[size],dtype='uint8',partition='sharded',bytes=size)],chunk_bytes=4<<20,deadline=deadline)
        if a.operation=='save':
            @contextmanager
            def read(name,offset,length):
                with executor.d2h(device.value+offset,length,owners=(device,)) as data:yield data
            receipt=session.save(read,dict(cursor=8,rank=a.rank),step=8)
        else:
            def apply(name,offset,data,digest):executor.h2d(device.value+offset,data,expected_sha256=digest,owners=(device,))
            def controls(value,step):
                assert value==dict(cursor=8,rank=a.rank) and step==8
                h=hashlib.sha256()
                for offset in range(0,size,4<<20):
                    with executor.d2h(device.value+offset,min(4<<20,size-offset),owners=(device,)) as data:h.update(data)
                assert h.hexdigest()==hashlib.sha256(expected).hexdigest()
            receipt=session.restore(apply,controls)
        (a.out/'result.json').write_text(json.dumps(dict(status='pass',receipt=receipt,operation=a.operation,rank=a.rank,bytes=size)))
    finally:
        sock.close()
        if executor and not executor.close():
            (a.out/'retained.json').write_text('{}');threading.Event().wait()
        if device.value:backend.acl_lib.aclrtFree(device)
if __name__=='__main__':main()
