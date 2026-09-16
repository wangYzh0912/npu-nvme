#!/usr/bin/env python3
"""One Host-only owner for a registered D2 save or fresh-process restore."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from npu_nvme.storage.bindings import load_backend
from npu_nvme.storage.full_transport import FullTransport
from npu_nvme.d2.backend import RegisteredBackend,registration_from_config
from npu_nvme.d2.format import Region
from npu_nvme.d2.session import OwnerSession
from npu_nvme.d2 import wire


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operation',choices=['save','restore'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--socket',required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--identity',type=Path,required=True)
    p.add_argument('--epoch',required=True);p.add_argument('--request-id',required=True)
    p.add_argument('--world-size',type=int,choices=[2,4],default=4)
    p.add_argument('--shm-id',type=int,required=True);p.add_argument('--step',type=int,default=8)
    p.add_argument('--generation',type=int);p.add_argument('--timeout',type=int,default=7200)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',pid=os.getpid(),operation=a.operation,world_size=a.world_size,
        library=str(a.library),library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    def record():(a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    record();transport=None;connections={};listener=None
    try:
        if a.timeout<=0 or len(a.socket.encode())>=104:raise ValueError('session deadline/socket bounds')
        registration=registration_from_config(ROOT/'config/d2_qwen_region.json')
        for pci,driver in ((registration['pci_addr'],'uio_pci_generic'),(registration['protected_pci_addr'],'nvme')):
            if (Path('/sys/bus/pci/devices')/pci/'driver').resolve().name!=driver:raise ValueError('device binding differs')
        os.environ['SPDK_SHM_ID']=str(a.shm_id)
        transport=FullTransport(load_backend(a.library),pci=registration['pci_addr'],npu=-1,depth=4,
            chunk_size=4<<20,profiling_dir=a.out,role='host_owner')
        backend=RegisteredBackend(transport,registration,region_id=registration['region_id'])
        region=Region(backend,offset=backend.base,length=backend.end-backend.base,retention=3)
        report['mount_errors']=region.mount()
        if region.header['region_id']!=registration['region_id']:raise ValueError('D2 identity differs')
        listener=socket.socket(socket.AF_UNIX);listener.bind(a.socket);os.chmod(a.socket,0o600);listener.listen(a.world_size)
        deadline=time.monotonic()+a.timeout
        (a.out/'ready.json').write_text(json.dumps(dict(socket=a.socket,epoch=a.epoch,pid=os.getpid())))
        for _ in range(a.world_size):
            listener.settimeout(max(.001,deadline-time.monotonic()));conn,_=listener.accept()
            try:
                hello,data=wire.receive(conn,deadline=deadline,max_payload=0)
                rank=hello.get('rank')
                if data or hello.get('kind')!='connect' or hello.get('epoch')!=a.epoch or type(rank) is not int or not 0<=rank<a.world_size or rank in connections:raise ValueError('connection identity')
                connections[rank]=conn
            except BaseException:conn.close();raise
        session=OwnerSession(region,connections,epoch=a.epoch,request_id=a.request_id,
            topology=dict(world_size=a.world_size),chunk_bytes=4<<20,deadline=deadline,
            identity=json.loads(a.identity.read_text()))
        started=time.monotonic()
        report['receipt']=session.save(step=a.step) if a.operation=='save' else session.restore(generation=a.generation)
        report.update(status='pass',seconds=time.monotonic()-started,process_status=Path('/proc/self/status').read_text())
    except BaseException as error:
        report.update(status='fail',error=repr(error));raise
    finally:
        for conn in connections.values():conn.close()
        if listener:listener.close();Path(a.socket).unlink(missing_ok=True)
        if transport:
            try:transport.close(120);report['closed']=True
            except BaseException as error:
                report.update(status='retained',closed=False,error=repr(error));record();threading.Event().wait()
        record()
if __name__=='__main__':main()
