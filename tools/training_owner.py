#!/usr/bin/env python3
"""One registered NVMe owner for the lifetime of a Qwen training process set."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connection',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    c=json.loads(args.connection.read_text())
    report=dict(status='running',pid=os.getpid(),operations=[],closed=False)
    def record():
        path=args.out/'result.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
    record();transport=None;listener=None;connections={}
    from npu_nvme.storage.bindings import load_backend
    from npu_nvme.storage.full_transport import FullTransport
    from npu_nvme.d2.backend import RegisteredBackend,registration_from_config
    from npu_nvme.d2.format import Region
    from npu_nvme.d2.training_session import TrainingOwner
    from npu_nvme.d2 import wire
    try:
        registration=registration_from_config(c['region'])
        if c['library']!=os.environ.get('NPU_NVME_LIBRARY_PATH'):
            raise ValueError('owner library differs from selected environment')
        for pci,driver in ((registration['pci_addr'],'uio_pci_generic'),(registration['protected_pci_addr'],'nvme')):
            if (Path('/sys/bus/pci/devices')/pci/'driver').resolve().name!=driver:
                raise ValueError('unexpected NVMe binding')
        affinity=sorted(os.sched_getaffinity(0));os.environ['SPDK_SHM_ID']=str(c['owner_shm_id'])
        transport=FullTransport(load_backend(c['library']),pci=registration['pci_addr'],npu=-1,
            depth=c['depth'],chunk_size=c['chunk_bytes'],profiling_dir=args.out,role='host_owner')
        if sorted(os.sched_getaffinity(0))!=affinity:raise RuntimeError('EAL changed owner caller affinity')
        backend=RegisteredBackend(transport,registration,region_id=registration['region_id'])
        region=Region(backend,offset=backend.base,length=backend.end-backend.base,retention=c['retention'])
        report['mount_errors']=region.mount()
        if region.header['region_id']!=registration['region_id'] or region.retention!=c['retention']:
            raise ValueError('mounted region differs from selected identity/retention')
        if args.inspect:
            report.update(status='pass',catalog=region.current['state'] if region.current else None,header=region.header)
            return
        listener=socket.socket(socket.AF_UNIX);listener.bind(c['socket'])
        os.chmod(c['socket'],0o600);listener.listen(4)
        deadline=time.monotonic()+c['timeout_seconds']
        state=region.current['state'] if region.current else dict(receipts=[],generations=[])
        retained={row['generation'] for row in state['generations']}
        (args.out/'ready.json').write_text(json.dumps(dict(epoch=c['epoch'],socket=c['socket'],
            receipts=[row for row in state['receipts'] if row['generation'] in retained])))
        for _ in range(4):
            listener.settimeout(max(.001,deadline-time.monotonic()));connection,_=listener.accept()
            try:
                hello,data=wire.receive(connection,deadline=deadline,max_payload=0)
                rank=hello.get('rank')
                if data or hello!=dict(kind='connect',rank=rank,epoch=c['epoch']) or type(rank) is not int or rank not in range(4) or rank in connections:
                    raise ValueError('invalid training connection')
                connections[rank]=connection
            except BaseException:connection.close();raise
        owner=TrainingOwner(region,connections,epoch=c['epoch'],identity=c['identity'],
            chunk_bytes=c['chunk_bytes'],deadline=deadline,operation_timeout=c['operation_timeout_seconds'],
            authoritative_schema=c['strategy'])
        def completed(row):report['operations'].append(row);record()
        report['operation_count']=owner.run(completed)
        report['status']='pass'
    except BaseException as error:
        report.update(status='fail',error=repr(error));raise
    finally:
        for connection in connections.values():connection.close()
        if listener:listener.close();Path(c['socket']).unlink(missing_ok=True)
        if transport:
            try:transport.close(c['copy_timeout_ms']/1000);report['closed']=True
            except BaseException as error:
                report.update(status='retained',error=repr(error));record();threading.Event().wait()
        record()


if __name__=='__main__':main()
