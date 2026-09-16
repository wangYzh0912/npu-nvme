#!/usr/bin/env python3
"""Actual async H2D negative cases; a timeout never kills an unknown DMA owner."""
import argparse
import ctypes as C
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import zlib
from stage4_fault_lifecycle import Worker, require, write_json, manifest, ROOT, OFFSET, ALIGN, PAYLOAD
from npu_nvme.storage.bindings import NPUNVMETransferItem, NPUNVMETransferSpec, NPUNVMETransferReceipt

FAULTS={
 'checksum':[],
 'submit':['NPU_NVME_TEST_FAIL_NVME_READ_SUBMIT'],
 'completion':['NPU_NVME_TEST_FAIL_NVME_READ_COMPLETION'],
 'record':['NPU_NVME_TEST_FAIL_EVENT_RECORD'],
 'query':['NPU_NVME_TEST_FAIL_EVENT_QUERY'],
 'late':['NPU_NVME_TEST_HOLD_NVME_COMPLETIONS'],
 'record-quarantine':['NPU_NVME_TEST_FAIL_EVENT_RECORD','NPU_NVME_TEST_FAIL_STREAM_SYNC'],
 'query-quarantine':['NPU_NVME_TEST_FAIL_EVENT_QUERY','NPU_NVME_TEST_FAIL_EVENT_SYNC','NPU_NVME_TEST_FAIL_STREAM_SYNC'],
}


def execute(a):
    for name in list(os.environ):
        if name.startswith('NPU_NVME_TEST_'):os.environ.pop(name)
    os.environ['SPDK_SHM_ID']=str(a.shm_id)
    record=dict(status='fail',case=a.case,pid=os.getpid(),scope='real async H2D with software fault injection')
    w=None
    try:
        w=Worker(a,record);w.open();record['superblock_before']=w.superblock()
        if a.case=='verify':
            w.verify();w.close();record.update(status='pass',safe_process_exit=True)
        else:
            require(w.host_write()==0 and w.lib.npu_nvme_flush(w.ctx)==0,'baseline write/flush failed')
            target=w.device_source();canary=w.buffer(b'Z'*ALIGN)
            require(w.acl.aclrtMemcpy(C.c_void_p(target),ALIGN,canary,ALIGN,1)==0,'target initialization')
            item=NPUNVMETransferItem(target,OFFSET,ALIGN,zlib.crc32(PAYLOAD))
            item.expected_sha256[:]=hashlib.sha256(PAYLOAD).digest()
            if a.case=='checksum':item.expected_sha256[0]^=1
            spec=NPUNVMETransferSpec(C.sizeof(NPUNVMETransferSpec),1,1,0,3,1,C.pointer(item))
            for name in FAULTS[a.case]:os.environ[name]='1'
            req=C.POINTER(w.request_type)()
            require(w.lib.npu_nvme_submit_transfer(w.ctx,C.byref(spec),C.byref(req))==0,'read admission')
            w.requests.append(req)
            rc=w.lib.npu_nvme_wait_request(req,50 if a.case=='late' or 'quarantine' in a.case else 5000)
            record['wait_rc']=rc;record['snapshot']=w.snapshot()
            if 'quarantine' in a.case:
                require(rc==-errno.ETIMEDOUT,'quarantine falsely terminal')
                receipt=NPUNVMETransferReceipt()
                require(w.lib.npu_nvme_get_transfer_receipt(req,C.byref(receipt),C.sizeof(receipt))==-errno.EAGAIN,'unsafe receipt')
                require(w.lib.npu_nvme_close(w.ctx,50)==-errno.EIO,'unsafe close')
                snap=w.snapshot();require(snap['retained_slots'] and snap['stats']['dma_inflight']==1,'DMA owner not retained')
                w.acl.aclrtSynchronizeDevice.argtypes=[];w.acl.aclrtSynchronizeDevice.restype=C.c_int
                stop=w.acl.aclrtSynchronizeDevice()
                require(stop==0 and snap['stats']['nvme_outstanding']==0,'physical stop not proven')
                record.update(driver_stop_rc=stop,production_cleanup_performed=False,safe_process_exit=True,status='pass')
            else:
                if a.case=='late':
                    require(rc==-errno.ETIMEDOUT,'late completion not observed')
                    w.lib.npu_nvme_release_request(req);w.requests.remove(req);req=None
                    os.environ.pop('NPU_NVME_TEST_HOLD_NVME_COMPLETIONS')
                    require(w.lib.npu_nvme_wait_quiescent(w.ctx,10000)==0,'late completion did not drain')
                else:
                    require(rc<0,'injected read reported success')
                    receipt=NPUNVMETransferReceipt()
                    require(w.lib.npu_nvme_get_transfer_receipt(req,C.byref(receipt),C.sizeof(receipt))==0,'failure receipt missing')
                    require(receipt.transport_safe and receipt.result<0 and not receipt.data_durable,'invalid failed receipt')
                result=w.buffer(b'\0'*ALIGN)
                require(w.acl.aclrtMemcpy(result,ALIGN,C.c_void_p(target),ALIGN,2)==0,'independent target oracle')
                if a.case in ('checksum','submit','completion'):
                    require(result.raw==b'Z'*ALIGN,'unverified data reached target')
                    require(w.snapshot()['stats']['async_dma_submit_count']==0,'failed integrity submitted H2D')
                if a.case=='late':require(result.raw==PAYLOAD,'late target bytes differ')
                for name in FAULTS[a.case]:os.environ.pop(name,None)
                w.close();record.update(status='pass',safe_process_exit=True)
    except BaseException:
        record['error']=traceback.format_exc()
        if w is None or not w.ctx:record['safe_process_exit']=True
        else:
            try:
                os.environ.pop('NPU_NVME_TEST_HOLD_NVME_COMPLETIONS',None)
                rc=w.lib.npu_nvme_close(w.ctx,10000)
                if rc==0:w.close();record['safe_process_exit']=True
                else:
                    w.acl.aclrtSynchronizeDevice.argtypes=[];w.acl.aclrtSynchronizeDevice.restype=C.c_int
                    stop=w.acl.aclrtSynchronizeDevice()
                    record['safe_process_exit']=rc==-errno.EIO and stop==0 and w.snapshot()['stats']['nvme_outstanding']==0
            except BaseException:record['safe_process_exit']=False
    write_json(a.output/'worker.json',record)
    sys.stdout.flush();sys.stderr.flush()
    if not record.get('safe_process_exit'):threading.Event().wait()
    os._exit(0 if record['status']=='pass' else 1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=[*FAULTS,'verify']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,default=ROOT/'build/libnpu_nvme.so')
    p.add_argument('--shm-id',type=int,default=61900)
    a=p.parse_args();a.output=a.output.resolve();a.library=a.library.resolve()
    a.pci='0000:83:00.0';a.npu=7;a.depth=2;a.close_timeout_ms=10000
    require(os.geteuid()==0,'hardware worker requires root')
    require(Path('/sys/bus/pci/devices/0000:84:00.0/driver').resolve().name=='nvme','protected filesystem driver changed')
    a.output.mkdir(parents=True,exist_ok=False)
    if a.case:execute(a)
    source={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest()
            for top in ('src','include','python','tests') for q in (ROOT/top).rglob('*')
            if q.is_file() and q.suffix in ('.c','.h','.py')}
    write_json(a.output/'source.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=source,
                                         library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest()))
    records=[]
    for case in FAULTS:
        before=None
        for phase in (case,'verify'):
            directory=a.output/f'{case}-{phase}'
            argv=[sys.executable,str(Path(__file__).resolve()),'--case',phase,'--output',str(directory),
                  '--library',str(a.library),'--shm-id',str(a.shm_id+len(records))]
            with (a.output/f'{directory.name}.log').open('w') as log:
                process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
                try:rc=process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    write_json(a.output/'retained_owner.json',dict(pid=process.pid,argv=argv,status='unknown_do_not_reopen'));return 4
            record=json.loads((directory/'worker.json').read_text()) if (directory/'worker.json').exists() else {}
            records.append(dict(case=case,phase=phase,returncode=rc,worker=record))
            write_json(a.output/'result.json',dict(status='running',phases=records))
            print(case,phase,rc,record.get('status'),flush=True)
            if rc or not record.get('safe_process_exit'):return 1
            if before is None:before=record['superblock_before']
            else:require(record['superblock_before']==before,'protected header changed')
    changed=[name for name,digest in source.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    write_json(a.output/'result.json',dict(status='invalid' if changed else 'pass',phases=records,changed_sources=changed))
    manifest(a.output);return int(bool(changed))

if __name__=='__main__':raise SystemExit(main())
