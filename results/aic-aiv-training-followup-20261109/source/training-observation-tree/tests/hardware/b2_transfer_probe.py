#!/usr/bin/env python3
"""B2 real transfer probe. Unknown device progress retains the owning process."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import zlib

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python')]
from npu_nvme.storage.bindings import (load_backend, NPUNVMEContext, NPUNVMERequest,
    NPUNVMETransferSpec, NPUNVMETransferItem, NPUNVMETransferReceipt,
    NPUNVMETransferDigest, NPUNVMEStats, NPUNVMECapabilities, NPUNVMECopySpec)


from npu_nvme.storage.requests import transfer_wait

def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n'); temporary.replace(path)


def check(ok, message):
    if not ok: raise RuntimeError(message)


def resources():
    return dict(process_status=Path('/proc/self/status').read_text(),
                meminfo=Path('/proc/meminfo').read_text(),
                fd_count=len(list(Path('/proc/self/fd').iterdir())))


def loaded_libraries():
    paths=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts=line.split()
        if len(parts)>=6 and parts[-1].startswith('/') and '.so' in parts[-1]:
            paths.add(parts[-1])
    return {path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in sorted(paths) if Path(path).is_file()}


def worker(a):
    record = dict(status='fail',pid=os.getpid(),kind=a.kind,chunk=a.chunk,depth=a.depth,
                  offset=a.offset,scope='real HW1 SPDK allocation shared with ACL',read_direction=a.read)
    record['resources_before']=resources()
    ctx = C.POINTER(NPUNVMEContext)(); req = C.POINTER(NPUNVMERequest)()
    device = C.c_void_p(); target_device = C.c_void_p(); backend = None; safe = True
    try:
        os.environ['SPDK_SHM_ID'] = str(a.shm_id)
        backend = load_backend(a.library); lib, acl = backend.lib, backend.acl_lib
        check(acl.aclrtSetDevice(a.npu)==0,'set device')
        check(lib.npu_nvme_init(C.byref(ctx),a.pci.encode(),a.npu,a.depth,a.chunk,
                               True,str(a.out).encode())==0,'open')
        safe = False
        caps=NPUNVMECapabilities()
        check(lib.npu_nvme_get_capabilities(ctx,C.byref(caps),C.sizeof(caps))==0,"capabilities")
        record["capabilities"]={name:getattr(caps,name) for name,_ in caps._fields_}
        check(caps.pipe_depth==a.depth and caps.chunk_size==a.chunk,"effective configuration")
        size = a.chunk * max(8,a.depth+1) + 517  # exceed even the deepest pool
        check(lib.npu_nvme_get_total_blocks(ctx)>=a.offset+size+4096,'scratch capacity')
        pattern = bytes((i * 131 + 17) & 255 for i in range(4096))
        raw = (pattern * ((size+4095)//4096))[:size]
        host = C.create_string_buffer(raw, size)
        pointer = C.addressof(host)
        if a.kind == 'hbm':
            check(acl.aclrtMalloc(C.byref(device),size,0)==0,'HBM allocate')
            check(acl.aclrtMemcpy(device,size,host,size,1)==0,'source setup')
            pointer = device.value
        parts = [(i,min(a.chunk,size-i)) for i in range(0,size,a.chunk)]
        items = (NPUNVMETransferItem * len(parts))()
        for item,(offset,length) in zip(items,parts):
            item.address=pointer+offset;item.offset=a.offset+offset;item.length=length
        spec=NPUNVMETransferSpec(C.sizeof(NPUNVMETransferSpec),1,0,int(a.kind=='host'),3,len(parts),items)
        started=time.monotonic()
        check(lib.npu_nvme_submit_transfer(ctx,C.byref(spec),C.byref(req))==0,'submit')
        record['api_stall_seconds']=time.monotonic()-started
        check(lib.npu_nvme_wait_request(req,120000)==0,'wait')
        record['transfer_seconds']=time.monotonic()-started
        receipt=NPUNVMETransferReceipt(); digests=(NPUNVMETransferDigest*len(parts))()
        check(lib.npu_nvme_get_transfer_receipt(req,C.byref(receipt),C.sizeof(receipt))==0,'receipt')
        check(receipt.logical_bytes==size and receipt.transport_safe and not receipt.data_durable,'receipt semantics')
        check(lib.npu_nvme_get_transfer_digests(req,digests,len(parts))==0,'digests')
        for result,(offset,length) in zip(digests,parts):
            payload=raw[offset:offset+length]
            check(result.crc32==zlib.crc32(payload),'CRC differs')
            check(bytes(result.sha256)==hashlib.sha256(payload).digest(),'SHA differs')
        lib.npu_nvme_release_request(req);req=C.POINTER(NPUNVMERequest)()
        flush_spec=NPUNVMETransferSpec(C.sizeof(NPUNVMETransferSpec),1,4,1,0,0,None)
        check(lib.npu_nvme_submit_transfer(ctx,C.byref(flush_spec),C.byref(req))==0,'flush submit')
        check(lib.npu_nvme_wait_request(req,120000)==0,'flush wait')
        check(lib.npu_nvme_get_transfer_receipt(req,C.byref(receipt),C.sizeof(receipt))==0 and receipt.data_durable,'flush receipt')
        lib.npu_nvme_release_request(req);req=C.POINTER(NPUNVMERequest)()
        # Dedicated scratch block after the payload; no format/header writes.
        meta_offset=a.offset+((size+4095)//4096)*4096
        meta_source=C.create_string_buffer(pattern,4096);meta_target=C.create_string_buffer(4096)
        meta_item=NPUNVMETransferItem(C.addressof(meta_source),meta_offset,4096)
        meta_spec=NPUNVMETransferSpec(C.sizeof(NPUNVMETransferSpec),1,3,1,0,1,C.pointer(meta_item))
        for operation,buffer in [(3,meta_source),(2,meta_target)]:
            meta_spec.operation=operation;meta_item.address=C.addressof(buffer)
            check(lib.npu_nvme_submit_transfer(ctx,C.byref(meta_spec),C.byref(req))==0,'metadata submit')
            check(lib.npu_nvme_wait_request(req,120000)==0,'metadata wait')
            check(lib.npu_nvme_get_transfer_receipt(req,C.byref(receipt),C.sizeof(receipt))==0,'metadata receipt')
            check(receipt.operation==operation and receipt.logical_bytes==4096 and receipt.item_count==1,'metadata identity')
            lib.npu_nvme_release_request(req);req=C.POINTER(NPUNVMERequest)()
        check(meta_target.raw==pattern,'metadata readback')
        allocation=(size+4095)//4096*4096
        target=C.create_string_buffer(allocation)
        ptrs=(C.c_void_p*len(parts))(*(C.addressof(target)+i for i,n in parts))
        offsets=(C.c_uint64*len(parts))(*(a.offset+i for i,n in parts))
        sizes=(C.c_size_t*len(parts))(*(n for i,n in parts))
        if a.read:
            target_pointer=C.addressof(target)
            if a.kind=='hbm':
                check(acl.aclrtMalloc(C.byref(target_device),size,0)==0,'HBM target allocation')
                target_pointer=target_device.value
            for item,(offset,length),digest in zip(items,parts,digests):
                item.address=target_pointer+offset
                item.expected_crc32=digest.crc32
                item.expected_sha256[:]=bytes(digest.sha256)
            spec.operation=1
            started=time.monotonic()
            check(lib.npu_nvme_submit_transfer(ctx,C.byref(spec),C.byref(req))==0,'read submit')
            check(lib.npu_nvme_wait_request(req,120000)==0,'read wait')
            record['read_transfer_seconds']=time.monotonic()-started
            check(lib.npu_nvme_get_transfer_receipt(req,C.byref(receipt),C.sizeof(receipt))==0,'read receipt')
            check(receipt.operation==1 and receipt.transport_safe,'read completion')
            lib.npu_nvme_release_request(req);req=C.POINTER(NPUNVMERequest)()
            if a.kind=='hbm':check(acl.aclrtMemcpy(target,size,target_device,size,2)==0,'target oracle')
        else:
            check(transfer_wait(lib,1,1,ctx,ptrs,offsets,sizes,len(parts))==0,'readback')
        check(target.raw[:size]==raw,'actual bytes differ')
        if a.copy and a.kind=='hbm':
            # Bounded capture/restore copies use the same shared pool, no NVMe.
            copied=C.create_string_buffer(size)
            for offset,length in parts:
                copy_spec=NPUNVMECopySpec(C.sizeof(NPUNVMECopySpec),1,5,0,
                    device.value+offset,C.addressof(copied)+offset,length)
                check(lib.npu_nvme_submit_copy(ctx,C.byref(copy_spec),C.byref(req))==0,'D2H copy submit')
                check(lib.npu_nvme_wait_request(req,120000)==0,'D2H copy wait')
                lib.npu_nvme_release_request(req);req=C.POINTER(NPUNVMERequest)()
            check(copied.raw==raw,'copy-only D2H differs')
            for offset,length in parts:
                copy_spec=NPUNVMECopySpec(C.sizeof(NPUNVMECopySpec),1,6,0,
                    C.addressof(copied)+offset,target_device.value+offset,length)
                check(lib.npu_nvme_submit_copy(ctx,C.byref(copy_spec),C.byref(req))==0,'H2D copy submit')
                check(lib.npu_nvme_wait_request(req,120000)==0,'H2D copy wait')
                lib.npu_nvme_release_request(req);req=C.POINTER(NPUNVMERequest)()
            check(acl.aclrtMemcpy(target,size,target_device,size,2)==0,'copy target oracle')
            check(target.raw[:size]==raw,'copy-only H2D differs')
            record['copy_only_byte_exact']=True
        stats=NPUNVMEStats();check(lib.npu_nvme_get_stats(ctx,C.byref(stats))==0,'stats')
        counters={k:int(getattr(stats,k)) for k,_ in stats._fields_}
        if a.kind=='hbm':check(counters['async_dma_submit_count']==len(parts)*((2 if a.read else 1)+(2 if a.copy else 0)),'async D2H path absent')
        check(not counters['stream_sync_fallback_count'] and not counters['async_event_query_error_count'],'unexpected synchronization recovery')
        record['loaded_libraries']=loaded_libraries()
        record['python']=sys.executable
        record['resources_after_transfer']=resources()
        record.update(status='pass',bytes=size,byte_exact=True,checksums_exact=True,stats=counters,
                      buffer_path='HBM -> SPDK_MALLOC_DMA -> NVMe' if a.kind=='hbm' else 'Host -> SPDK_MALLOC_DMA -> NVMe',
                      library_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest())
    except BaseException:
        record['error']=traceback.format_exc()
    finally:
        if ctx and backend:
            rc=backend.lib.npu_nvme_close(ctx,120000);record['close_rc']=rc
            safe=rc==0
            if safe:
                if req:backend.lib.npu_nvme_release_request(req)
                backend.lib.npu_nvme_cleanup(ctx)
        if safe and target_device and backend:
            rc=backend.acl_lib.aclrtFree(target_device);record['target_free_rc']=rc
            if rc:record['status']='fail'
        if safe and device and backend:
            rc=backend.acl_lib.aclrtFree(device);record['free_rc']=rc
            if rc:record['status']='fail'
        record['resources_after_close']=resources()
        record['safe_process_exit']=safe
        write(a.out/'result.json',record)
        if not safe:
            record['status']='retained';write(a.out/'result.json',record)
            sys.stdout.flush();sys.stderr.flush();threading.Event().wait()
    return 0 if record['status']=='pass' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--copy',action='store_true')
    p.add_argument('--high-depth',action='store_true')
    p.add_argument('--read',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--library',default=str(ROOT/'build/libnpu_nvme.so'))
    p.add_argument('--pci',default='0000:83:00.0');p.add_argument('--npu',type=int,default=7)
    p.add_argument('--offset',type=int,default=64*1024**3)
    p.add_argument('--kind',choices=['hbm','host'],default='hbm')
    p.add_argument('--chunk',type=int,default=4*1024**2);p.add_argument('--depth',type=int,default=4)
    p.add_argument('--shm-id',type=int,default=61500)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    check(a.pci=='0000:83:00.0' and a.npu==7,'authorized HW1 identity')
    check(a.offset==64*1024**3 and 0<a.chunk<=16*1024**2 and a.depth in (1,4,8,16,32,64),'probe extent/config')
    check(not a.copy or a.read,'copy validation requires --read')
    if a.worker:return worker(a)
    write(a.out/'source.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         source_files={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest()
          for top in ['src','include','python','tests'] for q in (ROOT/top).rglob('*')
          if q.is_file() and q.suffix in ('.c','.h','.py')}))
    runs=[]
    for kind in ['hbm','host']:
        for chunk in [1,4,16]:
            for depth in ([1,4,8,16,32,64] if a.high_depth else [1,4,8,16]):
                directory=a.out/f'{kind}-c{chunk}-d{depth}'
                cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--out',str(directory),
                     '--library',a.library,'--kind',kind,'--chunk',str(chunk*1024**2),'--depth',str(depth),
                     '--shm-id',str(a.shm_id+len(runs))]
                if a.read:cmd.append('--read')
                if a.copy:cmd.append('--copy')
                with (a.out/f'{directory.name}.log').open('w') as log:
                    process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    try:code=process.wait(timeout=300)
                    except subprocess.TimeoutExpired:
                        write(a.out/'retained_owner.json',dict(pid=process.pid,command=cmd,status='unknown_do_not_reopen'))
                        return 4
                result=json.loads((directory/'result.json').read_text()) if (directory/'result.json').exists() else {}
                runs.append(dict(path=str(directory),returncode=code,result=result));write(a.out/'result.json',dict(runs=runs,status='running'))
                print(directory.name,code,result.get('status'),flush=True)
                if code:return code
    write(a.out/'result.json',dict(status='pass',runs=runs));return 0

if __name__=='__main__':raise SystemExit(main())
