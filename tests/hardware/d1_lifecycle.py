#!/usr/bin/env python3
"""D1 lifecycle extensions on HW1; injected failures are labeled explicitly."""
import argparse
import ctypes
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from stage4_fault_lifecycle import Worker, require, resource_snapshot, manifest, write_json, ROOT
from experiment_evidence import sha256_file

CASES=['repeat','submit-close','lost-completion','query-quarantine','record-quarantine']


def compact_resource():
    raw=resource_snapshot()
    def number(text,key):
        return int(next(line.split()[1] for line in text.splitlines() if line.startswith(key+':')))
    return dict(fd_count=raw['fd_count'],threads=number(raw['process_status'],'Threads'),
                rss_kb=number(raw['process_status'],'VmRSS'),free_hugepages=number(raw['meminfo'],'HugePages_Free'))


def execute(args):
    for name in list(os.environ):
        if name.startswith('NPU_NVME_TEST_'): os.environ.pop(name)
    os.environ['SPDK_SHM_ID']=str(args.shm_id)
    record=dict(status='fail',case=args.case,pid=os.getpid(),scope='real hardware with controlled software fault injection')
    worker=None
    try:
        if args.case=='repeat':
            records=[];resources=[]
            for i in range(34):  # two allocator warmups, then 32 measured cycles
                item={}; worker=Worker(args,item); worker.open(); worker.verify(); worker.close()
                records.append(item); resources.append(compact_resource())
                write_json(args.output/'cycles.json',dict(cycles=records,resources=resources))
            steady=resources[2:]
            require(len(steady)==32,'missing reopen cycles')
            require(max(r['fd_count'] for r in steady)<=steady[0]['fd_count'], 'FD count grows after warmup')
            require(max(r['threads'] for r in steady)<=steady[0]['threads'], 'thread count grows after warmup')
            require(min(r['free_hugepages'] for r in steady)>=steady[0]['free_hugepages'], 'hugepage allocation grows after warmup')
            record.update(cycles=32,warmup_cycles=2,resources=resources,safe_process_exit=True)
        else:
            worker=Worker(args,record); worker.open()
            record['superblock_before']=worker.superblock()
            if args.case=='verify':
                worker.verify(); worker.close(); record['safe_process_exit']=True
            elif args.case=='submit-close':
                source=worker.device_source(); barrier=threading.Barrier(3); results=[]
                os.environ['NPU_NVME_TEST_NVME_SUBMIT_DELAY_MS']='100'
                require(worker.submit(source)[0]==0,'initial request was not admitted')
                start=time.monotonic()
                while not worker.snapshot()['stats']['dma_inflight']:
                    require(time.monotonic()-start<2,'initial DMA did not start')
                    time.sleep(.001)
                def submitter():
                    barrier.wait()
                    for _ in range(16): results.append(worker.submit(source)[0])
                threads=[threading.Thread(target=submitter) for _ in range(2)]
                for t in threads:t.start()
                barrier.wait()
                close_rc=worker.lib.npu_nvme_close(worker.ctx,50)
                for t in threads:t.join(5)
                record.update(submit_results=results,race_close_rc=close_rc)
                require(all(not t.is_alive() for t in threads),'submitter did not finish')
                require(all(rc in (0,-errno.EBUSY,-errno.ESHUTDOWN) for rc in results),'unexpected admission result')
                require(close_rc in (0,-errno.ETIMEDOUT),'unexpected close result')
                worker.close(); record.update(submit_results=results,race_close_rc=close_rc,safe_process_exit=True)
            elif args.case=='lost-completion':
                os.environ['NPU_NVME_TEST_HOLD_NVME_COMPLETIONS']='1'
                rc,req=worker.submit(ctypes.addressof(worker.buffer()),host=True)
                require(rc==0,'request not admitted')
                start=time.monotonic(); rc=worker.lib.npu_nvme_wait_request(req,50)
                snap=worker.snapshot()
                require(rc==-errno.ETIMEDOUT and time.monotonic()-start<0.2,'observation did not time out within bound')
                require(snap['stats']['nvme_outstanding']>0 and snap['retained_slots'],'in-flight request lost its leases')
                require(worker.submit(ctypes.addressof(worker.buffer()),host=True)[0]!=0,'timeout did not close admission')
                record['held_completion']=snap
                os.environ.pop('NPU_NVME_TEST_HOLD_NVME_COMPLETIONS')
                require(worker.lib.npu_nvme_wait_request(req,5000)==0,'late completion did not drain')
                worker.close(); record.update(safe_process_exit=True,completion_mode='withheld then delivered; not a physical lost interrupt')
            else:
                names=['NPU_NVME_TEST_FAIL_STREAM_SYNC']
                names += ['NPU_NVME_TEST_FAIL_EVENT_RECORD'] if args.case=='record-quarantine' else ['NPU_NVME_TEST_FAIL_EVENT_QUERY','NPU_NVME_TEST_FAIL_EVENT_SYNC']
                for name in names:os.environ[name]='1'
                rc,req=worker.submit(worker.device_source())
                require(rc==0,'request not admitted')
                require(worker.lib.npu_nvme_wait_request(req,50)==-errno.ETIMEDOUT,'quarantined request was reported terminal')
                require(worker.lib.npu_nvme_close(worker.ctx,50)==-errno.EIO,'unsafe close was reported successful')
                snap=worker.snapshot()
                require(snap['retained_slots'] and snap['stats']['dma_inflight']>0,'quarantine did not retain DMA ownership')
                require(worker.submit(worker.device_source())[0]!=0,'quarantine did not reject new work')
                record['quarantine']=snap
                # Independent, non-injected driver call proves the actual NPU
                # transfers are stopped before the test process may exit. The
                # production context remains quarantined; no reset/free occurs.
                worker.acl.aclrtSynchronizeDevice.argtypes=[]
                worker.acl.aclrtSynchronizeDevice.restype=ctypes.c_int
                stop_rc=worker.acl.aclrtSynchronizeDevice()
                require(stop_rc==0 and snap['stats']['nvme_outstanding']==0,'test cannot prove physical DMA stop')
                record.update(driver_stop_rc=stop_rc,production_cleanup_performed=False,
                              safe_process_exit=True,injected_flags=names)
        record['status']='pass'
    except BaseException:
        record['traceback']=traceback.format_exc()
        if worker is None or not worker.ctx:
            record['safe_process_exit']=True
        else:
            try:
                os.environ.pop('NPU_NVME_TEST_HOLD_NVME_COMPLETIONS',None)
                rc=worker.lib.npu_nvme_close(worker.ctx,5000)
                if rc==0:
                    worker.close(); record['safe_process_exit']=True
                else:
                    stop=worker.acl.aclrtSynchronizeDevice()
                    record['safe_process_exit']=(rc==-errno.EIO and stop==0 and
                        worker.snapshot()['stats']['nvme_outstanding']==0)
            except BaseException:
                record['safe_process_exit']=False
    write_json(args.output/'worker.json',record)
    sys.stdout.flush();sys.stderr.flush()
    if not record.get('safe_process_exit'):
        # The parent records this PID and stops the campaign without killing
        # the owner. No device reopen is allowed without a stop proof.
        threading.Event().wait()
    # In the quarantine cases even Python finalizers must not release the
    # retained target. Success requires the independent driver proof above.
    os._exit(0 if record['status']=='pass' else 1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=CASES+['verify'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shm-id',type=int,default=41100)
    p.add_argument('--pci',default='0000:83:00.0');p.add_argument('--npu',type=int,default=7)
    args=p.parse_args();args.output=args.output.resolve()
    args.depth=2;args.close_timeout_ms=5000;args.library=ROOT/'build_out/lib/libnpu_nvme.so'
    require(os.geteuid()==0 and args.pci=='0000:83:00.0','requires root and authorized 83 namespace')
    require(Path('/sys/bus/pci/devices/0000:84:00.0/driver').resolve().name=='nvme','protected device changed')
    require(Path('/sys/bus/pci/devices/0000:83:00.0/driver').resolve().name=='uio_pci_generic','raw device driver changed')
    if args.case: execute(args)
    args.output.mkdir(parents=True,exist_ok=False)
    source={str(p.relative_to(ROOT)):sha256_file(p) for directory in ('src','include','python','tests')
            for p in sorted((ROOT/directory).rglob('*')) if p.is_file() and p.suffix in ('.c','.h','.py')}
    write_json(args.output/'sources.json',source)
    results=[]
    for case in CASES:
        for phase in (case,'verify'):
            directory=args.output/f'{case}-{phase}';directory.mkdir()
            argv=[sys.executable,str(Path(__file__).resolve()),'--case',phase,'--output',str(directory),
                  '--shm-id',str(args.shm_id+len(results))]
            print(f'D1 lifecycle {case} {phase}',flush=True)
            with (directory/'output.log').open('w') as log:
                process=subprocess.Popen(argv,cwd=directory,stdout=log,stderr=subprocess.STDOUT)
                try: process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    write_json(args.output/'result.json',dict(status='fail',reason='owner retained after timeout',pid=process.pid,argv=argv))
                    return 1
            worker=json.loads((directory/'worker.json').read_text()) if (directory/'worker.json').exists() else {}
            results.append(dict(case=case,phase=phase,argv=argv,returncode=process.returncode,worker=worker))
            write_json(args.output/'phases.json',results)
            if process.returncode or worker.get('status')!='pass' or not worker.get('safe_process_exit'):
                write_json(args.output/'result.json',dict(status='fail',phases=results));manifest(args.output);return 1
    changed=[name for name,digest in source.items() if sha256_file(ROOT/name)!=digest]
    write_json(args.output/'result.json',dict(status='pass' if not changed else 'invalid',phases=results,
               changed_sources=changed,library_sha256=sha256_file(args.library)))
    manifest(args.output)
    return int(bool(changed))


if __name__=='__main__':sys.exit(main())
