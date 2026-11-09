#!/usr/bin/env python3
"""Real NVMe R0 FULL/delta persistence fixture; Host arrays, not training proof."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'python'))
import numpy as np
from incremental_manifest import build_training_state_manifest
from r0_session import state_digest
from npu_nvme.storage.full_transport import FullTransport
from npu_nvme.storage.bindings import load_backend
from npu_nvme.d2.backend import RegisteredBackend
from npu_nvme.d2.format import Region,SHARED_PAYLOAD,BLOCK
from npu_nvme.d2.incremental import PersistentR0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['inspect','initialize','source','restore'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--shm-id',type=int,required=True);p.add_argument('--expected-header-sha256')
    p.add_argument('--source-run',type=Path);p.add_argument('--inject-flush',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'config/f1_region.json').read_text());protected=config.pop('protected_qwen_region')
    config.pop('authorization_source');config.pop('required_feature')
    if (config['offset'],config['length'])!=(1536<<30,128<<30):raise ValueError('F1 extent changed')
    if config['offset']<protected['offset']+protected['length']:raise ValueError('F1 overlaps Qwen D2')
    for pci,driver in ((config['pci_addr'],'uio_pci_generic'),(config['protected_pci_addr'],'nvme')):
        if (Path('/sys/bus/pci/devices')/pci/'driver').resolve().name!=driver:raise ValueError('binding differs')
    report=dict(status='running',scope='real raw NVMe, Host-array fixture, no training acceptance',operation=a.operation,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest())
    def write():(a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    write();transport=None
    try:
        os.environ['SPDK_SHM_ID']=str(a.shm_id)
        transport=FullTransport(load_backend(a.library),pci=config['pci_addr'],npu=-1,depth=4,chunk_size=4<<20,profiling_dir=a.out,role='host_owner')
        backend=RegisteredBackend(transport,config,region_id=config['region_id'])
        region=Region(backend,offset=config['offset'],length=config['length'],retention=3)
        header=backend.read(region.base,3*BLOCK);(a.out/'header-before.bin').write_bytes(header)
        report['header_sha256']=hashlib.sha256(header).hexdigest()
        if a.operation=='initialize':
            if a.expected_header_sha256!=report['header_sha256']:raise ValueError('header inspection digest required')
            if header[:8]==b'NPUNVM3\0':raise ValueError('refusing existing D2 reformat')
            backend.write(region.base,bytes(3*BLOCK));backend.flush()
            region.format(region_id=config['region_id'],features=[SHARED_PAYLOAD])
        elif a.operation!='inspect':
            report['mount_errors']=region.mount()
            if region.header['region_id']!=config['region_id']:raise ValueError('F1 mounted region identity differs')
            class P:shape=(262144,);dtype=np.float32
            class M:
                def parameters_and_names(self):return [('weight',P()),('moment',P())]
            manifest=build_training_state_manifest({'model':M()},block_size=4096,small_threshold=0)
            store=PersistentR0(region,manifest,chunk_bytes=1<<20,max_chain_length=8)
            if a.operation=='source':
                state={f.canonical_name:np.zeros(f.shape,dtype=f.dtype) for f in manifest.fields}
                receipts=[]
                for step in range(1,4):
                    for array in state.values():array[step]=step if step!=2 else -0.0
                    receipts.append(store.save(state,dict(cursor=step),step=step))
                report.update(receipts=receipts,state_digest=state_digest(state),controls=dict(cursor=3))
                if a.inject_flush:
                    before=state_digest(store.ledger.persisted)
                    for array in state.values():array[10]=999
                    os.environ['NPU_NVME_TEST_FAIL_FLUSH']='1'
                    try:store.save(state,dict(cursor=4),step=4)
                    except BaseException as error:
                        report['injected_failure']=repr(error)
                        assert store.failed and state_digest(store.ledger.persisted)==before
                    else:raise AssertionError('injected flush unexpectedly succeeded')
                    finally:os.environ.pop('NPU_NVME_TEST_FAIL_FLUSH',None)
            else:
                if not a.source_run:raise ValueError('restore requires source evidence')
                source=json.loads((a.source_run/'result.json').read_text())
                result=store.recover(source['receipts'][-1]['generation'])
                assert state_digest(result['state'])==source['state_digest'] and result['controls']==source['controls']
                report.update(state_digest=state_digest(result['state']),generation=result['generation'])
        report['status']='pass'
    except BaseException as error:report.update(status='fail',error=repr(error));raise
    finally:
        if transport:
            try:transport.close(120);report['closed']=True
            except BaseException as error:report.update(status='retained',closed=False,error=repr(error));write();threading.Event().wait()
        write()
if __name__=='__main__':main()
