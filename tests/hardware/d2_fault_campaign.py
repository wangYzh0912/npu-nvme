#!/usr/bin/env python3
"""Registered scratch D2 fault/retention proof on actual NVMe, not power loss."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'python')]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--shm-id',type=int,required=True)
    p.add_argument('--expected-header-sha256');p.add_argument('--inspect',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    from npu_nvme.storage.bindings import load_backend
    from npu_nvme.storage.full_transport import FullTransport
    from npu_nvme.d2.backend import RegisteredBackend
    from npu_nvme.d2.format import Region,BLOCK
    os.environ['SPDK_SHM_ID']=str(a.shm_id)
    config=dict(schema_version=1,pci_addr='0000:83:00.0',protected_pci_addr='0000:84:00.0',region_id='d2-fault-1280g-128g',offset=1280<<30,length=128<<30,write_authorized=True,format='D2')
    transport=FullTransport(load_backend(),pci=config['pci_addr'],npu=-1,depth=4,chunk_size=4<<20,profiling_dir=a.out,role='host_owner')
    rows=[];report=dict(status='running',cases=rows,scope='real registered NVMe controlled flush failures/corruption; no power-loss claim')
    def write():(a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        backend=RegisteredBackend(transport,config,region_id=config['region_id']);base=backend.base
        header=backend.read(base,3*BLOCK);digest=hashlib.sha256(header).hexdigest()
        (a.out/'header-before.bin').write_bytes(header);report['header_sha256']=digest
        if a.inspect:report['status']='pass';write();return
        if a.expected_header_sha256!=digest:raise ValueError('explicit scratch header digest required')
        backend.write(base,bytes(3*BLOCK));backend.flush()
        class Faults:
            remaining=None
            def read(self,o,n):return backend.read(o,n)
            def write(self,o,data):backend.write(o,data)
            def flush(self):
                if self.remaining is not None:
                    self.remaining-=1
                    if self.remaining==0:raise OSError('injected pre-flush failure')
                backend.flush()
        fault=Faults()
        def fresh():return Region(fault,offset=base,length=config['length'],retention=3)
        region=fresh();region.format(region_id=config['region_id'])
        def commit(region,index):
            region.begin('fault-'+str(index),{'index':index})
            raw=bytes([index%251])*513;ref=region.payload(raw)
            return region.commit([dict(rank=0,name='x',shape=[513],dtype='uint8',partition='replicated',logical_offset=0,logical_bytes=513,payload=ref)],step=index,topology={'world_size':1},rank_controls=[dict(rank=0,step=index,controls={'cursor':index})])
        commit(region,1)
        for stage in (1,2,3):
            before=region.current['sequence'];fault.remaining=stage
            try:commit(region,10+stage)
            except OSError:pass
            else:raise AssertionError('fault did not fire')
            fault.remaining=None;region=fresh();region.mount()
            # Failed flush can still reach media; either complete generation is legal.
            assert region.current['sequence'] in (before,before+1)
            assert region.verify_payloads(region.current['state']['generations'][0])
            rows.append(dict(case='flush-'+str(stage),status='pass',selected=region.current['sequence']))
        receipt=commit(region,30)
        assert region.begin('fault-30',{'index':30},retry=True)['generation']==receipt['generation']
        try:region.begin('expired',{},retry=True)
        except KeyError:pass
        else:raise AssertionError('expired request not unknown')
        rows.append(dict(case='retry-and-expired',status='pass'))
        with region.selected(receipt['generation']) as pinned:
            for index in range(31,39):commit(region,index)
            assert region.verify_payloads(pinned)
        rows.append(dict(case='reader-pin-retention',status='pass'))
        old=backend.read(base+(region.current['slot']+1)*BLOCK,BLOCK)
        corrupted=bytearray(old);corrupted[5]^=1
        backend.write(base+(region.current['slot']+1)*BLOCK,corrupted);backend.flush()
        mounted=fresh();errors=mounted.mount();assert errors and mounted.verify_payloads(mounted.current['state']['generations'][0])
        rows.append(dict(case='anchor-fallback',status='pass'))
        try:mounted.begin('no-space',{},reserve_bytes=config['length']*2)
        except BufferError:pass
        else:raise AssertionError('no-space admitted')
        rows.append(dict(case='no-space',status='pass'));report['status']='pass'
    except BaseException as error:report.update(status='fail',error=repr(error));raise
    finally:
        try:transport.close(120)
        except BaseException as error:
            report.update(status='retained',error=repr(error));write();threading.Event().wait()
        write()
if __name__=='__main__':main()
