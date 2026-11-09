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
    p.add_argument('--library',type=Path,required=True);p.add_argument('--region-config',type=Path,required=True)
    p.add_argument('--expected-header-sha256');p.add_argument('--inspect',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    from npu_nvme.storage.bindings import load_backend
    from npu_nvme.storage.full_transport import FullTransport
    from npu_nvme.d2.backend import RegisteredBackend,registration_from_config,stage_config_from_config
    from npu_nvme.d2.format import Region,BLOCK
    os.environ['SPDK_SHM_ID']=str(a.shm_id)
    config=stage_config_from_config(a.region_config,required_purpose='validation')
    registration=registration_from_config(config,required_purpose='validation')
    transport=FullTransport(load_backend(a.library),pci=registration['pci_addr'],npu=-1,depth=4,chunk_size=4<<20,profiling_dir=a.out,role='host_owner')
    rows=[];report=dict(status='running',cases=rows,scope='real registered NVMe controlled flush failures/corruption; no power-loss claim',
        region_config=str(a.region_config),region_config_sha256=hashlib.sha256(a.region_config.read_bytes()).hexdigest(),
        library=str(a.library),library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest())
    def write():(a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        backend=RegisteredBackend(transport,registration,region_id=registration['region_id']);base=backend.base
        header=backend.read(base,3*BLOCK);digest=hashlib.sha256(header).hexdigest()
        (a.out/'header-before.bin').write_bytes(header);report['header_sha256']=digest
        if a.inspect:report['status']='pass';write();return
        if a.expected_header_sha256!=digest:raise ValueError('explicit scratch header digest required')
        backend.write(base,bytes(3*BLOCK));backend.flush()
        def fresh(point=None):
            def fault_hook(observed):
                if observed==point:raise OSError('injected '+observed)
            return Region(backend,offset=base,length=registration['length'],retention=3,fault_hook=fault_hook if point else None)
        region=fresh();region.format(region_id=config['region_id'])
        def commit(region,index):
            region.begin('fault-'+str(index),{'index':index})
            raw=bytes([index%251])*513;ref=region.payload(raw)
            return region.commit([dict(rank=0,name='x',shape=[513],dtype='uint8',partition='replicated',logical_offset=0,logical_bytes=513,payload=ref)],step=index,topology={'world_size':1},rank_controls=[dict(rank=0,step=index,controls={'cursor':index})])
        commit(region,1)
        for point,expect_new in (('after_payload_flush',False),('after_metadata_flush',False),('after_anchor_flush',True)):
            before=region.current['sequence']
            failing=fresh(point);failing.mount()
            try:commit(failing,10+before)
            except OSError:pass
            else:raise AssertionError('fault did not fire')
            region=fresh();region.mount()
            assert region.current['sequence']==before+int(expect_new)
            assert region.verify_payloads(region.current['state']['generations'][0])
            rows.append(dict(case=point,status='pass',selected=region.current['sequence']))
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
        try:mounted.begin('no-space',{},reserve_bytes=registration['length']*2)
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
