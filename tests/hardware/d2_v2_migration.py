#!/usr/bin/env python3
"""Migrate a strict V2 checkpoint to registered scratch; never write V2 source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'python')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--shm-id',type=int,required=True)
    p.add_argument('--step',type=int);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--region-config',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    os.environ['SPDK_SHM_ID']=str(a.shm_id)
    from npu_nvme.storage.bindings import load_backend
    from npu_nvme.storage.full_transport import FullTransport
    from npu_nvme.d2.backend import RegisteredBackend,ReadOnlyV2,registration_from_config,stage_config_from_config
    from npu_nvme.d2.format import Region
    from npu_nvme.d2.v2_migration import migrate
    config=stage_config_from_config(a.region_config,required_purpose='validation')
    registration=registration_from_config(config,required_purpose='validation')
    for pci,driver in ((registration['pci_addr'],'uio_pci_generic'),(registration['protected_pci_addr'],'nvme')):
        if (Path('/sys/bus/pci/devices')/pci/'driver').resolve().name!=driver:raise ValueError('binding differs')
    transport=FullTransport(load_backend(a.library),pci=registration['pci_addr'],npu=-1,depth=4,chunk_size=4<<20,profiling_dir=a.out,role='host_owner')
    report=dict(status='running',scope='strict D1 parser + real NVMe migration + fresh mount byte verification; no TP conversion',
        region_config=str(a.region_config),region_config_sha256=hashlib.sha256(a.region_config.read_bytes()).hexdigest(),
        library=str(a.library),library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest())
    def write():(a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        state=SimpleNamespace();transport.metadata.mount(state,transport.total_bytes,0)
        from npu_nvme.runtime.d1_schema import validate_record
        candidates=[]
        for key,record in state.meta_dict.get('checkpoints',{}).items():
            try:
                validate_record(record,state.layout)
            except (TypeError,ValueError,KeyError):
                continue
            if a.step is None or record['state_step']==a.step:candidates.append((record['state_step'],record['generation'],str(key),record))
        if not candidates:raise ValueError('no valid strict D1 FULL source record')
        _,_,source_key,record=max(candidates)
        report['source_checkpoint_key']=source_key;report['source_step']=record['state_step'];report['source_generation']=record['generation']
        header=transport.read(0,1<<20);(a.out/'source-header-before.bin').write_bytes(header)
        (a.out/'source-record.json').write_text(json.dumps(record,indent=2)+'\n')
        backend=RegisteredBackend(transport,registration,region_id=registration['region_id'])
        region=Region(backend,offset=backend.base,length=registration['length'],retention=3);region.mount()
        if region.header['region_id']!=config['region_id']:raise ValueError('scratch identity differs')
        def aligned_read(offset,length):
            start=offset//4096*4096;end=(offset+length+4095)//4096*4096
            raw=b''.join(transport.read(pos,min(4<<20,end-pos)) for pos in range(start,end,4<<20))
            return raw[offset-start:offset-start+length]
        source=ReadOnlyV2(aligned_read,transport.total_bytes)
        receipt=migrate(source=source,record=record,destination=region,request_id='v2-migration-'+a.out.name,
            identity={'source_manifest_sha256':record['manifest_sha256'],'spec':record['spec']},
            chunk_bytes=4<<20,source_layout=state.layout)
        fresh=Region(backend,offset=backend.base,length=config['length'],retention=3);fresh.mount()
        with fresh.selected(receipt['generation']) as selected:
            if not fresh.verify_payloads(selected):raise ValueError('migrated payload failed verification')
            descriptors=list(fresh.codec.read(fresh.backend,selected['pages'],fresh.data_base,fresh.end))
            names={row['name'] for row in descriptors}
            if names!=set(record['params']):raise ValueError('migrated state/control fields differ')
            for name,info in record['params'].items():
                digest=hashlib.sha256();cursor=0
                for row in sorted((r for r in descriptors if r['name']==name),key=lambda r:r['logical_offset']):
                    if row['logical_offset']!=cursor:raise ValueError('migrated tensor gap')
                    ref=row['payload'];raw=backend.read(ref['offset'],ref['length'])[:ref['logical_bytes']]
                    digest.update(raw);cursor+=len(raw)
                if cursor!=info['size'] or digest.hexdigest()!=info['sha256']:raise ValueError('migrated tensor differs from V2')
                # Re-read the source to prove no source extent was modified.
                original=hashlib.sha256()
                for offset in range(0,info['size'],4<<20):original.update(source.read(info['offset']+offset,min(4<<20,info['size']-offset)))
                if original.hexdigest()!=info['sha256']:raise ValueError('source tensor changed')
        if transport.read(0,1<<20)!=header:raise ValueError('V2 header changed')
        report.update(status='pass',receipt=receipt,source_unchanged=True,fields=len(record['params']),
            logical_bytes=sum(row['size'] for row in record['params'].values()))
    except BaseException as error:report.update(status='fail',error=repr(error));raise
    finally:
        try:transport.close(120)
        except BaseException as error:
            report.update(status='retained',error=repr(error));write();threading.Event().wait()
        write()
if __name__=='__main__':main()
