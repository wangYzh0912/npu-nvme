#!/usr/bin/env python3
"""Explicit strict D1 source -> registered D2 scratch, or independent verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'python')]


def digest_file(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--shm-id',type=int,required=True)
    p.add_argument('--step',type=int);p.add_argument('--library',type=Path,required=True)
    p.add_argument('--region-config',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--source-image',type=Path);group.add_argument('--source-raw',action='store_true')
    p.add_argument('--verify-generation',type=int);p.add_argument('--request-id')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    from npu_nvme.storage.bindings import load_backend
    from npu_nvme.storage.full_transport import FullTransport
    from npu_nvme.storage.d1_image import D1Image,select_record
    from npu_nvme.storage.metadata import mount_readonly
    from npu_nvme.d2.backend import RegisteredBackend,ReadOnlyV2,registration_from_config
    from npu_nvme.d2.format import Region
    from npu_nvme.d2.v2_migration import migrate
    report=dict(status='running',closed=False,scope='strict D1 offline migration; no historical Qwen/TP conversion claim')
    def write():(a.out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    write();transport=None;source=None
    try:
        registration=registration_from_config(a.region_config,required_purpose='validation')
        report.update(library_sha256=digest_file(a.library),region_config_sha256=digest_file(a.region_config))
        for pci,driver in ((registration['pci_addr'],'uio_pci_generic'),(registration['protected_pci_addr'],'nvme')):
            if (Path('/sys/bus/pci/devices')/pci/'driver').resolve().name!=driver:raise ValueError('binding differs')
        if a.source_image:
            before=digest_file(a.source_image);source=D1Image(a.source_image);state=source.state
            source_key,record=source.select(a.step)
        os.environ['SPDK_SHM_ID']=str(a.shm_id)
        transport=FullTransport(load_backend(a.library),pci=registration['pci_addr'],npu=-1,depth=4,chunk_size=4<<20,profiling_dir=a.out,role='host_owner')
        if a.source_raw:
            def aligned_read(offset,length):
                start=offset//4096*4096;end=(offset+length+4095)//4096*4096
                raw=b''.join(transport.read(pos,min(4<<20,end-pos)) for pos in range(start,end,4<<20))
                return raw[offset-start:offset-start+length]
            source=ReadOnlyV2(aligned_read,transport.total_bytes)
            state=mount_readonly(source.read,source.total_bytes);source_key,record=select_record(state,a.step)
            before=hashlib.sha256(source.read(0,1<<20)).hexdigest()
        report.update(source_checkpoint_key=source_key,source_step=record['state_step'],source_generation=record['generation'],source_before=before)
        (a.out/'source-record.json').write_text(json.dumps(record,indent=2)+'\n')
        backend=RegisteredBackend(transport,registration,region_id=registration['region_id'])
        region=Region(backend,offset=backend.base,length=registration['length'],retention=3);region.mount()
        if region.header['region_id']!=registration['region_id']:raise ValueError('scratch identity differs')
        if a.verify_generation is None:
            receipt=migrate(source=source,record=record,destination=region,request_id=a.request_id or 'migration-'+a.out.name,
                identity={'source_manifest_sha256':record['manifest_sha256'],'spec':record['spec']},
                chunk_bytes=4<<20,source_layout=state.layout)
            report['receipt']=receipt;generation=receipt['generation']
        else:generation=a.verify_generation
        fresh=Region(backend,offset=backend.base,length=registration['length'],retention=3);fresh.mount()
        with fresh.selected(generation) as selected:
            fresh.verify_payloads(selected)
            rows=list(fresh.codec.read(backend,selected['pages'],fresh.data_base,fresh.end))
            if {r['name'] for r in rows}!=set(record['params']) or selected['step']!=record['state_step']:raise ValueError('migrated state differs')
            for name,info in record['params'].items():
                digest=hashlib.sha256();cursor=0
                for row in sorted((r for r in rows if r['name']==name),key=lambda r:r['logical_offset']):
                    if row['logical_offset']!=cursor or row['shape']!=info['shape'] or row['dtype']!=info['dtype']:raise ValueError('migrated schema differs')
                    ref=row['payload'];raw=backend.read(ref['offset'],ref['length'])[:ref['logical_bytes']]
                    original=source.read(info['offset']+cursor,len(raw))
                    if raw!=original:raise ValueError('source/destination bytes differ')
                    digest.update(raw);cursor+=len(raw)
                if cursor!=info['size'] or digest.hexdigest()!=info['sha256']:raise ValueError('migrated tensor/control differs')
        after=digest_file(a.source_image) if a.source_image else hashlib.sha256(source.read(0,1<<20)).hexdigest()
        if before!=after:raise ValueError('source changed')
        report.update(status='pass',source_unchanged=True,fields=len(record['params']),generation=generation,
            verification_only=a.verify_generation is not None,logical_bytes=sum(v['size'] for v in record['params'].values()))
    except BaseException as error:report.update(status='fail',error=repr(error));raise
    finally:
        if isinstance(source,D1Image):source.close()
        if transport:
            try:transport.close(120);report['closed']=True
            except BaseException as error:
                report.update(status='retained',error=repr(error));write();threading.Event().wait()
        write()
if __name__=='__main__':main()
