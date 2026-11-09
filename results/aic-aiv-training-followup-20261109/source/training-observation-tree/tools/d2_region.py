#!/usr/bin/env python3
"""Explicit D2 format/inspection and fresh-process persistence probe.

This offline single-card format harness uses the existing combined transport;
it is not the future Host-only TP4 owner. It never initializes on dry-run and
never formats as a consequence of mount failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from npu_nvme.d2.backend import RegisteredBackend, registration_from_config
from npu_nvme.d2.format import Region, BLOCK
from npu_nvme.storage.bindings import load_backend
from npu_nvme.storage.full_transport import FullTransport


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['preflight','format','initialize','inspect','verify','probe-save'])
    parser.add_argument('--config',type=Path,default=ROOT/'config/d2_qwen_region.json')
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--npu',type=int,default=7,help='explicit offline format harness NPU, not TP4 owner')
    parser.add_argument('--shm-id',type=int,required=True)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--expected-header-sha256',help='explicit initialize only; compare saved 12 KiB header before clearing')
    args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text());registration=registration_from_config(config)
    if (registration['offset'],registration['length'])!=(256<<30,1<<40):
        raise ValueError('formal D2 extent differs from authorized [256,1280) GiB')
    report=dict(operation=args.operation,registration=registration,
                library=str(args.library.resolve()),library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                harness_npu=args.npu,shm_id=args.shm_id,scope='offline D2 format; no TP4 claim',status='planned')
    write(out/'result.json',report)
    if args.dry_run:return 0
    for pci,driver in ((registration['pci_addr'],'uio_pci_generic'),(registration['protected_pci_addr'],'nvme')):
        if (Path('/sys/bus/pci/devices')/pci/'driver').resolve().name!=driver:
            raise RuntimeError('device driver mismatch: '+pci)
    if args.npu!=7:raise ValueError('offline harness reserves only NPU7')
    smi=subprocess.check_output(['npu-smi','info'],text=True)
    if 'No running processes found in NPU 7' not in smi:raise RuntimeError('offline NPU is occupied')
    (out/'npu-before.txt').write_text(smi)
    os.environ['SPDK_SHM_ID']=str(args.shm_id)
    transport=None
    try:
        transport=FullTransport(load_backend(args.library),pci=registration['pci_addr'],npu=args.npu,
                                depth=4,chunk_size=config['chunk_bytes'],profiling_dir=out)
        backend=RegisteredBackend(transport,registration,region_id=registration['region_id'])
        report['namespace_bytes']=transport.total_bytes
        header=backend.read(backend.base,3*BLOCK)
        (out/'header-before.bin').write_bytes(header)
        report['header_blank']=not any(header)
        report['header_sha256']=hashlib.sha256(header).hexdigest()
        region=Region(backend,offset=backend.base,length=backend.end-backend.base,retention=3)
        if args.operation=='initialize':
            if not args.expected_header_sha256 or report['header_sha256']!=args.expected_header_sha256:
                raise ValueError('initialization requires exact inspected header digest')
            if header[:8]==b'NPUNVM3\0':raise ValueError('existing D2 header requires offline migration, not initialization')
            backend.write(backend.base,bytes(3*BLOCK));backend.flush()
            region.format(region_id=registration['region_id'])
        elif args.operation=='format':
            region.format(region_id=registration['region_id'])
        elif args.operation!='preflight':
            report['mount_errors']=region.mount()
            if region.header['region_id']!=registration['region_id']:raise ValueError('mounted region identity differs')
            if args.operation=='probe-save':
                request='format-probe-'+out.name
                region.begin(request,{'scope':'D2 persistent roundtrip'},reserve_bytes=1<<20)
                data=bytes((i*131+17)&255 for i in range(8197))
                ref=region.payload(data)
                rows=[dict(rank=0,name='probe',shape=[len(data)],dtype='uint8',partition='replicated',
                           logical_offset=0,logical_bytes=len(data),payload=ref)]
                report['receipt']=region.commit(rows,step=1,topology={'world_size':1},
                    rank_controls=[{'rank':0,'step':1,'controls':{'cursor':1}}])
            if region.current:
                report['generation']=region.current['sequence']
                report['retained_generations']=[g['generation'] for g in region.current['state']['generations']]
                if args.operation=='verify':
                    report['verified']=[region.verify_payloads(g) for g in region.current['state']['generations']]
            elif args.operation=='verify':raise ValueError('no committed generation to verify')
        report['status']='pass'
    except BaseException as error:
        report.update(status='fail',error=type(error).__name__+': '+str(error))
        raise
    finally:
        if transport is not None:
            try:
                transport.close(120)
                report['closed']=True
            except BaseException as error:
                report.update(status='fail',closed=False,close_error=str(error))
        write(out/'result.json',report)
        if transport is not None and report.get('closed') is False:
            # Preserve the live address space when DMA stop is unproven.
            # The supervisor observes the report and must not reopen the disk.
            import threading
            threading.Event().wait()
    return 0 if report['status']=='pass' else 1


if __name__=='__main__':raise SystemExit(main())
