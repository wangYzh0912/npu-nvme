#!/usr/bin/env python3
import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from npu_nvme.experiments.verify_media import verify_receipt
from npu_nvme.storage.bindings import load_backend
from npu_nvme.storage.full_transport import FullTransport


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--connection',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    config=json.loads(args.connection.read_text());catalog=json.loads(Path(config['catalog']).read_text())
    lock_path=Path('/models/npu_nvme_exp/user7-stack/hardware-campaign.lock')
    lock=lock_path.open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if lock_path.with_suffix('.lease.json').exists():raise RuntimeError('hardware lease still active')
    transport=FullTransport(load_backend(config['library']),pci=config['pci_addr'],npu=-1,depth=4,
        chunk_size=config['chunk_bytes'],profiling_dir=args.out,role='host_owner')
    try:rows=[verify_receipt(transport,receipt,chunk_bytes=config['chunk_bytes']) for receipt in catalog['commits']]
    finally:
        transport.close(config['timeout_seconds'])
        lock.close()
    result=dict(status='pass',commits=len(rows),rows=rows);args.out.mkdir(parents=True,exist_ok=True);(args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
