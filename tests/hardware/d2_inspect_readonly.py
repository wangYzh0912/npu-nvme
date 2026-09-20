#!/usr/bin/env python3
"""Read D2 header/anchors/manifests without exposing a backend write method."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'python')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--region-config', type=Path, required=True)
    parser.add_argument('--shm-id', type=int, required=True)
    args = parser.parse_args()
    from npu_nvme.storage.bindings import load_backend
    from npu_nvme.storage.full_transport import FullTransport
    from npu_nvme.d2.backend import RegisteredBackend, registration_from_config
    from npu_nvme.d2.format import Region
    args.out.mkdir(parents=True, exist_ok=False)
    registration = registration_from_config(args.region_config)
    os.environ['SPDK_SHM_ID'] = str(args.shm_id)
    report = dict(status='running', closed=False,
                  scope='read-only metadata inspection; payload integrity not verified')
    def write():
        (args.out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    write()
    transport = FullTransport(load_backend(args.library), pci=registration['pci_addr'],
        npu=-1, depth=4, chunk_size=4 << 20, profiling_dir=args.out, role='host_owner')
    try:
        backend = RegisteredBackend(transport, registration, region_id=registration['region_id'])
        class ReadOnly:
            def read(self, offset, size):
                return backend.read(offset, size)
        header = backend.read(backend.base, 3*4096)
        (args.out/'header.bin').write_bytes(header)
        report['header_sha256'] = hashlib.sha256(header).hexdigest()
        region = Region(ReadOnly(), offset=backend.base, length=registration['length'], retention=3)
        errors = region.mount()
        report.update(status='pass', rejected_anchors=errors,
            sequence=region.current['sequence'] if region.current else None)
    except Exception as error:
        report.update(status='unreadable', error=repr(error))
    finally:
        try:
            transport.close(120)
            report['closed'] = True
        except BaseException as error:
            report.update(status='retained', error=repr(error))
            write()
            threading.Event().wait()
        write()
    return int(report['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
