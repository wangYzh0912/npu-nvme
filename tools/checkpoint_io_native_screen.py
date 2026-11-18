#!/usr/bin/env python3
"""Bounded host-owner request-count screen in an allocated, unpublished extent."""
import argparse
import ctypes as C
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--region', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--shm-id', type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source / 'python'))
    from npu_nvme.storage.bindings import load_backend
    from npu_nvme.storage.full_transport import FullTransport
    from npu_nvme.d2.backend import RegisteredBackend, registration_from_config
    from npu_nvme.d2.format import Region, allocate
    from npu_nvme.experiments.checkpoint_io_closure import native_snapshot, verify_drained

    if os.geteuid() != 0:
        raise PermissionError('native host-owner screen requires root')
    lease = args.lock.with_suffix('.lease.json')
    if lease.exists():
        raise RuntimeError('hardware campaign lease is active')
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', pid=os.getpid(), closed=False, phases=[],
                  limits=dict(small_writes=66560, small_bytes=4096,
                              large_writes=512, large_bytes=4 << 20,
                              admission_seconds=300, request_timeout_ms=60000),
                  scope='single native host-owner; no NPU, training or cumulative-TB equivalence')
    record = lambda: write_json(args.output / 'result.json', report)
    record()
    lock = args.lock.open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if lease.exists():
        raise RuntimeError('hardware campaign lease appeared')
    os.environ['SPDK_SHM_ID'] = str(args.shm_id)
    transport = None
    stop = threading.Event()
    monitor = None
    monitor_errors = []
    try:
        registration = registration_from_config(args.region)
        for pci, expected in ((registration['pci_addr'], 'uio_pci_generic'),
                              (registration['protected_pci_addr'], 'nvme')):
            actual = (Path('/sys/bus/pci/devices') / pci / 'driver').resolve().name
            if actual != expected:
                raise ValueError(f'PCI driver mismatch: {pci}: {actual}')
        library = Path(os.environ['NPU_NVME_LIBRARY_PATH'])
        report['identity'] = dict(
            source_commit=subprocess.check_output(
                ['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
            library=str(library), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
            region=str(args.region), region_sha256=hashlib.sha256(args.region.read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        transport = FullTransport(load_backend(library), pci=registration['pci_addr'],
                                  npu=-1, depth=4, chunk_size=4 << 20,
                                  profiling_dir=args.output, role='host_owner')
        lib = transport.lib
        lib.npu_nvme_get_io_timeout_ms.argtypes = [C.c_void_p]
        lib.npu_nvme_get_io_timeout_ms.restype = C.c_uint32
        if lib.npu_nvme_get_io_timeout_ms(transport.ctx) != 60000:
            raise ValueError('native timeout must remain 60000 ms')
        backend = RegisteredBackend(transport, registration, region_id=registration['region_id'])
        region = Region(backend, offset=backend.base, length=backend.end-backend.base, retention=3)
        errors = region.mount()
        if errors or region.header['region_id'] != registration['region_id']:
            raise ValueError(f'existing catalog mount is not clean: {errors}')
        header_before = backend.read(backend.base, 3 * 4096)
        protected = region.protected()
        scratch = allocate(region.data_base, region.end, 4 << 20, protected)
        if any(scratch.offset < e.offset + e.length and e.offset < scratch.offset + scratch.length
               for e in protected):
            raise AssertionError('scratch overlaps catalog-protected data')
        report['scratch'] = dict(offset=scratch.offset, length=scratch.length,
                                 protected_extents=len(protected),
                                 publication='none; no format or catalog mutation')
        report['header_before_sha256'] = hashlib.sha256(header_before).hexdigest()
        report['before'] = native_snapshot(transport)
        record()

        def sample():
            try:
                with (args.output / 'native-samples.jsonl').open('w') as stream:
                    while not stop.is_set():
                        row = dict(monotonic_ns=time.monotonic_ns(), realtime_ns=time.time_ns(),
                                   native=native_snapshot(transport))
                        stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                        stream.flush()
                        stop.wait(.25)
            except BaseException as error:
                monitor_errors.append(repr(error))
                stop.set()

        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        deadline = time.monotonic() + 300
        first_payload = bytes(range(256)) * 16
        large_payload = bytes(range(256)) * (16384)
        with (args.output / 'request-events.jsonl').open('w', buffering=1) as events:
            for name, count, payload in (('request_count_4k', 66560, first_payload),
                                          ('chunk_path_4m', 512, large_payload)):
                phase = dict(name=name, target_writes=count, bytes_per_write=len(payload),
                             completed_writes=0, begin_ns=time.monotonic_ns(),
                             latencies_ms=[])
                report['phases'].append(phase)
                native_before = native_snapshot(transport)
                for index in range(count):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('bounded screen admission deadline expired')
                    if monitor_errors:
                        raise RuntimeError('native sampling failed: ' + str(monitor_errors))
                    begin = time.monotonic_ns()
                    events.write(json.dumps(dict(event='request_begin', phase=name, index=index,
                                                 offset=scratch.offset, bytes=len(payload),
                                                 monotonic_ns=begin, realtime_ns=time.time_ns())) + '\n')
                    transport.write(scratch.offset, payload)
                    end = time.monotonic_ns()
                    phase['latencies_ms'].append((end-begin)/1e6)
                    phase['completed_writes'] += 1
                    events.write(json.dumps(dict(event='request_complete', phase=name, index=index,
                                                 monotonic_ns=end, elapsed_ns=end-begin)) + '\n')
                    if index % 4096 == 0:
                        record()
                transport.flush()
                observed = backend.read(scratch.offset, len(payload))
                if observed != payload:
                    raise AssertionError(f'{name} readback mismatch')
                phase['end_ns'] = time.monotonic_ns()
                phase['seconds'] = (phase['end_ns']-phase['begin_ns'])/1e9
                phase['native_before'] = native_before
                phase['native_after'] = native_snapshot(transport)
                values = sorted(phase.pop('latencies_ms'))
                phase['latency_ms'] = dict(minimum=values[0], maximum=values[-1],
                                          median=values[len(values)//2],
                                          p99=values[(99*len(values)+99)//100-1])
                record()
        header_after = backend.read(backend.base, 3*4096)
        if header_after != header_before:
            raise AssertionError('catalog superblock/anchors changed')
        report['header_after_sha256'] = hashlib.sha256(header_after).hexdigest()
        report['after'] = native_snapshot(transport)
        verify_drained(report['after'])
        report['status'] = 'pass'
    except BaseException as error:
        report.update(status='fail', error=repr(error))
        if transport:
            report['failure_native'] = native_snapshot(transport)
        raise
    finally:
        stop.set()
        if monitor:
            monitor.join(timeout=2)
            if monitor.is_alive() or monitor_errors:
                report.update(status='fail', monitor_errors=monitor_errors,
                              monitor_stopped=not monitor.is_alive())
        if transport:
            try:
                transport.close(60)
                report['closed'] = True
            except BaseException as error:
                report.update(status='retained', close_error=repr(error))
                record()
                # Neither process exit nor lease release is safe without stop proof.
                threading.Event().wait()
        report['end_ns'] = time.monotonic_ns()
        record()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    if report['status'] != 'pass':
        raise RuntimeError('screen did not pass: ' + repr(report))
    print(json.dumps(dict(status=report['status'], output=str(args.output), closed=report['closed'])))


if __name__ == '__main__':
    main()
