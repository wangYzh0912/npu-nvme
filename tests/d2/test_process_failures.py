"""Real process/socket failures on a durable file backend (no DMA claim)."""
from contextlib import contextmanager
import multiprocessing as mp
import os
import socket
import time

import pytest

from npu_nvme.d2.format import Region
from npu_nvme.d2.session import OwnerSession, RankSession


class FileDisk:
    def __init__(self, path, crash=False):
        self.fd = os.open(path, os.O_RDWR)
        self.crash = crash

    def read(self, offset, size):
        return os.pread(self.fd, size, offset)

    def write(self, offset, data):
        assert os.pwrite(self.fd, data, offset) == len(data)

    def flush(self):
        os.fsync(self.fd)
        if self.crash:
            os._exit(77)


def owner_process(path, connections, operation, fault, results):
    disk = FileDisk(path)
    try:
        region = Region(disk, offset=0, length=32 << 20, retention=3)
        region.mount()
        disk.crash = fault == 'owner_exit'
        owner = OwnerSession(region, dict(enumerate(connections)), epoch='epoch',
            request_id='baseline' if operation == 'seed' else 'next',
            topology={'world_size':2}, chunk_bytes=4096,
            deadline=time.monotonic()+8, identity={'fixture':'process-faults'})
        value = owner.restore() if operation == 'restore' else owner.save(step=8)
        results.put(('owner', 'pass', value))
    except Exception as error:
        results.put(('owner', 'fail', repr(error)))
    finally:
        for connection in connections:
            connection.close()
        os.close(disk.fd)


def rank_process(connection, rank, operation, fault, results):
    try:
        schema = [dict(rank=rank, name='weight', shape=[8197], dtype='uint8',
                       partition='sharded', bytes=8197)]
        session = RankSession(connection, rank=rank, epoch='epoch',
            identity={'fixture':'process-faults'}, schema=schema,
            chunk_bytes=4096, deadline=time.monotonic()+8)
        expected = bytes([rank+1])*8197
        if operation != 'restore':
            @contextmanager
            def read(name, offset, size):
                if fault == 'rank_disconnect' and rank == 1:
                    raise ConnectionAbortedError('injected rank disconnect')
                yield expected[offset:offset+size]
            value = session.save(read, {'rank':rank}, step=8)
        else:
            observed = bytearray(len(expected))
            def apply(name, offset, data, digest):
                observed[offset:offset+len(data)] = data
            def controls(value, step):
                assert bytes(observed) == expected and value == {'rank':rank} and step == 8
                if fault == 'controls_rejected' and rank == 1:
                    raise ValueError('injected local control rejection')
            value = session.restore(apply, controls)
        results.put((str(rank), 'pass', value))
    except Exception as error:
        results.put((str(rank), 'fail', repr(error)))
    finally:
        connection.close()


def run_processes(path, operation, fault=None):
    context = mp.get_context('spawn')
    results = context.Queue()
    pairs = [socket.socketpair() for _ in range(2)]
    processes = [context.Process(target=owner_process,
        args=(path, [p[0] for p in pairs], operation, fault, results))]
    processes += [context.Process(target=rank_process,
        args=(pair[1], rank, operation, fault, results)) for rank, pair in enumerate(pairs)]
    try:
        for process in processes:
            process.start()
        for pair in pairs:
            for connection in pair:
                connection.close()
        deadline = time.monotonic()+15
        for process in processes:
            process.join(max(0, deadline-time.monotonic()))
        assert all(not p.is_alive() for p in processes), 'process failure did not terminate'
        expected_count = 2 if fault == 'owner_exit' else 3
        records = [results.get(timeout=2) for _ in range(expected_count)]
        assert [p.exitcode for p in processes] == ([77, 0, 0] if fault == 'owner_exit' else [0, 0, 0])
        return records
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        results.close()


@pytest.mark.parametrize('fault', ['rank_disconnect', 'owner_exit', 'controls_rejected'])
def test_process_failure_preserves_committed_generation(tmp_path, fault):
    path = tmp_path/'region.img'
    with path.open('wb') as stream:
        stream.truncate(32 << 20)
    disk = FileDisk(path)
    Region(disk, offset=0, length=32 << 20, retention=3).format(region_id='process-test')
    os.close(disk.fd)
    assert all(r[1] == 'pass' for r in run_processes(path, 'seed'))
    operation = 'restore' if fault == 'controls_rejected' else 'save'
    assert all(r[1] == 'fail' for r in run_processes(path, operation, fault))
    disk = FileDisk(path)
    try:
        fresh = Region(disk, offset=0, length=32 << 20, retention=3)
        fresh.mount()
        assert fresh.current['sequence'] == 1
    finally:
        os.close(disk.fd)
    assert all(r[1] == 'pass' for r in run_processes(path, 'restore'))
