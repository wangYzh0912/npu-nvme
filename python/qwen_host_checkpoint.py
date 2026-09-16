"""Rank-local ByteCheckpoint Host port for a fixed TP4 training workload."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
from multiprocessing import shared_memory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def checkpoint(framework, network, *, operation, rank, out, source=None,
               controls=None, verify=None, barrier=None, timeout=1800,
               host_budget=128 << 30):
    from npu_nvme.d2.tp_schema import validate, validate_rank, DTYPES
    from training_state import encode_control_value, decode_control_value
    strategy = json.loads((ROOT/'results/long-term-v1.3/EN/state_schema-002.json').read_text())
    authoritative = validate(strategy)
    framework.hal.synchronize()
    registry = {}; seen = set(); fields = []; offset = 0; names = {}
    for _, parameter in network.parameters_and_names():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter)); name = parameter.name
        if name in registry or name not in authoritative:
            raise ValueError('unexpected/duplicate Qwen tensor: '+name)
        spec = authoritative[name]
        shape = list(getattr(parameter, 'sliced_shape', None) or parameter.shape)
        if np.prod(parameter.data.shape) < np.prod(shape):
            shape = list(parameter.data.shape)
        dtype = np.dtype(framework.dtype_to_nptype(parameter.dtype))
        size = int(np.prod(shape)) * dtype.itemsize
        validate_rank_row = dict(rank=rank, name=name, shape=shape, dtype=dtype.name,
                                 partition=spec['partition'], bytes=size)
        registry[name] = (parameter, validate_rank_row)
        prefix = 'optimizer/' if spec['role'] in ('adam_m', 'adam_v') else 'model/'
        key = prefix+name; names[key] = name
        fields.append(dict(name=key, dtype=dtype.str, shape=shape, offset=offset, nbytes=size))
        offset += size
    validate_rank([row for _, row in registry.values()], strategy, rank)
    out = Path(out); directory = out/f'rank_{rank}'
    directory.mkdir(parents=True, exist_ok=True)
    storage = (Path(source) if source else out)/f'rank_{rank}'/'bytecheckpoint'
    if operation == 'restore':
        saved = json.loads((storage/'bridge.json').read_text())
        if saved['fields'] != fields or saved['strategy_sha256'] != canonical(strategy):
            raise ValueError('ByteCheckpoint restore schema differs')
        control_meta = saved['control_metadata']; control_size = saved['control_bytes']
    else:
        control_data, control_meta = encode_control_value(controls)
        control_size = control_data.nbytes
    size = offset+control_size
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    shm_free = os.statvfs('/dev/shm').f_bavail * os.statvfs('/dev/shm').f_frsize
    # Four ranks, worker planner copies, and one largest tensor export each.
    if size*3 > host_budget or size*12+(16 << 30) > available or size*4 > shm_free:
        raise MemoryError('ByteCheckpoint TP4 Host/shared-memory admission failed')
    owner = shared_memory.SharedMemory(create=True, size=size,
                                      name='qwen_bc_'+uuid.uuid4().hex)
    descriptor = dict(name=owner.name, size=size, fields=fields,
                      controls=dict(offset=offset, nbytes=control_size, metadata=control_meta), schema={})
    worker = None; started = time.monotonic(); expected = None
    try:
        if operation == 'save':
            digest = hashlib.sha256()
            for field in sorted(fields, key=lambda value: value['name']):
                param = registry[names[field['name']]][0]
                array = np.ascontiguousarray(param.asnumpy())
                if array.nbytes != field['nbytes'] or list(array.shape) != field['shape'] or array.dtype.str != field['dtype']:
                    raise ValueError('ByteCheckpoint source geometry changed')
                raw = memoryview(array).cast('B')
                owner.buf[field['offset']:field['offset']+len(raw)] = raw
                digest.update(field['name'].encode()); digest.update(raw)
                del raw, array
            owner.buf[offset:] = memoryview(control_data).cast('B')
            digest.update(control_data.tobytes()); expected = digest.hexdigest()
        worker_python = '/home/user7/npu-nvme-baseline-envs/bytecheckpoint/bin/python'
        environment = dict(os.environ)
        environment['PYTHONPATH'] = os.pathsep.join((str(ROOT),
            '/home/user7/npu-nvme-baseline-upstreams/ByteCheckpoint',
            '/home/user7/.local/lib/python3.9/site-packages'))
        environment['BYTECHECKPOINT_ENABLE_PINNED_MEM_D2H'] = '0'
        environment['BYTECHECKPOINT_ENABLE_TREE_TOPO'] = '0'
        with (directory/f'bytecheckpoint-{operation}.stderr.log').open('w') as log:
            worker = subprocess.Popen([worker_python, '-m',
                'experiments.baselines.repro.workers.bytecheckpoint_worker'], cwd=ROOT,
                env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=log, text=True)
            request = dict(operation=operation, generation=8, checkpoint_dir=str(storage),
                           descriptor=descriptor, timeout_seconds=timeout)
            output, _ = worker.communicate(json.dumps(request)+'\n'+json.dumps(dict(operation='close'))+'\n',
                                            timeout=timeout)
        (directory/f'bytecheckpoint-{operation}.stdout.log').write_text(output)
        replies = []
        for line in output.splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and 'status' in value:
                replies.append(value)
        if worker.returncode or len(replies) != 2 or replies[0]['status'] != 'ok' or replies[1]['status'] != 'closed':
            raise RuntimeError('ByteCheckpoint worker failed: '+repr(replies))
        response = replies[0]
        if operation == 'save':
            if response['sha256'] != expected:
                raise ValueError('ByteCheckpoint persisted bytes differ')
            saved = dict(fields=fields, control_metadata=control_meta, control_bytes=control_size,
                         strategy_sha256=canonical(strategy), sha256=expected)
            bridge = storage/'bridge.json'
            bridge.write_text(json.dumps(saved, indent=2)+'\n')
            with bridge.open('rb') as stream:
                os.fsync(stream.fileno())
            fd = os.open(storage, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        else:
            if response['sha256'] != saved['sha256']:
                raise ValueError('ByteCheckpoint restored digest differs')
            for field in fields:
                param = registry[names[field['name']]][0]
                array = np.ndarray(field['shape'], dtype=field['dtype'], buffer=owner.buf,
                                   offset=field['offset']).copy()
                param.set_data(framework.Tensor(array, dtype=param.dtype))
            restored = decode_control_value(np.frombuffer(owner.buf[offset:], np.uint8).copy(), control_meta)
            framework.hal.synchronize(); verify(restored, 8); barrier()
        return dict(receipt=dict(generation=8), capture='blocking_host_snapshot',
                    state_bytes=offset, seconds=time.monotonic()-started,
                    upstream_core_invoked=True, kind='host-adapted-semantic-port',
                    mapping='one upstream world_size=1 worker per TP shard; global commit in launcher',
                    shared_memory_bytes=size, worker=response)
    finally:
        if worker is not None and worker.poll() is None:
            (directory/'retained.json').write_text(json.dumps(dict(pid=os.getpid(), worker_pid=worker.pid,
                shared_memory=owner.name, reason='worker still owns mapping')))
            threading.Event().wait()
        owner.close(); owner.unlink()
