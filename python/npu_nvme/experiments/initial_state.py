"""Disk-backed initial state used to undo graph warmup outside timed steps."""
from pathlib import Path
import hashlib
import os

from npu_nvme.runtime.training_catalog import read_checked, write_checked


def registry(network):
    result = {}
    seen = set()
    for _, parameter in network.parameters_and_names():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.name in result:
            raise ValueError('duplicate initial-state parameter name')
        result[parameter.name] = parameter
    return result


def capture(ms, network, directory, *, identity, data_sha256, horizon):
    import numpy as np
    from qwen_native_state import capture_control
    directory = Path(directory)
    marker = directory / 'manifest.json'
    if marker.exists():
        manifest = read_checked(marker)
        if manifest['identity'] != identity:
            raise ValueError('initial FULL identity differs')
        for entry in manifest['tensors'].values():
            path = (directory / entry['file']).resolve()
            if path.parent != directory.resolve() or not path.is_file():
                raise ValueError('initial FULL payload missing or escaping')
            array = np.load(path, allow_pickle=False, mmap_mode='r')
            if (list(array.shape) != entry['shape'] or array.dtype.str != entry['dtype'] or
                    array.nbytes != entry['bytes'] or
                    hashlib.sha256(memoryview(array).cast('B')).hexdigest() != entry['sha256']):
                raise ValueError('initial FULL payload geometry differs')
        return manifest
    directory.mkdir(parents=True, exist_ok=False)
    ms.runtime.synchronize()
    rows = {}
    for index, (name, parameter) in enumerate(sorted(registry(network).items())):
        array = np.array(parameter.asnumpy(), copy=True, order='C')
        filename = f'tensor-{index:05d}.npy'
        with (directory / filename).open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        rows[name] = dict(file=filename, shape=list(parameter.shape), dtype=array.dtype.str,
                          bytes=array.nbytes, sha256=hashlib.sha256(memoryview(array).cast('B')).hexdigest())
    control = capture_control(ms, step=0, data_sha256=data_sha256, lr_horizon=horizon, small={})
    manifest = dict(version=1, identity=identity, tensors=rows, control=control)
    write_checked(marker, manifest)
    return manifest


def restore(ms, network, directory, *, identity):
    import numpy as np
    from qwen_native_state import restore_control
    directory = Path(directory)
    manifest = read_checked(directory / 'manifest.json')
    parameters = registry(network)
    if manifest['identity'] != identity or set(parameters) != set(manifest['tensors']):
        raise ValueError('initial FULL identity or parameter coverage differs')
    ms.runtime.synchronize()
    for name, entry in manifest['tensors'].items():
        path = (directory / entry['file']).resolve()
        if path.parent != directory.resolve():
            raise ValueError('initial FULL tensor escapes directory')
        array = np.load(path, allow_pickle=False)
        parameter = parameters[name]
        if (list(parameter.shape) != entry['shape'] or list(array.shape) != entry['shape'] or
                array.dtype.str != entry['dtype'] or array.nbytes != entry['bytes'] or
                hashlib.sha256(memoryview(array).cast('B')).hexdigest() != entry['sha256']):
            raise ValueError('initial FULL tensor integrity differs: ' + name)
        parameter.set_data(ms.Tensor(array, dtype=parameter.dtype))
        actual = np.array(parameter.asnumpy(), copy=True, order='C')
        if hashlib.sha256(memoryview(actual).cast('B')).hexdigest() != entry['sha256']:
            raise ValueError('initial FULL device restore differs: ' + name)
    restore_control(ms, network, manifest['control'])
    ms.runtime.synchronize()
    return dict(tensors=len(parameters), bytes=sum(row['bytes'] for row in manifest['tensors'].values()),
                verified=True)
