"""Native TP4 continuation sidecars; no runtime import until a callback uses it."""
import base64
import hashlib
import json
from pathlib import Path
import random
import time


def write_json(path, value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def parameter_manifest(network):
    import numpy as np
    result={};small={};seen=set()
    for _,param in network.parameters_and_names():
        if id(param) in seen: continue
        seen.add(id(param));name=param.name
        if name in result:raise ValueError('duplicate parameter name '+name)
        array=np.asarray(param.asnumpy())
        if not array.flags.c_contiguous: array=np.ascontiguousarray(array)
        digest=hashlib.sha256(memoryview(array).cast('B')).hexdigest()
        result[name]=dict(shape=list(array.shape),dtype=array.dtype.str,bytes=array.nbytes,sha256=digest)
        if array.nbytes<=1024:
            small[name]=dict(result[name],data=base64.b64encode(array.tobytes()).decode())
    return result,small


def capture_control(ms, *, step, data_sha256, lr_horizon, small):
    import numpy as np
    state=np.random.get_state();ms_rng=np.ascontiguousarray(ms.get_rng_state().asnumpy())
    return dict(schema_version=1,logical_optimizer_step=step,sink_iteration=step,
                next_data_row=step,data_sha256=data_sha256,lr_horizon=lr_horizon,
                python_rng=random.getstate(),numpy_rng=[state[0],state[1].tolist(),*state[2:]],
                mindspore_rng=dict(dtype=ms_rng.dtype.str,shape=list(ms_rng.shape),
                                   data=base64.b64encode(ms_rng.tobytes()).decode()),
                small_parameters=small,
                scope='Python/NumPy/default MindSpore RNG; fixed tokens and zero dropout; no reshard')


def restore_control(ms,network,control):
    import numpy as np
    registry={p.name:p for _,p in network.parameters_and_names()}
    for name,entry in control['small_parameters'].items():
        if name not in registry:raise ValueError('missing sidecar parameter '+name)
        raw=base64.b64decode(entry['data'],validate=True)
        if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError('damaged scalar sidecar '+name)
        value=np.frombuffer(raw,dtype=np.dtype(entry['dtype'])).reshape(entry['shape']).copy()
        target=registry[name]
        if tuple(target.shape)!=tuple(value.shape):raise ValueError('scalar sidecar shape mismatch')
        target.set_data(ms.Tensor(value,dtype=target.dtype))
    def tuples(value):return tuple(tuples(x) for x in value) if isinstance(value,list) else value
    random.setstate(tuples(control['python_rng']))
    state=control['numpy_rng'];np.random.set_state((state[0],np.asarray(state[1],np.uint32),*state[2:]))
    state=control['mindspore_rng'];raw=base64.b64decode(state['data'],validate=True)
    ms.set_rng_state(ms.Tensor(np.frombuffer(raw,dtype=np.dtype(state['dtype'])).reshape(state['shape']).copy()))


def compare_manifests(expected,actual):
    if set(expected)!=set(actual):
        raise ValueError(f'parameter names differ: missing={sorted(set(expected)-set(actual))[:8]}, extra={sorted(set(actual)-set(expected))[:8]}')
    different=[name for name in expected if expected[name]!=actual[name]]
    if different:raise ValueError('state bytes differ: '+repr(different[:12]))


def ready_barrier(out,rank,timeout=120):
    """File coordination precedes any continuation collective; finite deadline."""
    out=Path(out);write_json(out/f'ready-rank-{rank}.json',dict(rank=rank,ready=True))
    deadline=time.monotonic()+timeout
    while True:
        if any(out.glob('failed-rank-*.json')):raise RuntimeError('another rank failed restore')
        if all((out/f'ready-rank-{r}.json').is_file() for r in range(4)):return
        if time.monotonic()>=deadline:raise TimeoutError('four-rank ready deadline expired')
        time.sleep(0.05)
