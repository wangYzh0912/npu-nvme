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
        if all((out/f'ready-rank-{r}.json').is_file() for r in range(4)):
            for r in range(4):
                value=json.loads((out/f'ready-rank-{r}.json').read_text())
                if value.get('rank')!=r or value.get('ready') is not True:
                    raise ValueError('invalid rank readiness record')
            return
        if time.monotonic()>=deadline:raise TimeoutError('four-rank ready deadline expired')
        time.sleep(0.05)


def validate_restart_contract(run,rank,step,horizon):
    """Reject malformed sets/identities before importing or joining the runtime."""
    run=Path(run).resolve();contract=json.loads((run/'restart_contract.json').read_text())
    if (contract.get('schema_version')!=1 or contract.get('checkpoint_step')!=step or
        contract.get('lr_horizon')!=horizon or contract.get('topology')!=dict(tp=4,dp=1,pp=1)):
        raise ValueError('restart step/horizon/topology contract mismatch')
    rows=contract.get('ranks',[])
    if len(rows)!=4 or {r.get('rank') for r in rows}!={0,1,2,3}:
        raise ValueError('restart requires exactly four ranks')
    for row in rows:
        r=row['rank']
        required={f'rank_{r}/control-checkpoint.json',f'rank_{r}/state-checkpoint.json',
                  f'rank_{r}/input_ids.npy',f'rank_{r}/resolved_config.json',
                  f'training/checkpoint/rank_{r}/qwen3_rank_{r}-{step}_1.safetensors'}
        if set(row.get('artifacts',{}))!=required:
            raise ValueError('restart artifact coverage differs from required set')
        if Path('/proc',str(row['source_pid'])).exists():raise ValueError('source process has not exited')
        for name,record in row['artifacts'].items():
            path=(run/name).resolve()
            if run not in path.parents or not path.is_file() or path.stat().st_size!=record['bytes']:
                raise ValueError('missing, escaping or truncated restart artifact')
            # Every worker hashes its own payload before communication; all
            # workers validate the complete rank/file set independently.
            if row['rank']==rank:
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
                if h.hexdigest()!=record['sha256']:raise ValueError('damaged restart artifact')
    return contract


def training_identity(config):
    """Semantic settings that must match across a fixed-topology restart."""
    context = dict(config['context'])
    # Per-rank physical device selection is checked by the launcher/runtime.
    context.pop('device_id', None)
    identity = {key: config[key] for key in
                ('seed', 'model', 'parallel_config', 'optimizer', 'lr_schedule')}
    identity['context'] = context
    runner = config['runner_config']
    identity['runner'] = {key: runner[key] for key in ('sink_mode', 'sink_size', 'stop_step')}
    return identity


def validate_training_identity(run, resolved_config, model_config_sha256):
    """Call after shard-contract validation and before entering collectives."""
    run = Path(run)
    expected = training_identity(resolved_config)
    contract = json.loads((run/'restart_contract.json').read_text())
    for row in contract['ranks']:
        source_config = json.loads((run/f"rank_{row['rank']}/resolved_config.json").read_text())
        if row['config_sha256'] != model_config_sha256:
            raise ValueError('restart model configuration identity differs')
        if training_identity(source_config) != expected:
            raise ValueError('restart training configuration identity differs')
