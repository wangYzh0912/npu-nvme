"""Fixed TP4 D2 callback contracts, kept distinct from Native checkpoint files."""
import hashlib
import json
from pathlib import Path


def partitions_for(manifest,small,strategy_path):
    strategy=json.loads(Path(strategy_path).read_text())
    result={row['name']:row['partition'] for row in strategy['tensors']}
    for name in manifest:
        if name not in result:
            if name not in small:raise ValueError('runtime tensor absent from authoritative strategy: '+name)
            result[name]='per_rank_control'
    return {name:result[name] for name in manifest}


def validate_restart(run,rank,step,horizon):
    run=Path(run);contract=json.loads((run/'d2_restart_contract.json').read_text())
    if contract.get('step')!=step or contract.get('lr_horizon')!=horizon or contract.get('world_size')!=4:
        raise ValueError('D2 restart schedule/topology differs')
    rows=contract.get('ranks',[])
    if len(rows)!=4 or {v.get('rank') for v in rows}!={0,1,2,3}:raise ValueError('D2 source rank set incomplete')
    for row in rows:
        if Path('/proc',str(row['pid'])).exists():raise ValueError('D2 source still alive')
        required={f'rank_{row["rank"]}/{name}' for name in ('state-checkpoint.json','control-checkpoint.json','resolved_config.json','input_ids.npy','d2-save.json','d2-partitions.json')}
        if set(row['artifacts'])!=required:raise ValueError('D2 source artifact set differs')
        for name,digest in row['artifacts'].items():
            path=(run/name).resolve()
            if run.resolve() not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('D2 source artifact changed')
    return contract


def validate_identity(run,config,model_hash,*,remaining_steps):
    from qwen_native_state import training_identity
    expected=training_identity(config)
    for rank in range(4):
        report=json.loads((Path(run)/f'rank_{rank}/acceptance.json').read_text())
        source=training_identity(json.loads((Path(run)/f'rank_{rank}/resolved_config.json').read_text()))
        # Only execution length changes on the raw restart. Logical step and LR
        # state are restored and checked before the first optimizer update.
        if expected['runner']['stop_step']!=remaining_steps:raise ValueError('D2 continuation length differs')
        source['runner']['stop_step']=remaining_steps
        if source!=expected or report['config_sha256']!=model_hash:raise ValueError('D2 training identity differs')
