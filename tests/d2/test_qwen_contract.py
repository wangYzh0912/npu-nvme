import hashlib
import json
from pathlib import Path
import pytest
from qwen_d2_state import partitions_for,validate_restart


def test_partitions_require_authoritative_tensor_or_recorded_control(tmp_path):
    p=tmp_path/'strategy.json';p.write_text(json.dumps({'tensors':[{'name':'weight','partition':'replicated'}]}))
    assert partitions_for({'weight':{},'step':{}},{'step':{}},p)=={'weight':'replicated','step':'per_rank_control'}
    with pytest.raises(ValueError):partitions_for({'unknown_tensor':{}},{},p)


def contract(tmp_path):
    rows=[]
    for rank in range(4):
        directory=tmp_path/f'rank_{rank}';directory.mkdir();artifacts={}
        for name in ('state-checkpoint.json','control-checkpoint.json','resolved_config.json','input_ids.npy','d2-save.json','d2-partitions.json'):
            path=directory/name;path.write_bytes(b'{}')
            artifacts[str(path.relative_to(tmp_path))]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(dict(rank=rank,pid=2**30+rank,artifacts=artifacts))
    value=dict(step=8,lr_horizon=11,world_size=4,ranks=rows)
    (tmp_path/'d2_restart_contract.json').write_text(json.dumps(value))
    return value


def test_restart_contract_checks_every_rank_before_runtime(tmp_path):
    contract(tmp_path);validate_restart(tmp_path,0,8,11)
    (tmp_path/'rank_3/control-checkpoint.json').write_text('changed')
    with pytest.raises(ValueError,match='artifact changed'):validate_restart(tmp_path,0,8,11)


def test_restart_rejects_live_source(tmp_path):
    import os
    value=contract(tmp_path);value['ranks'][3]['pid']=os.getpid()
    (tmp_path/'d2_restart_contract.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='still alive'):validate_restart(tmp_path,0,8,11)
