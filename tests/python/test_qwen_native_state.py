import hashlib
import json
import os
from pathlib import Path

import pytest
from qwen_native_state import validate_restart_contract, ready_barrier, compare_manifests


def fixture(root):
    ranks=[]
    for rank in range(4):
        names=[f'rank_{rank}/{n}' for n in ['control-checkpoint.json','state-checkpoint.json','input_ids.npy','resolved_config.json']]
        names.append(f'training/checkpoint/rank_{rank}/qwen3_rank_{rank}-8_1.safetensors')
        artifacts={}
        for name in names:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'data')
            artifacts[name]=dict(bytes=4,sha256=hashlib.sha256(b'data').hexdigest())
        ranks.append(dict(rank=rank,source_pid=2147483647,artifacts=artifacts))
    data=dict(schema_version=1,checkpoint_step=8,lr_horizon=32,topology=dict(tp=4,dp=1,pp=1),ranks=ranks)
    return data


@pytest.mark.parametrize('fault',['none','missing_rank','duplicate_rank','wrong_step','wrong_topology','corrupt_shard','empty_artifacts','live_source'])
def test_restart_preflight(tmp_path,fault):
    data=fixture(tmp_path)
    if fault=='missing_rank':data['ranks'].pop()
    if fault=='duplicate_rank':data['ranks'][3]['rank']=0
    if fault=='wrong_step':data['checkpoint_step']=7
    if fault=='wrong_topology':data['topology']['tp']=2
    if fault=='empty_artifacts':data['ranks'][0]['artifacts']={}
    if fault=='live_source':data['ranks'][0]['source_pid']=os.getpid()
    if fault=='corrupt_shard':(tmp_path/'training/checkpoint/rank_0/qwen3_rank_0-8_1.safetensors').write_bytes(b'evil')
    (tmp_path/'restart_contract.json').write_text(json.dumps(data))
    if fault=='none':validate_restart_contract(tmp_path,0,8,32)
    else:
        with pytest.raises(ValueError):validate_restart_contract(tmp_path,0,8,32)


def test_missing_ready_rank_times_out(tmp_path):
    with pytest.raises(TimeoutError):ready_barrier(tmp_path,0,timeout=0)


def test_invalid_ready_record_cannot_release_barrier(tmp_path):
    for r in range(1,4):(tmp_path/f'ready-rank-{r}.json').write_text(json.dumps(dict(rank=r,ready='true')))
    with pytest.raises(ValueError):ready_barrier(tmp_path,0,timeout=0)


def test_state_comparison_requires_exact_names_and_bytes():
    with pytest.raises(ValueError):compare_manifests({'x':dict(sha256='a')},{'x':dict(sha256='b')})
    with pytest.raises(ValueError):compare_manifests({'x':{}},{})


@pytest.mark.parametrize('fault', ['none', 'optimizer', 'deterministic', 'seed', 'model', 'sink', 'horizon'])
def test_training_identity_checked_before_communication(tmp_path, fault):
    import copy
    from qwen_native_state import validate_training_identity
    config = dict(seed=42, model={'model_config': {'seq_length': 128}},
                  parallel_config=dict(model_parallel=4), optimizer=dict(type='AdamW', eps=1e-8),
                  lr_schedule=dict(total_steps=32), context=dict(device_id=0, deterministic='ON'),
                  runner_config=dict(sink_mode=True, sink_size=1, stop_step=11))
    rows = []
    for rank in range(4):
        directory = tmp_path/f'rank_{rank}';directory.mkdir()
        (directory/'resolved_config.json').write_text(json.dumps(config))
        rows.append(dict(rank=rank, config_sha256='original'))
    (tmp_path/'restart_contract.json').write_text(json.dumps(dict(ranks=rows)))
    target = copy.deepcopy(config);model_hash = 'original'
    if fault == 'optimizer': target['optimizer']['eps'] = 1e-5
    if fault == 'deterministic': target['context']['deterministic'] = 'OFF'
    if fault == 'seed': target['seed'] = 43
    if fault == 'model': model_hash = 'wrong'
    if fault == 'sink': target['runner_config']['sink_size'] = 2
    if fault == 'horizon': target['lr_schedule']['total_steps'] = 11
    if fault == 'none': validate_training_identity(tmp_path, target, model_hash)
    else:
        with pytest.raises(ValueError, match='identity differs'):
            validate_training_identity(tmp_path, target, model_hash)


@pytest.mark.parametrize('fault', ['missing_rank', 'duplicate_rank', 'wrong_rank', 'wrong_step', 'wrong_topology', 'corrupt_shard'])
def test_real_worker_rejects_restart_before_framework_import(tmp_path, fault):
    import subprocess
    import sys
    source = tmp_path/'source';source.mkdir()
    data = fixture(source)
    if fault == 'missing_rank': data['ranks'].pop()
    if fault == 'duplicate_rank': data['ranks'][3]['rank'] = 0
    if fault == 'wrong_rank': data['ranks'][3]['rank'] = 4
    if fault == 'wrong_step': data['checkpoint_step'] = 7
    if fault == 'wrong_topology': data['topology']['tp'] = 2
    if fault == 'corrupt_shard':
        (source/'training/checkpoint/rank_0/qwen3_rank_0-8_1.safetensors').write_bytes(b'evil')
    (source/'restart_contract.json').write_text(json.dumps(data))
    output = tmp_path/'output'
    entry = Path(__file__).resolve().parents[2]/'experiments/training/train_qwen3_full_restart.py'
    result = subprocess.run([sys.executable, str(entry), '--output', str(output),
                             '--resume-run', str(source), '--checkpoint-step', '8',
                             '--source-stop-step', '11', '--lr-horizon', '32'],
                            env=dict(os.environ, RANK_ID='0'), timeout=10,
                            capture_output=True, text=True)
    report = json.loads((output/'rank_0/acceptance.json').read_text())
    print(json.dumps(dict(fault=fault, returncode=result.returncode, error=report.get('error'),
                         ready_files=[str(p) for p in output.glob('ready-rank-*')],
                         communication=report['phases']['communication'])))
    assert result.returncode == 1 and report['failed_stage'] == 'config'
    assert report['phases']['communication'] == 'not_run'
    assert not list(output.glob('ready-rank-*'))
    assert (output/'failed-rank-0.json').is_file()
