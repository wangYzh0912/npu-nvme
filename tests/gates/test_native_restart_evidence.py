"""Audit preserved real Native runs; these tests do not execute or simulate training."""
import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[2]/'results/long-term-v1.3/EN'
SOURCE=ROOT/'source-002'
RESTORE=ROOT/'restore-003'


def read(path):
    return json.loads(path.read_text())


def test_recorded_runs_have_exited_distinct_processes_and_exact_source_identity():
    before=read(SOURCE/'process_exit.json');after=read(RESTORE/'process_exit.json')
    for value in (before,after):
        assert value['launcher_exit_code']==0 and value['all_rank_processes_exited'] is True
        assert len(set(value['rank_pids']))==4
    assert not set(before['rank_pids']) & set(after['rank_pids'])
    assert read(RESTORE/'acceptance.json')['status']=='native_restart_and_continuation_pass'
    for run in (SOURCE,RESTORE):
        manifest=read(run/'run_manifest.json')
        for name in ['experiments/training/train_qwen3_full_restart.py','python/qwen_native_state.py',
                     'scripts/run_qwen3_four_rank.sh','config/qwen_native_continuation.json']:
            assert hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==manifest['source_sha256'][name]
    index=read(ROOT/'artifacts.json')
    for name,item in index.items():
        data=(ROOT/name).read_bytes()
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],name


@pytest.mark.parametrize('rank',range(4))
def test_recorded_rank_exact_state_controls_and_three_step_oracle(rank):
    source=SOURCE/f'rank_{rank}';restore=RESTORE/f'rank_{rank}'
    expected=read(source/'state-checkpoint.json')
    assert expected and expected==read(restore/'restored-state.json')
    control=read(source/'control-checkpoint.json')
    assert control==read(restore/'restored-control.json')
    assert control['logical_optimizer_step']==control['next_data_row']==8
    assert control['lr_horizon']==32
    for name in ('python_rng','numpy_rng','mindspore_rng'):assert control[name]
    assert read(source/'state-final.json')==read(restore/'state-final.json')
    final=read(source/'control-final.json')
    assert final==read(restore/'control-final.json')
    assert final['logical_optimizer_step']==final['next_data_row']==11
    original=read(source/'acceptance.json');actual=read(restore/'acceptance.json')
    assert original['status']=='training_pass_restart_not_tested'
    assert actual['status']=='restored_and_continued'
    assert original['deterministic'] is actual['deterministic'] is True
    assert original['data_sha256']==actual['data_sha256']
    assert original['parallel']==actual['parallel']==dict(tensor_parallel=4,data_parallel=1,pipeline_parallel=1)
    assert original['environment_id']==actual['environment_id']
    expected=[x for x in original['losses'] if x['step']>8]
    assert len(expected)==len(actual['losses'])==3
    for left,right in zip(expected,actual['losses']):
        assert left['step']==right['step'] and left['loss']==right['loss']
        assert math.isfinite(right['loss']) and left['overflow'] is right['overflow'] is False
    ready=read(RESTORE/f'ready-rank-{rank}.json')
    assert ready==dict(rank=rank,ready=True)
    assert actual['restore_seconds']>0
