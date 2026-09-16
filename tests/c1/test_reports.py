import copy
import json
from pathlib import Path
import pytest
from npu_nvme.cli.contracts import validate_restore_report


@pytest.fixture
def reports():
    h='a'*64
    source=dict(status='trend_measured',adapter='ours',source_pid=1,checkpoints=[dict(generation=3,step=8,sha256=h)])
    restored=dict(status='pass',byte_exact=True,verification_performed=True,mandatory_integrity=True,controls_verified=True,
        restored_losses=[1,2,3],source_oracle_losses=[1,2,3],source_pid=1,restore_pid=2,generation=3,checkpoint_step=8,
        expected_state_sha256=h,applied_state_sha256=h)
    return source,restored


@pytest.mark.parametrize('field,value',[
    ('status','fail'),('byte_exact','yes'),('mandatory_integrity',False),('controls_verified',1),
    ('restore_pid',1),('source_pid',5),('generation',4),('checkpoint_step',9),
    ('applied_state_sha256','b'*64),('expected_state_sha256',None),('source_oracle_losses',[]),
    ('restored_losses',[float('nan'),2,3]),
])
def test_restore_report_fails_closed(reports,field,value):
    source,restored=reports;restored[field]=value
    with pytest.raises(ValueError):validate_restore_report(restored,source,3)


def test_valid_restore_report(reports):
    source,restored=reports;validate_restore_report(restored,source,3)


def test_prune_only_own_completed_generations(tmp_path):
    from experiments.baselines.repro.adapters.base import Adapter
    a=Adapter({'fs_test_dir':str(tmp_path)},tmp_path/'run');a.name='mindspore_native_save'
    own=tmp_path/'repro_checkpoints/run';other=tmp_path/'repro_checkpoints/other/generation_000001'
    other.mkdir(parents=True)
    for n in range(1,5):(own/f'generation_{n:06d}').mkdir(parents=True)
    a.prune(2)
    assert {p.name for p in own.iterdir()}=={'generation_000003','generation_000004'}
    assert other.exists()
