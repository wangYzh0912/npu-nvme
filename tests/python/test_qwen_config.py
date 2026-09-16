import json
from pathlib import Path

import pytest

from npu_nvme.runtime.qwen_config import resolve


def fixture(tmp_path):
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text(json.dumps(dict(model_type='qwen3',hidden_size=4096,num_hidden_layers=36)))
    strategy=Path(__file__).resolve().parents[2]/'config/qwen_runtime_schema.json'
    manifest=tmp_path/'environment.json';manifest.write_text('{}')
    return dict(schema_version=2,model=str(model),environment_manifest=str(manifest),method='none',
        stop_step=8,lr_horizon=8,seq_length=128,checkpoint_interval=4,retention=3,
        timeout_seconds=10,operation_timeout_seconds=5,copy_timeout_ms=1,shm_base=1,strategy=str(strategy),
        hardware_lock=str(tmp_path/'lock'))


def test_identity_allows_continuation_stop_change(tmp_path):
    value=fixture(tmp_path);first=resolve(value,tmp_path)
    value['stop_step']=24;value['lr_horizon']=24;second=resolve(value,tmp_path)
    first['identity'].pop('lr_horizon');second['identity'].pop('lr_horizon')
    assert first['identity']==second['identity']


def test_ours_requires_actual_three_generation_media(tmp_path):
    value=fixture(tmp_path);value.update(method='ours',retention=2)
    with pytest.raises(ValueError,match='physical retention'):
        resolve(value,tmp_path)


@pytest.mark.parametrize('key,value', [('seq_length',7),('operation_timeout_seconds',11),('stop_step',9)])
def test_boundaries_are_rejected(tmp_path,key,value):
    row=fixture(tmp_path);row[key]=value
    with pytest.raises(ValueError):resolve(row,tmp_path)
