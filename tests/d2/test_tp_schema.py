import copy
import json
from pathlib import Path
import pytest
from npu_nvme.d2.tp_schema import validate,validate_rank,DTYPES


def schema():
    return json.loads((Path(__file__).resolve().parents[2]/'results/long-term-v1.3/EN/state_schema-002.json').read_text())

def test_actual_tp_schema_and_local_geometry():
    s=schema();assert validate(s)
    rows=[dict(rank=0,name=r['name'],shape=r['local_shape'],dtype=DTYPES[r['dtype']],partition=r['partition'],bytes=r['logical_bytes_per_rank']) for r in s['tensors']]
    validate_rank(rows,s,0)
    rows[0]['shape']=[1]
    with pytest.raises(ValueError,match='geometry'):validate_rank(rows,s,0)

def test_overlapping_shards_are_rejected():
    s=schema();r=next(r for r in s['tensors'] if r['partition']=='sharded')
    r['shards'][1]['start']=r['shards'][0]['start'][:];r['shards'][1]['end']=r['shards'][0]['end'][:]
    with pytest.raises(ValueError,match='overlapping'):validate(s)

def test_missing_rank_rejected():
    s=schema();s['tensors'][0]['shards'].pop()
    with pytest.raises(ValueError,match='ranks'):validate(s)

def test_optimizer_mapping_rejected():
    s=schema();s['tensors'][0]['model_parameter']='missing'
    with pytest.raises(ValueError,match='optimizer'):validate(s)

def test_runtime_schema_excludes_only_explicit_native_container_fields():
    s=json.loads((Path(__file__).resolve().parents[2]/'config/qwen_runtime_schema.json').read_text())
    rows=[dict(rank=0,name=r['name'],shape=r['local_shape'],dtype=DTYPES[r['dtype']],partition=r['partition'],bytes=r['logical_bytes_per_rank']) for r in s['tensors'] if r.get('placement')!='native_container_metadata']
    assert len(rows)==952
    validate_rank(rows,s,0)
    rows.pop()
    with pytest.raises(ValueError,match='missing'):validate_rank(rows,s,0)
