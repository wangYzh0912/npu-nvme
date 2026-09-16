import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from npu_nvme.d2.backend import RegisteredBackend, registration_from_config
from tools.qwen_resource_plan import plan

ROOT=Path(__file__).resolve().parents[2]


def test_registered_formal_extent_accepts_exact_end_and_rejects_escape():
    reg=registration_from_config(ROOT/'config/d2_qwen_region.json')
    writes=[]
    transport=SimpleNamespace(total_bytes=3840755982336,capabilities=SimpleNamespace(chunk_size=4<<20),write=lambda o,d:writes.append((o,len(d))))
    backend=RegisteredBackend(transport,reg,region_id=reg['region_id'])
    assert backend.base==256<<30 and backend.end==1280<<30
    backend.write(backend.end-4096,b'x'*4096)
    for offset in (backend.base-4096,backend.end):
        with pytest.raises(ValueError):backend.write(offset,b'x'*4096)
    assert writes==[(backend.end-4096,4096)]


def test_socket_budget_counts_private_rank_pools_and_one_pin():
    schema=json.loads((ROOT/'results/long-term-v1.3/EN/state_schema-002.json').read_text())
    args=dict(chunk=4<<20,depth=4,mode='blocking',hbm_free=8<<30,host_free=1<<40,hbm_headroom=1<<30,host_headroom=16<<30)
    result=plan(schema,**args)
    assert result['rank_pools_bytes']+result['owner_pool_bytes']==80<<20
    assert result['aggregate_host_required']>80<<20
    assert result['disk']['admitted']
    with pytest.raises(ValueError):plan(schema,active_pins=2,**args)
    args['host_free']=16<<30
    assert not plan(schema,**args)['aggregate_host_admitted']
