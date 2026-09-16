from types import SimpleNamespace
import pytest
from npu_nvme.runtime.resources import Allocation,allocation_totals,capture_admission,phase_deadlines,framework_memory_snapshot


def test_shared_host_mapping_counted_once_hbm_per_rank():
    items=[Allocation(str(r),'host',1024,'spdk',r,'pool') for r in range(4)]
    items += [Allocation('snapshot','hbm',4096,'snapshot',r) for r in range(4)]
    totals=allocation_totals(items)['totals']
    assert totals==dict(host=1024,**{f'hbm_rank_{r}':4096 for r in range(4)})
    with pytest.raises(ValueError,match='inconsistent'):
        allocation_totals(items+[Allocation('x','host',2048,'spdk',0,'pool')])


def test_qwen_capture_rejects_hbm_without_implicit_fallback():
    args=dict(state_bytes=24574980144,hbm_free=8*1024**3,host_free=1024**4,
              hbm_headroom=1024**3,host_headroom=16*1024**3,transport_host=1024**3)
    assert not capture_admission(mode='hbm_snapshot',**args)['admitted']
    assert capture_admission(mode='host_snapshot',**args)['admitted']
    assert capture_admission(mode='blocking',**args)['training_blocked_until']=='source_safe'


def test_deadline_uses_slowest_successful_phase_preserves_existing_budget():
    assert phase_deadlines({'compile':7200000,'restore':120000},{'compile':[161.81],'restore':[270,290]})=={'compile':7200000,'restore':870000}
    with pytest.raises(ValueError):phase_deadlines({'compile':1000},{})


def test_framework_reserved_is_not_added_to_allocated():
    hal=SimpleNamespace(**{name:lambda value=value:value for name,value in
        [('memory_allocated',100),('memory_reserved',200),('max_memory_allocated',150),('max_memory_reserved',250)]})
    report=framework_memory_snapshot(SimpleNamespace(hal=hal),rank=0,phase='step')
    assert report['framework']['max_memory_reserved']==250 and report['external']['totals']=={}


def test_actual_qwen_schema_has_no_c1_generation_limit():
    import json
    from pathlib import Path
    from tools.qwen_resource_plan import plan
    root=Path(__file__).resolve().parents[2]
    schema=json.loads((root/'results/long-term-v1.3/EN/state_schema-002.json').read_text())
    expected={1:95520,4:25224,16:7872}
    for mib,count in expected.items():
        result=plan(schema,chunk=mib*1024**2,depth=64,mode='blocking',hbm_free=8*1024**3,
                    host_free=1024**4,hbm_headroom=1024**3,host_headroom=16*1024**3)
        assert result['descriptors_total']==count
        assert result['state_bytes_per_rank']==24574980144
        assert result['tensor_count_per_rank']==882
        assert result['partition_counts']==dict(replicated=435,sharded=438,per_rank_control=9)
        assert result['admission_per_rank']['admitted']
