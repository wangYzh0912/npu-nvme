from types import SimpleNamespace
import json
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
    schema=json.loads((root/'config/qwen_runtime_schema.json').read_text())
    expected={1:95800,4:25504,16:8152}
    for mib,count in expected.items():
        result=plan(schema,chunk=mib*1024**2,depth=64,mode='blocking',hbm_free=8*1024**3,
                    host_free=1024**4,hbm_headroom=1024**3,host_headroom=16*1024**3)
        assert result['descriptors_total']==count
        assert result['state_bytes_per_rank']==24574980708
        assert result['tensor_count_per_rank']==952
        assert result['partition_counts']==dict(replicated=435,sharded=438,per_rank_control=79)
        assert result['admission_per_rank']['admitted']


def test_capacity_rejects_host_disk_and_hbm_shortage_without_fallback(monkeypatch):
    from npu_nvme.runtime import resources
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    config=dict(strategy=str(root/'config/qwen_runtime_schema.json'),
                region=str(root/'config/d2_qwen_region.json'),host_headroom_bytes=16<<30,hbm_headroom_bytes=1<<30)
    monkeypatch.setattr(resources,'available_host_bytes',lambda:0)
    assert not resources.qwen_capacity(config)['admitted']
    monkeypatch.setattr(resources,'available_host_bytes',lambda:1<<50)
    # Pools are resident at the runtime boundary; hugepage capacity is not charged twice.
    resident=5*(4<<20)*4+4*(4<<20)+(1<<20)
    rejected=resources.qwen_capacity(config,hbm_free=0,resident_host=resident)
    assert not rejected['admitted'] and rejected['hugepage_required_remaining']==0
    allowed=resources.qwen_capacity(config,hbm_free=2<<30,resident_host=resident)
    assert allowed['admitted']
    assert allowed['host_required_remaining']==allowed['aggregate_host_required']-resident
    assert allowed['tensor_scratch_budget_bytes']>0
    assert allowed['admission_per_rank']['mode']=='blocking'
    schema=json.loads(Path(config['strategy']).read_text())
    short=resources.qwen_plan(schema,chunk=4<<20,depth=4,mode='blocking',hbm_free=2<<30,
        host_free=1<<50,hbm_headroom=1<<30,host_headroom=16<<30,region_bytes=1<<20)
    assert not short['disk']['admitted']
