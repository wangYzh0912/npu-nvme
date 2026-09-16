import pytest
from tools.validate_qwen_baselines import validate,METHODS

def test_primary_requires_all_four_methods():
    with pytest.raises(ValueError,match='four methods'):validate({'methods':{'ours':[]}})

def test_primary_rejects_one_pilot_per_method():
    with pytest.raises(ValueError,match='three independent'):validate({'methods':{name:[{}] for name in METHODS}})


def test_formal_campaign_has_independent_sources_and_timing(tmp_path,monkeypatch):
    import json
    from npu_nvme.runtime import training_benchmark as benchmark
    from npu_nvme.runtime.training_catalog import write_checked
    calls=[]
    def fit(config,path,**kwargs):
        calls.append((path.name,dict(config),kwargs))
        path.mkdir()
        for rank in range(4):
            write_checked(path/f'rank_{rank}/training.json',dict(
                restore_global_ready_ns=200+rank,begin_ns=100+rank))
        return 0
    monkeypatch.setattr('npu_nvme.cli.qwen_training.run_fit',fit)
    monkeypatch.setattr('npu_nvme.runtime.payload_dedup.deduplicate_completed',lambda *args:None)
    monkeypatch.setattr(benchmark,'compare',lambda *args:dict(status='pass'))
    assert benchmark.campaign(dict(identity={}),tmp_path/'campaign')==0
    result=json.loads((tmp_path/'campaign/result.json').read_text())
    assert len(calls)==37 and len(result['comparisons'])==35
    for method in ('mindspore_native_save','ours','bytecheckpoint_host'):
        sources=[row for row in calls if row[0].startswith(f'repeat-{method}-source-')]
        restores=[row for row in calls if row[0].startswith(f'repeat-{method}-restore-')]
        assert len(sources)==3 and len(restores)==4
        assert all(row[2]['resume'] and row[1]['stop_step']==11 for row in restores)
        assert result['timings'][method]['samples_seconds']==[103e-9]*3
        explicit=next(row for row in calls if row[0]==method+'-explicit4')
        assert explicit[2]['restore']=='1' and explicit[1]['checkpoint_interval']==24


def test_native_monitor_keeps_periodic_save_without_forced_end_save():
    from qwen_native_state import fixed_step_checkpoint_monitor
    class Upstream:
        def __init__(self, **kwargs):self.events=[]
        def on_train_step_end(self, context):self.events.append('periodic')
        def on_train_end(self, context):self.events.append('extra_final')
    monitor=fixed_step_checkpoint_monitor(Upstream)(async_save=False)
    monitor.on_train_step_end(None);monitor.on_train_end(None)
    assert monitor.events==['periodic']
    with pytest.raises(ValueError,match='synchronous'):fixed_step_checkpoint_monitor(Upstream)(async_save=True)
