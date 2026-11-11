from contextlib import contextmanager
import json

import pytest

from npu_nvme.experiments.checkpoint_io import FIELDS,consume_to_host_sink,emit
from npu_nvme.workloads.qwen import records_state
from npu_nvme.experiments.checkpoint_io_analysis import analyse,distribution,probe_overhead
from npu_nvme.experiments.checkpoint_io_events import union_ns,timeline


def test_sink_consumes_each_chunk_before_releasing_its_source():
    schema=[dict(name='b',bytes=5),dict(name='a',bytes=3)]
    released=[];consumed=[]

    @contextmanager
    def read_chunk(name,offset,length):
        data=bytes([len(name)])*length
        try:yield data
        finally:released.append((name,offset))

    def consume(name,offset,length,digest,data):
        assert (name,offset) not in released
        assert len(digest)==64 and len(data)==length
        consumed.append((name,offset,length))

    result=consume_to_host_sink(schema,2,read_chunk,consume)
    assert result==dict(chunks=5,logical_bytes=8)
    assert consumed==[('a',0,2),('a',2,1),('b',0,2),('b',2,2),('b',4,1)]
    assert released==[(name,offset) for name,offset,_ in consumed]


def test_sink_does_not_ack_short_source():
    accepted=[]

    @contextmanager
    def read_chunk(name,offset,length):yield b'x'*(length-1)

    with pytest.raises(ValueError,match='short read'):
        consume_to_host_sink([dict(name='x',bytes=3)],2,read_chunk,
                             lambda *args:accepted.append(args))
    assert accepted==[]


def test_event_record_has_stable_identity_and_timestamp(tmp_path):
    path=tmp_path/'events.jsonl'
    emit(path,event='source_safe_ack',run_id='run-a',rank=1,physical_npu=5,
         step=28,generation=3,raw_request_id='epoch:7',native_request_id=9,
         tensor='weights',chunk=2,buffer_id='pool:0',slot_generation=4,byte_count=4096)
    row=json.loads(path.read_text())
    assert tuple(row)==FIELDS
    assert (row['run_id'],row['rank'],row['physical_npu'],row['step'])==('run-a',1,5,28)
    assert (row['raw_request_id'],row['native_request_id'],row['bytes'])==('epoch:7',9,4096)
    assert row['clock_domain']=='CLOCK_MONOTONIC' and row['timestamp_ns']>0


def test_performance_t0_does_not_remove_acceptance_oracle_states():
    kwargs=dict(step=8,stop=24,period=4,method='none')
    assert records_state(**kwargs,experiment=None)
    assert records_state(**kwargs,experiment={'events':True})
    assert not records_state(**kwargs,experiment={'skip_periodic_state':True})
    assert records_state(step=24,stop=24,period=4,method='none',
                         experiment={'skip_periodic_state':True})
    assert records_state(step=8,stop=24,period=4,method='ours',
                         experiment={'skip_periodic_state':True})


def test_analysis_counts_processes_and_slowest_rank_without_inflating_samples():
    runs=[]
    for rep in range(3):
        for method in ('none','checkpoint_io_sink','ours'):
            ranks=[]
            for rank in range(4):
                rows=[dict(step=step,begin_ns=step*10**10,end_ns=step*10**10+10**9,
                           resume_ns=step*10**10+(rank+1)*10**9) for step in range(28,65)]
                ranks.append(dict(rank=rank,loss_steps=rows,checkpoints=[] if method=='none' else [
                    dict(step=step,blocking_seconds=rank+1) for step in range(28,65,4)]))
            runs.append(dict(name=f'{method}-{rep}',group=method,replicate=rep,ranks=ranks))
    result=analyse(dict(runs=runs))
    assert result['groups']['ours']['processes']==3
    assert result['groups']['ours']['aligned_step_seconds']['n']==30
    assert result['groups']['ours']['aligned_step_seconds']['p95']==4
    assert result['processes'][0]['aligned_step_seconds']['p95'] is None
    with pytest.raises(ValueError,match='three independent'):
        analyse(dict(runs=runs[:-1]))


def test_quantile_gate_rejects_nonfinite_and_small_sample_p95():
    assert distribution(range(1,31))['p95']==29
    assert distribution([1]*29)['p95'] is None
    with pytest.raises(ValueError):distribution([float('nan')])


def test_sink_reuses_one_executor_across_checkpoint_boundaries(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from npu_nvme.framework import training_checkpoint as module
    allocations=[]
    def executor(*args,**kwargs):
        owner=SimpleNamespace(close=lambda:True)
        allocations.append(owner)
        return owner
    monkeypatch.setattr(module,'load_backend',lambda *args:None)
    monkeypatch.setattr(module,'RankCopyExecutor',executor)
    config=dict(sink_only=True,library='test',chunk_bytes=4,rank_shm_base=1,depth=1,copy_timeout_ms=1)
    session=module.TrainingCheckpoint(config,rank=0,npu_id=0,output=tmp_path)
    session._open();session._open()
    assert allocations==[session.executor]
    session.close()


def test_rank_event_uses_physical_device_mapping(monkeypatch,tmp_path):
    from npu_nvme.framework.training_checkpoint import TrainingCheckpoint
    session=TrainingCheckpoint(dict(io_event_root=str(tmp_path),identity={'physical_npu_ids':[4,5,6,7]}),
                               rank=2,npu_id=2,output=tmp_path/'rank')
    session.event('data_ready',step=4)
    row=json.loads((tmp_path/'rank-2.jsonl').read_text())
    assert row['rank']==2 and row['physical_npu']==6


def test_overlapping_event_intervals_are_not_added():
    assert union_ns([(10,30),(20,40),(50,60)])==40
    with pytest.raises(ValueError):union_ns([(30,10)])


def test_timeline_reports_missing_pairs_and_bad_device_mapping():
    events=[]
    for rank in range(4):
        for event,timestamp in [('capture_begin',10),('capture_end',20),('training_resume',30)]:
            events.append(dict(event=event,timestamp_ns=timestamp,step=4,rank=rank,
                physical_npu=rank+4,role='rank',raw_request_id='epoch:0',chunk=None))
    assert timeline(events)['status']=='correlated'
    events.pop(1)
    events[0]['physical_npu']=0
    result=timeline(events)
    assert result['status']=='coverage_incomplete'
    assert result['physical_mapping_error_count']==1
    assert result['checkpoints'][0]['missing_pairs'][0]['stage']=='rank_capture'


def test_probe_overhead_is_a_screen_even_below_ten_percent(tmp_path):
    from npu_nvme.runtime.training_catalog import write_checked
    rows=[dict(step=step,begin_ns=step*10**10,end_ns=step*10**10+10**9,
               resume_ns=step*10**10+10**9) for step in (28,32)]
    runs=[]
    for method in ('none','checkpoint_io_sink','ours'):
        runs.append(dict(group=method,replicate=0,ranks=[dict(loss_steps=rows) for _ in range(4)]))
        for rank in range(4):
            write_checked(tmp_path/method/f'rank_{rank}/training.json',dict(losses=rows))
    result=probe_overhead(dict(runs=runs),tmp_path)
    assert result['status']=='screening_only'
    assert all(method['median_ratio']==1 for method in result['methods'])
