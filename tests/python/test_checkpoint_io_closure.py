"""CPU closure gates: missing evidence must never become an acceptance pass."""
import json
from types import SimpleNamespace

import pytest

from npu_nvme.experiments.checkpoint_io import FIELDS
from npu_nvme.experiments.checkpoint_io_closure import (Boundary,boundary_coverage,
    complete_timeline,overhead_gate,parse_hbm,native_snapshot,verify_drained,lifecycle_gate)
from npu_nvme.experiments.checkpoint_io_closure import device_summary,dependency_bound
from npu_nvme.experiments.checkpoint_io_closure import process_tree
from tools.checkpoint_io_closure import schedule
from tools.checkpoint_io import archive_run


def test_boundary_records_owning_intervals_and_rejects_missing_stages():
    ticks=iter((1,2,101,102));boundary=Boundary(4,clock=lambda:next(ticks))
    with boundary.stage('manifest') as stage:stage['aggregate']={'logical_bytes':4}
    row=boundary.finish()
    assert row['stages'][0]['aggregate']['logical_bytes']==4
    assert boundary_coverage(row,{'manifest'})['status']=='failed'
    with pytest.raises(ValueError,match='missing'):boundary_coverage(row,{'sync'})


def test_boundary_uses_union_not_sum_for_overlap():
    row=dict(step=4,begin_ns=1,end_ns=101,stages=[
        dict(name='a',begin_ns=1,end_ns=61),dict(name='b',begin_ns=41,end_ns=81)])
    result=boundary_coverage(row,{'a','b'})
    assert result['stage_union_seconds']==80/1e9 and result['status']=='failed'


def sink_evidence():
    events=[];reports=[]
    for rank in range(4):
        def event(name,timestamp,chunk=None):
            row=dict.fromkeys(FIELDS)
            row.update(event=name,timestamp_ns=timestamp,step=4,rank=rank,physical_npu=rank+4,
                role='rank',raw_request_id=None if name.startswith('capture_setup') else 'epoch:0',
                chunk=chunk)
            events.append(row)
        names=('admission_collectives','framework_synchronize','manifest_read_hash',
               'control_capture','sidecar_write','save_setup','capture_transport')
        stages=[dict(name=name,begin_ns=1000+i*1000,end_ns=2000+i*1000) for i,name in enumerate(names)]
        stages[-1]['end_ns']=9000
        stages[2]['aggregate']=dict(logical_bytes=8,tensors=1,read_ns=500,hash_ns=300)
        reports.append(dict(rank=rank,boundaries=[dict(step=4,operation='save',begin_ns=1000,end_ns=9000,stages=stages)]))
        for name,timestamp in (('training_step',999),('capture_setup_begin',7000),
            ('capture_setup_end',7100),('capture_begin',7100),('capture_end',8990),('training_resume',9000)):
            event(name,timestamp)
        for name,timestamp in (('source_read_begin',7100),('data_ready',7200),
                              ('sink_consumed_source_safe',8900),('source_released',8950)):
            event(name,timestamp,0)
    return events,reports


def test_full_boundary_requires_events_even_when_both_endpoints_absent():
    events,reports=sink_evidence()
    assert complete_timeline(events,reports,'checkpoint_io_sink',[4])['status']=='pass'
    events=[row for row in events if row['event'] not in ('capture_setup_begin','capture_setup_end')]
    with pytest.raises(ValueError,match='required critical-path pair'):
        complete_timeline(events,reports,'checkpoint_io_sink',[4])


def test_full_boundary_does_not_accept_missing_resume_or_chunk_release():
    events,reports=sink_evidence()
    events=[row for row in events if not (row['rank']==0 and row['event']=='training_resume')]
    assert complete_timeline(events,reports,'checkpoint_io_sink',[4])['status']=='failed'
    events,reports=sink_evidence()
    events=[row for row in events if row['event']!='source_released']
    with pytest.raises(ValueError,match='required critical-path pair'):
        complete_timeline(events,reports,'checkpoint_io_sink',[4])


def test_umbrella_transport_stage_does_not_hide_unexplained_critical_path():
    events,reports=sink_evidence()
    for row in events:
        if row['event']=='source_released':row['timestamp_ns']=8000
    result=complete_timeline(events,reports,'checkpoint_io_sink',[4])
    assert all(v['unexplained_fraction']==0 for v in result['full_boundaries'])
    assert result['status']=='failed'


def matrix():
    result=[]
    for spec in schedule():
        ranks=[]
        for rank in range(4):
            losses=[dict(step=step,begin_ns=step*10**10,end_ns=step*10**10+10**9,
                resume_ns=step*10**10+2*10**9) for step in range(28,65)]
            saves=[] if spec['method']=='none' else [dict(step=step,blocking_seconds=1) for step in range(28,65,4)]
            ranks.append(dict(rank=rank,loss_steps=losses,checkpoints=saves))
        result.append(dict(name=spec['name'],group=spec['method'],mode=spec['mode'],
                           replicate=spec['replicate'],ranks=ranks))
    return result


def test_matrix_preregisters_rotating_methods_and_alternating_ab_order():
    rows=schedule()
    assert len(rows)==18 and len({row['name'] for row in rows})==18
    assert [rows[i]['method'] for i in range(0,18,2)]==[
        'none','checkpoint_io_sink','ours','checkpoint_io_sink','ours','none','ours','none','checkpoint_io_sink']
    for method in ('none','checkpoint_io_sink','ours'):
        assert [next(row['mode'] for row in rows if row['method']==method and row['replicate']==rep)
                for rep in range(3)] in (['A','B','A'],['B','A','B'])


def test_overhead_requires_18_processes_and_reports_30_sample_p95():
    rows=matrix();gate=overhead_gate(rows)
    assert gate['status']=='pass' and len(gate['pairs'])==9
    assert all(v['modes']['B']['n']==30 and v['p95_ratio']==1 for v in gate['groups'])
    with pytest.raises(ValueError,match='eighteen'):overhead_gate(rows[:-1])
    with pytest.raises(ValueError,match='duplicate'):overhead_gate(rows+[rows[0]])


def test_one_failed_process_pair_stops_overhead_acceptance():
    rows=matrix();run=next(v for v in rows if v['group']=='ours' and v['mode']=='B' and v['replicate']==1)
    for rank in run['ranks']:
        for row in rank['loss_steps']:row['resume_ns']+=10**9
    gate=overhead_gate(rows)
    assert gate['status']=='failed'
    assert next(v for v in gate['pairs'] if v['method']=='ours' and v['replicate']==1)['median_ratio']==1.5


def test_hbm_parser_preserves_physical_mapping_and_rejects_missing_devices():
    text=chr(10).join(f'| {device} 910B3 | OK |'+chr(10)+
        f'| 0 | 0000:41:00.0 | 0 0 / 0 {3400+device} / 65536 |' for device in range(8))
    result=parse_hbm(text)
    assert result['6']==dict(used_mib=3406,total_mib=65536)
    with pytest.raises(ValueError,match='4..7'):parse_hbm(text.replace('7 910B3','8 910B3'))


def test_process_tree_includes_workers_in_independent_sessions(tmp_path):
    for pid,parent,group in ((100,1,100),(101,100,101),(102,100,102),(103,101,101),(200,1,200)):
        directory=tmp_path/str(pid);directory.mkdir()
        fields=['S',str(parent),str(group),str(group)]+['0']*15+[str(pid*1000)]
        (directory/'stat').write_text(str(pid)+' (python worker) '+' '.join(fields))
    assert process_tree([100],proc_root=tmp_path)=={100:100000,101:101000,102:102000,103:103000}


def test_process_tree_ignores_exited_processes_but_rejects_invalid_records(tmp_path):
    directory=tmp_path/'100';directory.mkdir()
    fields=['Z','1','100','100']+['0']*15+['1000']
    path=directory/'stat';path.write_text('100 (python) '+' '.join(fields))
    assert process_tree([100],proc_root=tmp_path)=={}
    path.write_text('invalid')
    with pytest.raises(IndexError):process_tree([100],proc_root=tmp_path)


def test_native_counter_scope_and_errors_are_explicit():
    def read(ctx,stats):return 0
    transport=SimpleNamespace(ctx=1,retained=[],lib=SimpleNamespace(npu_nvme_get_stats=read))
    row=native_snapshot(transport);verify_drained(row)
    assert 'queued writes only' in row['scopes']['request_ring_peak']
    row['counters']['nvme_submit_count']=1
    with pytest.raises(ValueError,match='not drained'):verify_drained(row)
    transport.lib.npu_nvme_get_stats=lambda *args:-5
    with pytest.raises(RuntimeError,match='-5'):native_snapshot(transport)


def test_lifecycle_gate_rejects_empty_success_and_retained_children(tmp_path):
    (tmp_path/'result.json').write_text(json.dumps(dict(validation_status='pass')))
    with pytest.raises(ValueError,match='safe close'):lifecycle_gate(tmp_path,detailed=False)
    (tmp_path/'result.json').write_text(json.dumps(dict(validation_status='pass',
        end_to_end=dict(children_remaining=[1],retained=[],lease_released=True))))
    with pytest.raises(ValueError,match='safe close'):lifecycle_gate(tmp_path,detailed=False)


def test_rank_protocol_close_failure_is_not_reported_as_success(tmp_path):
    from npu_nvme.framework.training_checkpoint import TrainingCheckpoint
    session=TrainingCheckpoint({},rank=0,npu_id=0,output=tmp_path)
    def fail():raise OSError('owner close ACK differs')
    closed=[]
    session.client=SimpleNamespace(close=fail)
    session.executor=SimpleNamespace(close=lambda:closed.append(True) or True)
    with pytest.raises(OSError,match='ACK'):session.close()
    result=json.loads((tmp_path/'d2-session-status.json').read_text())
    assert result['status']=='fail' and result['closed'] and closed==[True]


def test_device_exports_require_real_task_rows_and_positive_trace_intervals(tmp_path):
    (tmp_path/'trace_view.json').write_text(json.dumps([dict(ph='X',dur=1,cat='kernel')]))
    table=tmp_path/'kernel_details.csv'
    table.write_text(chr(10).join(('Name,Accelerator Core,Duration(us),mac_ratio',
                                 'MatMul,AI_CORE,2,0.4','')))
    result=device_summary(tmp_path)
    assert result['positive_duration_device_tasks']==1
    assert result['observed_pmu_columns']==['mac_ratio']
    table.write_text('Name,Duration(us)'+chr(10)+'host,2'+chr(10))
    with pytest.raises(ValueError,match='device task'):device_summary(tmp_path)


def test_archive_defers_profiler_payloads_for_separate_compression(tmp_path):
    import gzip
    source=tmp_path/'run';profile=source/'rank_0/device-profile';profile.mkdir(parents=True)
    (source/'result.json').write_text('{}')
    (source/'event.jsonl').write_text('{}'+chr(10))
    (profile/'trace.json').write_text('[]')
    target=tmp_path/'archive'
    archive_run(source,target,exclude=('rank_0/device-profile',))
    assert (target/'result.json').read_text()=='{}'
    with gzip.open(target/'event.jsonl.gz','rt') as stream:assert stream.read()=='{}'+chr(10)
    assert not (target/'rank_0/device-profile/trace.json').exists()


def test_dependency_bound_requires_full_target_step_coverage_and_save_before_resume():
    reports=[]
    for rank in range(4):
        reports.append(dict(rank=rank,method='ours',
            device_profile=dict(start_anchor={'monotonic_ns':1},stop_anchor={'monotonic_ns':100}),
            losses=[dict(step=step,begin_ns=begin,end_ns=end,resume_ns=resume) for step,begin,end,resume in
                    ((28,10,20,30),(29,31,40,40),(32,50,60,70))],
            boundaries=[dict(step=28,begin_ns=21,end_ns=29),dict(step=32,begin_ns=61,end_ns=69)]))
    assert dependency_bound(reports)['status']=='verified'
    reports[0]['losses'][1]['begin_ns']=28
    with pytest.raises(ValueError,match='before save returns'):dependency_bound(reports)
