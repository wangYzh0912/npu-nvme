import pytest

from tools.aic_kv.training_jit_campaign import conditions


def test_prevalence_matrix_has_six_cells_and_three_process_repeats():
    rows=conditions('prevalence','O1')
    cells={(mode,seq,batch) for mode,seq,batch,_ in rows}
    assert len(rows)==18
    assert cells=={('O1',seq,batch) for seq in (128,512,2048) for batch in (1,2)}
    assert all(sum(row[:3]==cell for row in rows)==3 for cell in cells)


def test_smoke_is_the_first_formal_baseline_replicate():
    assert conditions('o1-smoke','O1')==[('O1',128,1,0)]
    assert conditions('o2-smoke','O2')==[('O2',128,1,0)]
    assert conditions('o0-hbm-confirm','O0')==[('O0',128,1,1)]


@pytest.mark.parametrize('stage,mode', [('prevalence','O0'),('other','O1')])
def test_campaign_rejects_unapproved_matrix_mode(stage,mode):
    with pytest.raises(ValueError):
        conditions(stage,mode)


def test_prevalence_does_not_pass_incomplete_or_invalid_replicates():
    from tools.aic_kv.training_jit_report import recurrence
    rows=[dict(jit_level='O1',physical_device=4,sequence_length=s,micro_batch=b,
               replicate=repeat,model_process_run_id=f'{s}-{b}-{repeat}',
               measurement_accepted=True,joint_fraction=.8,low_aic_fraction=.9) for s in (128,512,2048)
          for b in (1,2) for repeat in range(3)]
    assert recurrence(rows,'O1')[0]['low_aic_high_aiv_covers_matrix']
    assert not recurrence(rows[:-1],'O1')[0]['matrix_complete']
    rows[-1]['measurement_accepted']=False
    assert not recurrence(rows,'O1')[0]['matrix_complete']
    rows[-1]['measurement_accepted']=True
    rows[-1]['joint_fraction']=.2
    assert recurrence(rows,'O1')[0]['matrix_complete']
    assert not recurrence(rows,'O1')[0]['low_aic_high_aiv_covers_matrix']
    assert recurrence(rows,'O1')[0]['low_aic_alone_covers_matrix']
    rows[-1]['model_process_run_id']=rows[-2]['model_process_run_id']
    assert not recurrence(rows,'O1')[0]['matrix_complete']


def test_fragment_timeline_preserves_sub_microsecond_epoch_offsets(tmp_path):
    import gzip
    from tools.aic_kv.training_fragment_report import task_timeline
    path=tmp_path/'tasks.csv.gz'
    with gzip.open(path,'wt') as stream:
        stream.write('kernel_name,kernel_type,stream_id,task_id,task_start(us),task_time(us)\n'
                     'TensorMove1,KERNEL_AIVEC,2,4,1794219991367702.265,1.020\n'
                     'N/A,MEMCPY_ASYNC,2,5,1794219991367703.305,2.000\n')
    result=task_timeline(path)
    assert result['tasks'][1]['start_us']==1.04
    assert result['memcpy_count']==1
    assert result['transfer_payload_bytes'] is None
    assert not result['bulk_transfer_absence_proven']


def test_static_operand_path_respects_graph_identity_and_cycles():
    from tools.aic_kv.training_fragment_audit import operand_path
    nodes={('g',str(i)):dict(graph='g',node=str(i),operands=inputs)
           for i,inputs in [(1,['2','9']),(2,['1','3']),(3,[])]}
    nodes[('other','9')]=dict(graph='other',node='9',operands=['3'])
    assert operand_path(nodes,('g','1'),('g','3'))==[('g','1'),('g','2'),('g','3')]
    assert operand_path(nodes,('g','1'),('other','9')) is None


def test_joint_interval_never_bridges_invalid_bins_or_training_steps():
    from tools.aic_kv.training_jit_report import longest_joint_intervals
    rows=[dict(window_start_ns=i*10,window_end_ns=(i+1)*10,window_id=window,
               valid=valid,aic_activity_pct='0' if valid=='True' else '',
               aiv_activity_pct='90' if valid=='True' else '')
          for i,window,valid in [(0,'a','True'),(1,'a','True'),(2,'a','False'),
                                  (3,'a','True'),(4,'b','True')]]
    assert longest_joint_intervals(rows,['a','b','c'])==dict(a=20,b=10,c=0)


def test_model_tail_includes_matrix_tasks_crossing_its_start_boundary():
    from tools.aic_kv.training_model_fragment_report import summarize
    tasks=[dict(name=name,start_ns=begin,end_ns=end,host_task_type='KERNEL_AIVEC')
           for name,begin,end in [('Kernel::KernelLaunch::Gradients/MatMul',2,5),
              ('Kernel::KernelLaunch::Default/TensorMove-op0/TensorMove',10,15),
              ('Kernel::KernelLaunch::Default/GraphKernel_Mul_Add_Assign_fusion-op0/update',20,30)]]
    native=tasks+[dict(name='crossing',start_ns=8,end_ns=12,host_task_type='KERNEL_AICORE')]
    row=summarize(tasks,native,dict(step=1,begin_ns=0,end_ns=40))
    assert row['tail_aic_task_count']==1
    assert row['tail_task_types']==dict(KERNEL_AIVEC=2,KERNEL_AICORE=1)
    assert row['tensor_move_count']==1


def test_model_task_engine_join_requires_stream_id_task_id_and_timestamp(tmp_path):
    import csv,gzip
    from tools.aic_kv.training_model_fragment_report import joined_tasks
    native=[dict(stream_id='2',task_id='1',kernel_name='a',kernel_type='KERNEL_AIVEC',
                 **{'task_start(us)':'100.000','task_time(us)':'1.000'}),
            dict(stream_id='2',task_id='1',kernel_name='b',kernel_type='KERNEL_AICORE',
                 **{'task_start(us)':'200.000','task_time(us)':'1.000'})]
    scoped=[{'Stream ID':'2','Task ID':'1','Name':'cannot_infer_from_name',
             'Start Time(us)':'200.000','Duration(us)':'1.000','Accelerator Core':'N/A'}]
    for name,rows in [('task_time.csv.gz',native),('kernel_details.csv.gz',scoped)]:
        with gzip.open(tmp_path/name,'wt') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    joined,_=joined_tasks(tmp_path)
    assert joined[0]['host_task_type']=='KERNEL_AICORE'


def test_archive_replay_preserves_native_half_nanosecond_timestamps(tmp_path):
    import csv,gzip,json,shutil,sqlite3
    from tools.aic_kv.training_counter_replay import replay_rank
    from tools.aic_kv.reduce import export_pmu
    from tools.aic_kv.hbm import export as export_hbm
    device=tmp_path/'native';(device/'sqlite').mkdir(parents=True)
    run=tmp_path/'run';rank=run/'rank0';rank.mkdir(parents=True)
    base=2_819_741_645_679_689
    metadata={'info.json.4':{'DeviceInfo':[dict(ai_core_num=1,aiv_num=1,aic_frequency=1000,aiv_frequency=1000)]},
              'start_info.4':dict(collectionTimeBegin=0,clockMonotonicRaw=0),
              'sample.json':dict(aicore_sampling_interval=10,aiv_sampling_interval=10,
                                 ai_core_profiling_mode='sample-based',aiv_profiling_mode='sample-based')}
    for name,value in metadata.items():
        (device/name).write_text(json.dumps(value));shutil.copy2(device/name,rank/name)
    for name in ('aicore.db','ai_vector_core.db'):
        c=sqlite3.connect(device/'sqlite'/name)
        c.execute('create table AICoreOriginalData(timestamp NUMERIC,coreid,task_cyc,replayid)')
        values=[(base+.5+i*10_000_000,0,5_000_000,0) for i in range(5)]
        c.executemany('insert into AICoreOriginalData values(?,?,?,?)',values);c.commit();c.close()
        with gzip.open(rank/(name+'.replay0.csv.gz'),'wt') as stream:
            writer=csv.writer(stream);writer.writerow(['timestamp','coreid','task_cyc','replayid']);writer.writerows(values)
    c=sqlite3.connect(device/'sqlite/hbm.db')
    c.execute('create table HBMOriginalData(device_id,replayid,timestamp,counts,event_type,hbmid)')
    c.executemany('insert into HBMOriginalData values(4,0,?,625,?,?)',
                  [(base+i*20_000_000,event,channel) for i in range(3)
                   for event in ('read','write') for channel in range(4)])
    c.commit();c.close()
    windows=[dict(id='step',begin_ns=base+10_000_000,end_ns=base+30_000_000)]
    (run/'sample-evidence.json').write_text(json.dumps(dict(ranks=[dict(windows=windows)])))
    export_pmu(device,rank/'pmu',windows);export_hbm(device,rank/'hbm',windows)
    (rank/'hbm-clock.json').write_text(json.dumps(dict(offset_ns=0)))
    with gzip.open(rank/'hbm.db.gz','wb') as stream:stream.write((device/'sqlite/hbm.db').read_bytes())
    assert replay_rank(run,0)['passed']
