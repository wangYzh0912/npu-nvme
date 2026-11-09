"""Reduce exported profiler tables into phase-one resource evidence."""
import csv
import json
import math
import statistics
from pathlib import Path

from npu_nvme.experiments.timeline import overlap


def analyse(profile_root, report, output):
    profile_root=Path(profile_root); output=Path(output)
    table=next(profile_root.rglob('op_summary*.csv'))
    anchor=json.loads(next(profile_root.rglob('start_info')).read_text())
    wall_us=float(anchor['collectionTimeBegin'])
    raw_us=float(anchor['clockMonotonicRaw'])/1000
    wall_to_monotonic=wall_us-raw_us
    intervals=[];raw_rows=[];get_next=[];counters={key:[] for key in ('aic_mac_ratio','aiv_vec_ratio','cube_utilization(%)')}
    with table.open(newline='') as stream:
        for row in csv.DictReader(stream):
            kind=row['Task Type'];start=float(row['Task Start Time(us)'].strip())
            duration=float(row['Task Duration(us)'])
            raw_rows.append((kind,start,duration,int(row['Mix Block Dim']) if row.get('Mix Block Dim','').isdigit() else 0))
            if row['OP Type']=='GetNext':get_next.append(start)
            for key,values in counters.items():
                try:value=float(row[key])
                except (KeyError,ValueError):continue
                if math.isfinite(value):values.append(value)
    get_next=sorted(set(get_next))
    if len(get_next)!=20:
        raise ValueError('profile does not contain exactly 20 formal GetNext boundaries')
    step_bounds=list(zip(get_next[:-1],get_next[1:]))
    begin,end=step_bounds[0][0],step_bounds[-1][1]
    instruction={'aicore_tasks':0,'aivector_tasks':0,'mixed_tasks':0}
    for kind,start,duration,mix_blocks in raw_rows:
            if start+duration<=begin or start>=end:continue
            engines=[]
            if kind.startswith('MIX_') and mix_blocks>0:
                engines=['cube','vector']
            else:
                if kind in ('AI_CORE','MIX_AIC'):engines.append('cube')
                if kind in ('AI_VECTOR_CORE','MIX_AIV'):engines.append('vector')
            if kind.startswith('MIX_'):instruction['mixed_tasks']+=1
            elif kind=='AI_CORE':instruction['aicore_tasks']+=1
            elif kind=='AI_VECTOR_CORE':instruction['aivector_tasks']+=1
            for engine in engines:intervals.append(dict(engine=engine,start=start,end=start+duration))
    result=overlap(intervals,step_start=begin,step_end=end)
    result['units']='microseconds';result['clock_alignment']=dict(source='same op_summary wall-clock domain; no callback clock conversion',callback_alignment=None)
    result['task_counts']=instruction
    result['instruction_activity']={key:dict(samples=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values))
                                   for key,values in counters.items() if values}
    result['step_boundaries']=step_bounds
    details=next(profile_root.rglob('kernel_details.csv'))
    phases=[]
    with details.open(newline='') as stream:
        for row in csv.DictReader(stream):
            name=row['Name']
            phase=('optimizer' if 'optimizer-' in name else
                   'backward' if 'Gradients/' in name else
                   'forward' if 'network-' in name else 'other')
            start=float(row['Start Time(us)']);duration=float(row['Duration(us)'])
            if start+duration>begin and start<end:
                phases.append(dict(phase=phase,start=start,end=start+duration))
    result['phase_tasks']=phases
    result['per_step']=[]
    for index,(left,right) in enumerate(step_bounds,1):
        relevant=[r for r in intervals if r['start']<right and r['end']>left]
        optimizer=[r for r in phases if r['phase']=='optimizer' and r['start']<right and r['end']>left]
        ready=max((r['end'] for r in optimizer),default=None)
        row=overlap(relevant,step_start=left,step_end=right,ready_time=ready)
        row.pop('segments');row['step']=index;row['last_optimizer_task_end']=ready
        row['ready_rule']='last optimizer-named kernel end; excludes possible later unnamed update tasks'
        result['per_step'].append(row)
    lengths=sorted(r['duration'] for r in result['candidate_windows'])
    result['candidate_length_distribution_us']={str(q):lengths[min(len(lengths)-1,int(q*(len(lengths)-1)))]
                                               for q in (0,.25,.5,.75,.95,1)} if lengths else {}
    result['vector_task_active_fraction_during_cube']=(result['durations']['both']/
        max(1e-12,result['durations']['cube_only']+result['durations']['both']))
    result['exact_instruction_overlap']=None
    result['classification']='MIX tasks with nonzero Mix Block Dim contribute to both engine task envelopes'
    result['scope']='19 complete GetNext-to-GetNext intervals; twentieth end not inferred'
    hbm=[]
    for file in profile_root.rglob('hbm*.csv'):
        with file.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        if rows and 'Metric' in rows[0]:
            hbm=[row for row in rows if row['Metric'] in ('Average','Maximum','Minimum')]
            break
    result['hbm_summary']=hbm or None
    result['hbm_window_samples']=None
    result['limitations']=['Task intervals do not measure active core count.',
                           'Mixed tasks with nonzero Mix Block Dim count in both task envelopes; this does not prove simultaneous instructions.',
                           'Exported HBM CSV has run aggregates; candidate-window bandwidth is unavailable.']
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return result
