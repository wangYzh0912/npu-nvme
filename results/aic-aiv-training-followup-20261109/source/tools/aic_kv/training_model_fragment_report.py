"""Join model kernel scopes to native engine dispatch and bound optimizer tail."""
import argparse
from collections import Counter,defaultdict
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path

from .training_jit_campaign import RESULTS


def ns(value):return int(Decimal(value.strip())*1000)


def union_ns(tasks,begin,end):
    spans=sorted((max(begin,t['start_ns']),min(end,t['end_ns'])) for t in tasks
                 if t['start_ns']<end and t['end_ns']>begin)
    total=0;last=begin
    for low,high in spans:
        total+=max(0,high-max(low,last));last=max(last,high)
    return total


def joined_tasks(rank):
    native=defaultdict(list);native_tasks=[]
    for row in csv.DictReader(gzip.open(rank/'task_time.csv.gz','rt')):
        native[(row['stream_id'],row['task_id'],ns(row['task_start(us)']))].append(row)
        start=ns(row['task_start(us)']);duration=ns(row['task_time(us)'])
        native_tasks.append(dict(name=row['kernel_name'],start_ns=start,end_ns=start+duration,
                                 host_task_type=row['kernel_type']))
    tasks=[]
    for row in csv.DictReader(gzip.open(rank/'kernel_details.csv.gz','rt')):
        start=ns(row['Start Time(us)']);duration=ns(row['Duration(us)'])
        matches=native.get((row['Stream ID'],row['Task ID'],start),[])
        if len(matches)!=1:
            kind='UNKNOWN_NATIVE_JOIN'
        else:
            match=matches[0]
            if abs(ns(match['task_time(us)'])-duration)>1:
                raise ValueError('native and scoped task durations differ')
            kind=match['kernel_type']
        tasks.append(dict(name=row['Name'],stream=row['Stream ID'],task_id=row['Task ID'],
                          start_ns=start,end_ns=start+duration,host_task_type=kind,
                          kernel_details_core=row['Accelerator Core']))
    return tasks,native_tasks


def summarize(tasks,native_tasks,window):
    begin=window['begin_ns'];end=window['end_ns']
    selected=[t for t in tasks if begin<=t['start_ns']<end]
    copies=[t for t in selected if '::Default/TensorMove-op0/' in t['name']]
    if len(copies)!=1:raise ValueError('expected one explicit first optimizer-state TensorMove-op0')
    start=copies[0]['start_ns']
    tail=[t for t in selected if t['start_ns']>=start]
    finish=max(t['end_ns'] for t in tail)
    gradient=[t for t in selected if 'Gradients/' in t['name']]
    if not gradient:raise ValueError('no scoped backward task')
    backward=min(t['start_ns'] for t in gradient)
    native_tail=[t for t in native_tasks if t['start_ns']<finish and t['end_ns']>start]
    types=Counter(t['host_task_type'] for t in native_tail)
    return dict(step=window['step'],step_begin_ns=begin,step_end_ns=end,
        first_scoped_backward_ns=backward,
        forward_boundary_quality='inferred from first explicit Gradients scope; recomputation remains in backward',
        tail_begin_ns=start,tail_end_ns=finish,tail_envelope_ms=(finish-start)/1e6,
        tail_begin_source=copies[0],tail_scoped_task_count=len(tail),
        tail_task_types=dict(types),
        tensor_move_count=sum('/TensorMove-op' in t['name'] for t in tail),
        tensor_move_duration_sum_ms=sum(t['end_ns']-t['start_ns'] for t in tail if '/TensorMove-op' in t['name'])/1e6,
        first_moment_fused_updates=sum('/Default/GraphKernel_Mul_Add_Assign_fusion-op' in t['name'] for t in tail),
        second_moment_fused_updates=sum('/Default/GraphKernel_Addcmul_Assign_fusion-op' in t['name'] for t in tail),
        tail_aiv_union_ms=union_ns([t for t in tail if t['host_task_type']=='KERNEL_AIVEC'],start,finish)/1e6,
        tail_aic_task_count=sum(t['host_task_type'] in ('KERNEL_AICORE','KERNEL_MIX_AIC','KERNEL_MIX_AIV') for t in native_tail),
        tail_boundary_scope='all overlapping native tasks, including tasks starting before the first state copy',
        unmatched_tail_tasks=sum(t['host_task_type']=='UNKNOWN_NATIVE_JOIN' for t in tail),
        model_ready_aic_work='unknown',
        interpretation='Exact task identity join; diagnostic execution with profiler overhead. Not normal-path critical time or recoverable overlap.')


def render(root,rows,tasks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    row=rows[0];origin=row['step_begin_ns'];end=row['step_end_ns']
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout='constrained')
    colors={'AIC':'#2563eb','AIV':'#ea580c','MIX':'#a855f7','HCCL / other':'#94a3b8'}
    lanes=defaultdict(list)
    for task in tasks:
        if not origin<=task['start_ns']<end:continue
        kind=task['host_task_type']
        lane='AIC' if kind=='KERNEL_AICORE' else 'AIV' if kind=='KERNEL_AIVEC' else 'MIX' if 'KERNEL_MIX' in kind else 'HCCL / other'
        lanes[lane].append(((task['start_ns']-origin)/1e6,(task['end_ns']-task['start_ns'])/1e6))
    labels=['HCCL / other','AIV','MIX','AIC']
    boundaries=[(origin,row['first_scoped_backward_ns'],'Forward (inferred)', '#dbeafe'),
                (row['first_scoped_backward_ns'],row['tail_begin_ns'],'Backward / recompute / norm','#fee2e2'),
                (row['tail_begin_ns'],row['tail_end_ns'],'Optimizer state / update','#dcfce7')]
    for ax in axes:
        for lane,label in enumerate(labels):
            ax.broken_barh(lanes[label],(lane-.3,.6),facecolors=colors[label],rasterized=True)
        ax.set_yticks(range(4),labels);ax.set_ylim(-.7,4.2);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Time from diagnostic step start (ms)')
    for begin,finish,label,color in boundaries:
        low=(begin-origin)/1e6;high=(finish-origin)/1e6
        axes[0].axvspan(low,high,color=color,alpha=.35,zorder=0)
        axes[0].text((low+high)/2,3.65,label,ha='center',fontsize=9)
    axes[0].set_xlim(0,(end-origin)/1e6)
    axes[0].set_title('Real Qwen3-8B O1 training, card 4, step '+str(row['step']))
    axes[1].set_xlim((row['tail_begin_ns']-origin)/1e6,(row['tail_end_ns']-origin)/1e6)
    axes[1].set_title('Optimizer tail: native task types; no AIC/MIX tasks observed')
    fig.suptitle('Separate task diagnostic; ~45% whole-step overhead. No normal-path or readiness claim.')
    for suffix in ('png','svg','pdf'):
        fig.savefig(root/'figures'/('real-model-optimizer-tail.'+suffix),dpi=170)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=RESULTS)
    args=parser.parse_args();root=args.root;source=root/'model-fragment'
    quality=json.loads((source/'diagnostic-evidence.json').read_text());rows=[]
    for rank in quality['ranks']:
        target=source/f"rank{rank['rank']}";tasks,native_tasks=joined_tasks(target)
        with gzip.open(target/'tasks-with-native-types.jsonl.gz','wt') as stream:
            for task in tasks:stream.write(json.dumps(task,separators=(',',':'))+'\n')
        own=[dict(rank=rank['rank'],physical_device=rank['physical_device'],**summarize(tasks,native_tasks,w)) for w in rank['windows']]
        rows.extend(own)
        if rank['rank']==0:render(root,own,tasks)
    (source/'optimizer-tail-evidence.json').write_text(json.dumps(dict(
        diagnostic_only=True,normal_performance_accepted=False,rows=rows),indent=2)+'\n')
    print('identified',len(rows),'real-model diagnostic optimizer tails',flush=True)


if __name__=='__main__':main()
