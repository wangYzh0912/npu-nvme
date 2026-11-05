"""Attribute device tasks through exported correlation flows, never CPU time overlap."""
import json,gzip
from bisect import bisect_right
from collections import defaultdict,Counter
from decimal import Decimal
from pathlib import Path

def ns(value):return int(Decimal(str(value))*1000)

def index_intervals(values):
    result={}
    for tid,items in values.items():
        items.sort();result[tid]=([x[0] for x in items],items)
    return result

def containing(index,tid,ts):
    if tid not in index:return None
    starts,items=index[tid];i=bisect_right(starts,ts)-1
    if i>=0 and items[i][0]<=ts<items[i][1]:return items[i][2]
    return None

def annotate(rows,path,evidence_path=None):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as f:events=json.load(f)
    else:events=json.loads(path.read_text())
    if isinstance(events,dict):events=events['traceEvents']
    scopes=defaultdict(list)
    for e in events:
        if e.get('ph')=='X' and e.get('name','').startswith('AIC_ACTIVITY::'):
            start=ns(e['ts']);scopes[e['tid']].append((start,start+ns(e['dur']),e['name'].split('::',1)[1]))
    scope_index=index_intervals(scopes)
    queue={}
    for e in events:
        if e.get('cat')=='async_task_queue' and e.get('ph')=='s':
            queue[e['id']]=containing(scope_index,e['tid'],ns(e['ts']))
    dequeues=defaultdict(list)
    for e in events:
        if e.get('cat')=='dequeue' and e.get('ph')=='X':
            start=ns(e['ts']);dequeues[e['tid']].append((start,start+ns(e['dur']),queue.get(e['args']['correlation_id'])))
    dequeue_index=index_intervals(dequeues)
    launches={};launch_evidence={}
    for e in events:
        cat=e.get('cat')
        if cat in ('HostToDevice','async_npu') and e.get('ph')=='s':
            ts=ns(e['ts']);phase=containing(scope_index,e['tid'],ts) or containing(dequeue_index,e['tid'],ts)
            if phase:
                launches[(cat,e['id'])]=phase
                launch_evidence[(cat,e['id'])]=dict(phase=phase,flow_category=cat,flow_id=e['id'],host_tid=e['tid'],host_launch_ns=ts,
                    scope_source='host_scope' if containing(scope_index,e['tid'],ts) else 'enqueue_dequeue_correlation')
    targets=defaultdict(set);target_evidence=defaultdict(list)
    for e in events:
        if e.get('cat') in ('HostToDevice','async_npu') and e.get('ph')=='f':
            phase=launches.get((e['cat'],e['id']))
            if phase:
                key=(e['tid'],ns(e['ts']));targets[key].add(phase)
                target_evidence[key].append(launch_evidence[(e['cat'],e['id'])])
    mapped={};mapped_evidence={};conflicts=0
    for e in events:
        args=e.get('args',{})
        if e.get('ph')!='X' or 'Task Type' not in args:continue
        phase=targets.get((e['tid'],ns(e['ts'])),set())
        if len(phase)==1:
            key=(str(args.get('Physic Stream Id')),str(args.get('Task Id')),ns(e['ts']))
            mapped[key]=next(iter(phase));mapped_evidence[key]=target_evidence[(e['tid'],ns(e['ts']))]
        elif len(phase)>1:conflicts+=1
    counts=Counter();durations=Counter()
    for row in rows:
        phase=mapped.get((row['stream'],row['task_id'],row['start']))
        row['phase']=phase or 'unclassified'
        row['phase_evidence']='cpu_enqueue_device_flow' if phase else 'unmatched'
        counts[row['phase']]+=1
        if row['engine'] in ('aic','aiv','mixed'):durations[row['phase']]+=row['end']-row['start']
    total=sum(durations.values())
    if evidence_path:
        evidence_path=Path(evidence_path);evidence_path.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(evidence_path,'wt') as f:
            for key,value in mapped_evidence.items():f.write(json.dumps(dict(stream=key[0],task_id=key[1],device_start_ns=key[2],links=value),separators=(',',':'))+'\n')
    return dict(method='CPU scope -> enqueue/dequeue correlation -> HostToDevice/async_npu flow -> exact device task key',
                task_counts=dict(counts),engine_task_ns=dict(durations),conflicting_device_endpoints=conflicts,
                attributed_engine_task_fraction=1-durations['unclassified']/total if total else 0)
