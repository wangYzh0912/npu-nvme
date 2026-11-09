"""Exact flow attribution and pure engine task-envelope overlap for controls."""
from collections import defaultdict
import csv
import json
from .provenance import Ranges,ns,digest


def union(intervals):
    merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
        else:merged.append((lo,hi))
    return merged


def overlap(left,right):
    a=union(left);b=union(right);i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total


def extract(root):
    rows=[]
    for path in sorted((root/'corun').glob('corun-r*/trace/*/ASCEND_PROFILER_OUTPUT')):
        trace=path/'trace_view.json'
        events=json.loads(trace.read_text())
        if isinstance(events,dict):events=events['traceEvents']
        scopes=Ranges(events);starts=defaultdict(list);ends=defaultdict(list)
        for e in events:
            if e.get('cat') not in ('async_npu','HostToDevice'):continue
            if e.get('ph')=='s':starts[(e['cat'],e['id'])].append(e)
            if e.get('ph')=='f':ends[(e['pid'],e['tid'],ns(e['ts']))].append(e)
        with (path/'kernel_details.csv').open() as f:tasks=list(csv.DictReader(f))
        lookup={(ns(t['Start Time(us)']),str(t['Task ID']),str(t['Stream ID'])):t for t in tasks}
        attributed=[];unknown=0
        for e in events:
            args=e.get('args',{})
            if e.get('ph')!='X' or 'Physic Stream Id' not in args or 'Task Id' not in args:continue
            key=(ns(e['ts']),str(args['Task Id']),str(args['Physic Stream Id']))
            if key not in lookup:continue
            modes=set()
            for endpoint in ends.get((e['pid'],e['tid'],ns(e['ts'])),[]):
                launch=starts.get((endpoint['cat'],endpoint['id']),[])
                if len(launch)!=1:continue
                modes.update(r['name'].split('::',1)[1] for r in scopes.containing(launch[0]) if r['name'].startswith('KV_CORUN::'))
            if len(modes)!=1:unknown+=1;continue
            task=lookup[key]
            attributed.append(dict(mode=next(iter(modes)),name=task['Name'],engine=task['Accelerator Core'],
                start_ns=key[0],end_ns=key[0]+ns(task['Duration(us)']),stream=key[2]))
        modes={}
        for name in ('matrix_only','copy_only','serial_pair','parallel_pair'):
            selected=[t for t in attributed if t['mode']==name]
            aic=[(t['start_ns'],t['end_ns']) for t in selected if t['engine']=='AI_CORE']
            aiv=[(t['start_ns'],t['end_ns']) for t in selected if t['engine']=='AI_VECTOR_CORE']
            modes[name]=dict(tasks=len(selected),aic_tasks=len(aic),aiv_tasks=len(aiv),
                aic_union_ns=sum(hi-lo for lo,hi in union(aic)),aiv_union_ns=sum(hi-lo for lo,hi in union(aiv)),
                pure_engine_overlap_ns=overlap(aic,aiv),engines=sorted({t['engine'] for t in selected}))
        rows.append(dict(run=path.parents[2].name,trace_sha256=digest(trace),unattributed_device_tasks=unknown,
            modes=modes,tasks=attributed,metric='task envelopes, not core-level PMU activity'))
    (root/'corun/task-evidence.json').write_text(json.dumps(rows,indent=2)+'\n')
    return rows
