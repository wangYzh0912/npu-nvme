"""Task activity with explicit windows, orthogonal phase/category, and quality."""
import csv
import gzip
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from tools.aic_aiv_analyse import segments, curve, union_intervals

STATES = ('aic_only', 'aiv_only', 'both', 'neither', 'unknown')
ENGINES = {'AI_CORE': 'aic', 'AI_VECTOR_CORE': 'aiv', 'MIX_AIC': 'mixed', 'MIX_AIV': 'mixed'}
OTHER = {'AI_CPU', 'AICPU', 'COMMUNICATION', 'HCCL', 'DMA', 'MEMCPY', 'EVENT', 'HOST_CPU'}

def classify(name, kind, op_type=''):
    s = (name + ' ' + op_type).lower()
    phase = ('optimizer' if 'optimizer-' in s or 'adam' in s else
             'backward' if 'gradient' in s or '/grad_' in s else
             'recompute' if 'recompute' in s else
             'input' if 'getnext' in s else
             'forward' if 'network-' in s else 'unclassified')
    category = ('communication' if kind in ('COMMUNICATION', 'HCCL') or 'hcom' in s else
                'data_movement' if any(x in s for x in ('contiguous', 'tensormove', 'transpose', 'viewcopy', 'memcpy')) else
                'compute' if kind in ENGINES else 'other')
    return phase, category

def read(path):
    path = Path(path)
    if path.name.endswith('.jsonl.gz'):
        with gzip.open(path, 'rt') as f:
            rows = [json.loads(s) for s in f]
        for r in rows:
            phase,category = classify(r['name'], r['kind'])
            r.setdefault('phase',phase);r.setdefault('category',category)
        return rows
    rows = []
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        rich = 'Accelerator Core' in reader.fieldnames
        fields = ('Start Time(us)', 'Duration(us)', 'Accelerator Core', 'Name', 'Type') if rich else ('Task Start Time(us)', 'Task Duration(us)', 'Task Type', 'Op Name', 'OP Type')
        for line, r in enumerate(reader, 2):
            try:
                start = int(Decimal(r[fields[0]].strip()) * 1000)
                duration = int(Decimal(r[fields[1]].strip()) * 1000)
                if duration < 0: raise ValueError('negative duration')
                kind, name, typ = (r[k] for k in fields[2:])
                phase, category = classify(name, kind, typ)
                rows.append(dict(start=start, end=start+duration, engine=ENGINES.get(kind, 'unknown' if kind not in OTHER else 'other'),
                                 kind=kind, name=name, phase=phase, category=category, stream=r.get('Stream ID'),
                                 task_id=r.get('Task ID'), step_id=r.get('Step ID')))
            except (ValueError, KeyError, ArithmeticError) as e:
                raise ValueError(f'{path}:{line}: {e}') from e
    if not rows: raise ValueError(f'empty task table: {path}')
    return rows

def partition(rows, begin, end):
    # Unrecognized task types mask activity rather than silently mean idle.
    return segments([dict(r, engine='mixed') if r['engine']=='unknown' else r for r in rows], begin, end)

def metrics(rows, begin, end, quality='diagnostic'):
    if end <= begin: raise ValueError('empty or reversed observation window')
    parts = partition(rows, begin, end)
    total = {s:sum(p['end']-p['start'] for p in parts if p['state']==s) for s in STATES}
    width=end-begin
    selected=[r for r in rows if r['start']<end and r['end']>begin]
    elapsed=lambda r:max(0,min(end,r['end'])-max(begin,r['start']))
    phase={}
    for name in sorted({r['phase'] for r in selected}):
        own=[r for r in selected if r['phase']==name]
        windows=union_intervals(own,begin,end)
        states=Counter()
        # Restrict the global timeline to the union of this phase's tasks.
        i=0
        for l,h in windows:
            while i<len(parts) and parts[i]['end']<=l:i+=1
            j=i
            while j<len(parts) and parts[j]['start']<h:
                p=parts[j];states[p['state']]+=max(0,min(h,p['end'])-max(l,p['start']));j+=1
        phase[name]=dict(task_union_ns=sum(h-l for l,h in windows),state_ns=dict(states),
                         task_ns_by_engine=dict(Counter({k:sum(elapsed(r) for r in own if r['engine']==k) for k in {r['engine'] for r in own}})))
    windows=sorted(p['end']-p['start'] for p in parts if p['state']=='aiv_only')
    quantile=lambda q:windows[round(q*(len(windows)-1))] if windows else None
    top=Counter()
    for r in selected:
        if r['engine'] in ('aic','aiv','mixed'): top[(r['engine'],r['name'].split('/')[-1])]+=elapsed(r)
    result=dict(schema_version=3,metric_kind='task_envelope',quality=quality,begin_ns=begin,end_ns=end,duration_ns=width,
                state_ns=total,state_fractions={k:v/width for k,v in total.items()},known_coverage=1-total['unknown']/width,
                task_count=len(selected),engine_counts=dict(Counter(r['engine'] for r in selected)),
                conditional_known=dict(aic_absent_given_aiv=total['aiv_only']/(total['aiv_only']+total['both']) if total['aiv_only']+total['both'] else None,
                                       aiv_absent_given_aic=total['aic_only']/(total['aic_only']+total['both']) if total['aic_only']+total['both'] else None),
                aiv_only_windows=dict(count=len(windows),p50_ns=quantile(.5),p95_ns=quantile(.95),max_ns=max(windows,default=0)),
                category_union_ns={c:sum(h-l for l,h in union_intervals([r for r in selected if r['category']==c],begin,end)) for c in sorted({r['category'] for r in selected})},
                phases=phase,top_tasks=[dict(engine=k[0],name=k[1],task_ns=v) for k,v in top.most_common(20)])
    if sum(total.values())!=width: raise AssertionError('activity partition does not conserve time')
    return result, parts

def save(rows, windows, out, metadata, bin_us=1000):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    summaries=[];bins=[];stages=[]
    for w in windows:
        d,parts=metrics(rows,w['begin_ns'],w['end_ns'],w.get('quality','diagnostic'))
        d.update(metadata,window_id=w['id']);summaries.append(d)
        for b in curve(parts,w['begin_ns'],w['end_ns'],bin_us*1000):
            b.update(window_id=w['id'],physical_device_id=metadata['physical_device_id'],metric_kind='task_envelope',valid=w.get('quality')!='invalid')
            bins.append(b)
        for phase in sorted({r['phase'] for r in rows}):
            for l,h in union_intervals([r for r in rows if r['phase']==phase],w['begin_ns'],w['end_ns']):
                stages.append(dict(window_id=w['id'],stage=phase,device_start_ns=l,device_end_ns=h,evidence_source=metadata.get('phase_evidence','kernel_scope')))
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    for filename,values in [('timeseries.csv',bins),('stages.csv',stages)]:
        with (out/filename).open('w',newline='') as f:
            if values:
                writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    with gzip.open(out/'events.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r,separators=(',',':'))+'\n')
    return summaries
