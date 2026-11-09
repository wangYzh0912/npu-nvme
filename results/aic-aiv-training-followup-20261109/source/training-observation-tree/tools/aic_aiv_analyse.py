#!/usr/bin/env python3
"""Exact interval unions for AIC/AIV task envelopes, never core occupancy.

MIX kernel envelopes mask both engines as unknown. Timestamps are parsed as
integer nanoseconds before subtracting their epoch to retain sub-us precision.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path

KINDS = {'AI_CORE': 'aic', 'AI_VECTOR_CORE': 'aiv', 'MIX_AIC': 'mixed', 'MIX_AIV': 'mixed'}
STATES = ('aic_only', 'aiv_only', 'both', 'neither', 'unknown')


def stage(name, kind):
    s = (name + ' ' + kind).lower()
    if 'optimizer-' in s or 'adam' in s: return 'optimizer'
    if 'gradient' in s or '/grad_' in s: return 'backward'
    if 'recompute' in s: return 'recompute'
    if 'hcom' in s or any(x in s for x in ('allreduce', 'allgather', 'reducescatter')): return 'communication'
    if 'getnext' in s: return 'input'
    if 'network-' in s or 'forward' in s: return 'forward'
    return 'unclassified'


def ns(value):
    return int(Decimal(value.strip()) * 1000)


def read_tasks(path):
    output = []
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        rich = 'Accelerator Core' in reader.fieldnames
        fields = ('Start Time(us)', 'Duration(us)', 'Accelerator Core', 'Name', 'Type') if rich else ('Task Start Time(us)', 'Task Duration(us)', 'Task Type', 'Op Name', 'OP Type')
        for n, row in enumerate(reader, 2):
            startkey, durationkey, enginekey, namekey, typekey = fields
            # Malformed device input must fail, rather than silently drop work.
            try:
                start, duration = ns(row[startkey]), ns(row[durationkey])
            except Exception as e:
                raise ValueError(f'{path}:{n}: invalid timing') from e
            if duration < 0: raise ValueError(f'{path}:{n}: negative duration')
            name, op_type = row[namekey], row[typekey]
            output.append(dict(start=start, end=start+duration, engine=KINDS.get(row[enginekey], 'other'),
                               kind=row[enginekey], name=name, stage=stage(name, op_type), stream=row.get('Stream ID')))
    if not any(r['engine'] != 'other' for r in output): raise ValueError('no engine tasks')
    return output


def segments(rows, begin, end):
    events = defaultdict(lambda: [0, 0, 0])
    events[begin]; events[end]
    for row in rows:
        if row['engine'] not in ('aic', 'aiv', 'mixed'): continue
        l, r = max(begin, row['start']), min(end, row['end'])
        if r <= l: continue
        axis = {'aic': 0, 'aiv': 1, 'mixed': 2}[row['engine']]
        events[l][axis] += 1; events[r][axis] -= 1
    active, parts = [0, 0, 0], []
    times = sorted(events)
    for l, r in zip(times, times[1:]):
        active = [a+b for a,b in zip(active, events[l])]
        assert min(active) >= 0
        state = ('unknown' if active[2] else 'both' if active[0] and active[1] else
                 'aic_only' if active[0] else 'aiv_only' if active[1] else 'neither')
        if parts and parts[-1]['state'] == state: parts[-1]['end'] = r
        else: parts.append(dict(start=l, end=r, state=state))
    assert sum(p['end']-p['start'] for p in parts) == end-begin
    return parts


def union_intervals(rows, begin, end):
    intervals = sorted((max(begin, x['start']), min(end, x['end'])) for x in rows if x['end'] > begin and x['start'] < end)
    out = []
    for l, r in intervals:
        if out and l <= out[-1][1]: out[-1][1] = max(out[-1][1], r)
        elif r > l: out.append([l, r])
    return out


def curve(parts, begin, end, bin_ns):
    bins = []
    for l in range(begin, end, bin_ns):
        r = min(l+bin_ns, end)
        counts = dict.fromkeys(STATES, 0)
        bins.append([l,r,counts])
    for p in parts:
        first = max(0, (p['start']-begin)//bin_ns)
        last = min(len(bins)-1, (p['end']-1-begin)//bin_ns)
        for i in range(first, last+1):
            l,r,c = bins[i]; c[p['state']] += max(0, min(r,p['end'])-max(l,p['start']))
    return [dict(start_ns=l-begin, end_ns=r-begin, **{k:v/(r-l) for k,v in c.items()}) for l,r,c in bins]


def analyze(path, output, label, bin_us=1000):
    rows = read_tasks(path)
    origin = min(x['start'] for x in rows)
    for x in rows: x['start']-=origin; x['end']-=origin
    anchors = sorted(set(x['start'] for x in rows if x['stage']=='input'))
    # Adjacent GetNext starts bound complete steady iterations; discard the
    # final partial step, instead of inventing a completion boundary.
    if len(anchors)>=2:
        begin,end = anchors[0],anchors[-1]
        window = 'adjacent GetNext starts; last partial iteration discarded'
    else:
        engine_rows = [r for r in rows if r['engine']!='other']
        begin,end = min(r['start'] for r in engine_rows),max(r['end'] for r in engine_rows)
        window = 'first to last device engine task (no iteration boundary markers)'
    parts = segments(rows, begin, end)
    totals = {s: sum(p['end']-p['start'] for p in parts if p['state']==s) for s in STATES}
    duration=end-begin
    aiv=totals['aiv_only']+totals['both']; aic=totals['aic_only']+totals['both']
    scope = {s: union_intervals([r for r in rows if r['stage']==s],begin,end) for s in sorted(set(r['stage'] for r in rows))}
    result = dict(schema_version=2,label=label,source=str(path),time_origin_us=str(Decimal(origin)/1000),
                  window=window,begin_ns=begin,end_ns=end,duration_ns=duration,
                  complete_iterations=len(anchors)-1 if len(anchors)>=2 else None,
                  task_count=sum(r['end']>begin and r['start']<end for r in rows),
                  engine_task_counts=dict(Counter(r['engine'] for r in rows if r['end']>begin and r['start']<end)),
                  state_ns=totals,state_fractions={s:v/duration for s,v in totals.items()},
                  known_coverage=1-totals['unknown']/duration,
                  p_aic_task_absent_given_aiv_task_present_known=totals['aiv_only']/aiv if aiv else None,
                  p_aiv_task_absent_given_aic_task_present_known=totals['aic_only']/aic if aic else None,
                  stage_union_ns={s:sum(r-l for l,r in xs) for s,xs in scope.items()},
                  limitations=['Task-envelope activity is not core occupancy or computational utilization.',
                               'Mixed task envelopes are unknown; conditional probabilities only cover known intervals.',
                               'Stage labels classify framework scope names, not a mutually exclusive host phase segmentation.',
                               'A pilot with short warmup cannot establish steady-state performance or acceleration potential.'])
    output.mkdir(parents=True,exist_ok=True)
    (output/'activity_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    binned=curve(parts,begin,end,int(bin_us*1000))
    with (output/'activity_bins.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(binned[0]));w.writeheader();w.writerows(binned)
    with gzip.open(output/'task_events.jsonl.gz','wt') as f:
        for row in rows: f.write(json.dumps(row,separators=(',',':'))+'\n')
    plot(result,binned,scope,output)
    return result


def plot(result,bins,scope,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    x=np.array([r['start_ns']/1e6 for r in bins]); xe=np.r_[x, bins[-1]['end_ns']/1e6]
    aic=np.array([r['aic_only']+r['both'] for r in bins])*100
    aiv=np.array([r['aiv_only']+r['both'] for r in bins])*100
    unk=np.array([r['unknown'] for r in bins])*100
    fig,axes=plt.subplots(3,1,figsize=(14,8),sharex=True,gridspec_kw={'height_ratios':[3,1.4,2]})
    ax=axes[0]
    ax.stairs(aic,xe,label='AIC known task activity',color='#275dad',linewidth=.9)
    ax.stairs(aiv,xe,label='AIV known task activity',color='#da780f',linewidth=.9)
    ax.set(ylabel='Task time in bin (%)',ylim=(-2,102),title=result['label']+' — task envelopes, not core occupancy')
    ax.legend(loc='upper right',ncol=2);ax.grid(alpha=.2)
    axes[1].stairs(unk,xe,color='#7e43ad',fill=True,alpha=.6,label='Mixed envelope: engine activity unknown')
    axes[1].set(ylabel='Unknown (%)',ylim=(0,100));axes[1].legend(loc='upper right')
    palette={'forward':'#2878b5','backward':'#cb5555','recompute':'#ad6aaf','optimizer':'#5c9e53','communication':'#8e8169','input':'#6ca7ad','unclassified':'#bcbcbc'}
    for i,(s,intervals) in enumerate(scope.items()):
        axes[2].broken_barh([((l-result['begin_ns'])/1e6,(r-l)/1e6) for l,r in intervals],(i-.35,.7),facecolors=palette[s],rasterized=True)
    axes[2].set(yticks=range(len(scope)),yticklabels=list(scope),xlabel='Device time from first complete iteration start (ms)')
    axes[2].set_xlim(0,result['duration_ns']/1e6)
    fig.text(.01,.01,f"Known coverage: {result['known_coverage']:.1%}. Bin width: {(bins[0]['end_ns']-bins[0]['start_ns'])/1e6:g} ms. Phase stripes show scoped task intervals; gaps are not a proof of idle hardware.",fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1))
    for ext in ('png','svg','pdf'):fig.savefig(out/('activity_timeline.'+ext),dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--label',required=True)
    p.add_argument('--bin-us',type=int,default=1000)
    a=p.parse_args()
    if a.bin_us<=0:p.error('--bin-us must be positive')
    print(json.dumps(analyze(a.tasks,a.output,a.label,a.bin_us),indent=2))
if __name__=='__main__':main()
