"""Attribute Slice tasks through exact device endpoints and native flow IDs."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path


def ns(value):
    return int(Decimal(str(value))*1000)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
    return h.hexdigest()


class Ranges:
    def __init__(self, events):
        ranges=defaultdict(list)
        for e in events:
            if e.get('ph')=='X' and (e.get('cat')=='cpu_op' or e.get('name','').startswith(('AscendCL@','KV_MICRO::','KV_CORUN::'))):
                lo=ns(e['ts'])
                ranges[(e['pid'],e['tid'])].append((lo,lo+ns(e.get('dur',0)),e['name']))
        self.index={}
        for key,items in ranges.items():
            items.sort();mx=-1;prefix=[]
            for _,hi,_ in items:mx=max(mx,hi);prefix.append(mx)
            self.index[key]=([x[0] for x in items],items,prefix)

    def containing(self, event):
        key=(event['pid'],event['tid']);ts=ns(event['ts'])
        if key not in self.index:return []
        starts,items,prefix=self.index[key];i=bisect_right(starts,ts)-1;result=[]
        while i>=0 and prefix[i]>ts:
            lo,hi,name=items[i]
            if lo<=ts<hi:result.append(dict(name=name,start_ns=lo,end_ns=hi))
            i-=1
        return result


def extract(run, output, rank=None):
    run,output=Path(run),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    windows=[w for w in json.loads((run/'windows.json').read_text()) if w['kind'] in ('measured','task')]
    results=[]
    traces=sorted((run/'trace').glob('rank*_*/ASCEND_PROFILER_OUTPUT'))
    if len(traces)!=4:raise ValueError(f'expected four exported rank traces: {run}')
    if rank is not None:
        traces=[trace for trace in traces if trace.parent.name.startswith(f'rank{rank}_')]
    for trace in traces:
        tracefile=trace/'trace_view.json.gz'
        if not tracefile.exists():tracefile=trace/'trace_view.json'
        opener=gzip.open if tracefile.suffix=='.gz' else open
        with opener(tracefile,'rt') as f:events=json.load(f)
        if isinstance(events,dict):events=events['traceEvents']
        intervals=Ranges(events)
        starts=defaultdict(list);ends=defaultdict(list)
        for e in events:
            if e.get('cat') not in ('async_npu','HostToDevice'):continue
            if e.get('ph')=='s':starts[(e['cat'],e['id'])].append(e)
            elif e.get('ph')=='f':ends[(e['pid'],e['tid'],ns(e['ts']))].append(e)
        kernel=trace/'kernel_details.csv'
        with kernel.open() as f:rows=list(csv.DictReader(f))
        device=int(rows[0]['Device_id']);rank=device-4
        slice_keys={(str(r['Stream ID']),str(r['Task ID']),ns(r['Start Time(us)'])):r for r in rows if 'aclnnContiguous_SliceAiCore_Slice' in r['Name']}
        evidence=[]
        for e in events:
            if e.get('ph')!='X' or 'aclnnContiguous_SliceAiCore_Slice' not in e.get('name',''):continue
            ts=ns(e['ts']);w=next((w for w in windows if w['begin_ns']<=ts<w['end_ns']),None)
            if w is None:continue
            args=e['args'];task= slice_keys[(str(args['Physic Stream Id']),str(args['Task Id']),ts)]
            links=[]
            for endpoint in ends.get((e['pid'],e['tid'],ts),[]):
                matches=starts.get((endpoint['cat'],endpoint['id']),[])
                if len(matches)==1:
                    launch=matches[0]
                    links.append(dict(category=endpoint['cat'],id=endpoint['id'],host_pid=launch['pid'],
                        host_tid=launch['tid'],host_launch_ns=ns(launch['ts']),ranges=intervals.containing(launch)))
            names={r['name'] for link in links for r in link['ranges']}
            evidence.append(dict(device=device,rank=rank,stream=args['Physic Stream Id'],task_id=args['Task Id'],
                start_ns=ts,end_ns=ts+ns(e['dur']),window_index=w.get('index',w.get('id')),
                input_shapes=task['Input Shapes'],output_shapes=task['Output Shapes'],
                input_dtype=task['Input Data Types'],output_dtype=task['Output Data Types'],
                block_num=task['Block Num'],links=links,
                fia='npu::npu_fused_infer_attention_score' in names,
                inner_fia='AscendCL@aclnnInnerFusedInferAttentionScore' in names,
                contiguous='AscendCL@aclnnContiguous' in names,decode='AIC_ACTIVITY::decode' in names))
        with gzip.open(output/f'rank{rank}-slice-links.jsonl.gz','wt') as f:
            for e in evidence:f.write(json.dumps(e,separators=(',',':'))+'\n')
        summary=dict(device=device,rank=rank,requests=len(windows),slice_count=len(evidence),
            matched={name:sum(e[name] for e in evidence) for name in ('fia','inner_fia','contiguous','decode')},
            slice_time_ns=sum(e['end_ns']-e['start_ns'] for e in evidence),
            output_shapes=dict(Counter(e['output_shapes'] for e in evidence)),
            files={str(p):digest(p) for p in (tracefile,kernel,run/'windows.json')})
        results.append(summary)
        print('PROVENANCE',device,summary['slice_count'],summary['matched'],flush=True)
        del events,intervals,starts,ends,rows,evidence
    (output/'provenance.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--rank',type=int,choices=range(4))
    a=p.parse_args();extract(a.run,a.output,a.rank)


if __name__=='__main__':main()
