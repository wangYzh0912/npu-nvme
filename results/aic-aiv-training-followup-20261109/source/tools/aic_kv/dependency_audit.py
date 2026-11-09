"""Audit retained training evidence without inventing runtime dependency edges."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3


def audit(run, normalized):
    rows=[]
    for rank in range(4):
        root=run/f'rank_{rank}/activity-profiler'
        api=list(root.glob('**/host/sqlite/api_event.db'))
        if len(api)!=1:raise ValueError(f'expected one host API DB for rank {rank}: {api}')
        c=sqlite3.connect('file:'+str(api[0])+'?mode=ro',uri=True)
        events=[dict(api=name,count=n,distinct_item_ids=unique) for name,n,unique in c.execute(
            "select id,count(*),count(distinct item_id) from ApiData where id in ('aclrtRecordEvent','aclrtStreamWaitEvent') group by id")]
        ids=[row[0] for row in c.execute(
            "select distinct item_id from ApiData where id in ('aclrtRecordEvent','aclrtStreamWaitEvent')")]
        schema=[dict(name=x[1],type=x[2]) for x in c.execute('pragma table_info(ApiData)')]
        c.close()
        times=Counter();ops=Counter();counts=Counter()
        source=normalized/f'rank{rank}/events.jsonl.gz'
        with gzip.open(source,'rt') as f:
            for line in f:
                row=json.loads(line)
                if row['engine']!='aiv':continue
                duration=row['end']-row['start'];name=row['name'].split('/')[-1]
                times[row['phase']]+=duration;ops[name]+=duration;counts[name]+=1
        rows.append(dict(rank=rank,device=rank+4,events=events,item_ids=ids,api_schema=schema,
            api_db=str(api[0]),api_db_sha256=hashlib.sha256(api[0].read_bytes()).hexdigest(),
            normalized_events_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            aiv_task_duration_ns_by_phase=dict(times),
            top_aiv_ops=[dict(name=name,task_duration_ns=ns,count=counts[name]) for name,ns in ops.most_common(12)],
            readiness='unknown',reason='event handles/producer-consumer edges absent; API issue time is not dependency completion'))
    quality=json.loads((normalized/'acceptance.json').read_text())
    return dict(run=str(run),normalized=str(normalized),ranks=rows,
        trace_quality=quality,real_model_ready_candidate_count=None,
        interpretation='high-overhead traces identify tasks; cannot prove normal timing or runnable queue state')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path('/models/npu_nvme_exp/user7-stack/aic-aiv-v2/analysis')
    raw=Path('/dev/shm/user7-aic-aiv-v2/runs')
    result={str(n):audit(raw/f'train-{n}-light-r0',root/f'train-{n}-light-r0') for n in (128,512)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
