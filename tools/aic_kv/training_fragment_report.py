"""Keep ordinary input transport and resident-gradient optimizer evidence apart."""
import argparse
from collections import Counter
from decimal import Decimal
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import tempfile

from .training_jit_campaign import RESULTS


def task_timeline(path):
    """Preserve device ordering; durations are diagnostic, never step timings."""
    with gzip.open(path,'rt') as stream:
        records=list(csv.DictReader(stream))
    if not records:
        raise ValueError('empty task timeline: '+str(path))
    origin=min(Decimal(r['task_start(us)'].strip()) for r in records)
    tasks=[dict(name=r['kernel_name'],kind=r['kernel_type'],
                stream=int(r['stream_id']),task_id=int(r['task_id']),
                start_us=float(Decimal(r['task_start(us)'].strip())-origin),
                duration_us=float(r['task_time(us)'])) for r in records]
    copies=[t for t in tasks if t['kind']=='MEMCPY_ASYNC']
    return dict(tasks=tasks,task_count_by_kind=dict(Counter(t['kind'] for t in tasks)),
                memcpy_count=len(copies),
                memcpy_duration_sum_us=sum(t['duration_us'] for t in copies),
                memcpy_max_duration_us=max((t['duration_us'] for t in copies),default=0),
                transfer_payload_bytes=None,
                transfer_payload_scope='Level0 task timeline does not report payload bytes',
                bulk_transfer_absence_proven=not copies)


def task_types(path):
    with tempfile.TemporaryDirectory() as tmp:
        db=Path(tmp)/'tasks.db'
        with gzip.open(path,'rb') as f:db.write_bytes(f.read())
        conn=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True)
        values=conn.execute('select host_task_type,device_task_type,count(*),sum(duration) from AscendTask group by host_task_type,device_task_type').fetchall()
        conn.close()
    return [dict(host_task_type=a,device_task_type=b,count=n,tasks_per_update=n/8,duration_sum_ns=t) for a,b,n,t in values]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RESULTS);args=p.parse_args()
    rows=[];traces=[]
    for suite in ('fragments','resident-fragments'):
        for run in sorted((args.root/suite/'runs').glob('*')):
            result=run/'result.json'
            if not result.is_file():continue
            d=json.loads(result.read_text());times=d['timings_ms_per_update']
            rows.append(dict(suite=suite,run=run.name,mode=d['mode'],shape=d['shape'],
                oracle_passed=d['oracle']['passed'],median_ms=statistics.median(times),
                cv=statistics.pstdev(times)/statistics.mean(times),
                measurement_accepted=d['oracle']['passed'] and statistics.pstdev(times)/statistics.mean(times)<=.05,
                input_residency=d.get('gradient_input','host_tensor_input'),
                model_ready_work_proven=False))
            native=run/'ascend_task.db.gz'
            if native.is_file():
                traces.append(dict(suite=suite,run=run.name,native_task_types=task_types(native),
                    source_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                    timeline=task_timeline(run/'task_time.csv.gz'),
                    diagnostic_iterations=8,interpretation='diagnostic native task identity; no steady-time or bandwidth-saturation claim'))
    (args.root/'fragment-comparisons.json').write_text(json.dumps(rows,indent=2)+chr(10))
    (args.root/'fragment-native-tasks.json').write_text(json.dumps(traces,indent=2)+chr(10))
    print(json.dumps(dict(ordinary_processes=sum(r['suite']=='fragments' for r in rows),
                         resident_processes=sum(r['suite']=='resident-fragments' for r in rows),
                         native_traces=len(traces))),flush=True)


if __name__=='__main__':main()
