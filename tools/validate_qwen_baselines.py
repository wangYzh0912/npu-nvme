#!/usr/bin/env python3
"""Fail-closed four-method Qwen acceptance; no primary promotion on pilot data."""
import argparse
import json
import math
from pathlib import Path

METHODS=('none','mindspore_native_save','ours','bytecheckpoint_host')

def read(path):return json.loads(Path(path).read_text())

def validate(plan):
    if set(plan['methods'])!=set(METHODS):raise ValueError('four methods required')
    reference={};summary=[]
    for method in METHODS:
        groups=plan['methods'][method]
        if len(groups)!=3:raise ValueError('three independent sources required')
        source_pids=set();restore_pids=set();accepted_restores=set()
        for index,group in enumerate(groups):
            source=Path(group['source']);result=read(source/'result.json')
            if result.get('validation_status')!='pass':raise ValueError('source not accepted')
            for rank in range(4):
                report=read(source/f'rank_{rank}/acceptance.json');pid=report['pid']
                if pid in source_pids:raise ValueError('source processes reused')
                source_pids.add(pid)
                if report['parallel']!={'tensor_parallel':4,'data_parallel':1,'pipeline_parallel':1}:
                    raise ValueError('Qwen topology differs')
                if [r['step'] for r in report['losses']]!=list(range(1,12)):
                    raise ValueError('source schedule differs')
                initial=read(source/f'rank_{rank}/state-initial.json')
                baseline=reference.setdefault((index,rank),(report['data_sha256'],report['losses'],read(source/f'rank_{rank}/state-final.json'),initial))
                if initial!=baseline[3]:raise ValueError('baseline initial state differs')
                if baseline[0]!=report['data_sha256'] or baseline[2]!=read(source/f'rank_{rank}/state-final.json'):
                    raise ValueError('baseline input/final bytes differ')
                if any(a['overflow'] or b['overflow'] or not math.isclose(a['loss'],b['loss'],rel_tol=1e-5,abs_tol=1e-6) for a,b in zip(baseline[1],report['losses'])):
                    raise ValueError('baseline loss differs')
            if method=='none':
                if group.get('restores'):raise ValueError('none cannot have restore')
                continue
            restores=group['restores']
            if len(restores)<1:raise ValueError('fresh correctness restore missing')
            for restore in restores:
                root=Path(restore);accepted_restores.add(str(root.resolve()));report=read(root/'result.json')
                if report.get('validation_status')!='pass':raise ValueError('restore not accepted')
                for rank in range(4):
                    row=read(root/f'rank_{rank}/acceptance.json')
                    if row['pid'] in source_pids or row['pid'] in restore_pids:raise ValueError('restore process not fresh')
                    restore_pids.add(row['pid'])
                    for actual,expected in [('restored-state.json','state-checkpoint.json'),('restored-control.json','control-checkpoint.json'),('state-final.json','state-final.json'),('control-final.json','control-final.json')]:
                        if read(root/f'rank_{rank}'/actual)!=read(source/f'rank_{rank}'/expected):raise ValueError('restored state/control differs')
            summary.append(dict(method=method,source=str(source),restores=len(restores)))
        if method!='none':
            timing=plan['timing'][method]
            if len(timing['measured'])!=3 or not timing['warmup']:raise ValueError('one warmup/three measured restores required')
            timed=[str(Path(path).resolve()) for path in [timing['warmup'],*timing['measured']]]
            if len(set(timed))!=4 or not set(timed)<=accepted_restores:
                raise ValueError('timed restores must be distinct accepted fresh restores')
            for path in timed:
                row=read(Path(path)/'timing.json')
                if row.get('boundary')!='restore_begin_to_global_ready' or row.get('mandatory_integrity') is not True or not math.isfinite(row['seconds']) or row['seconds']<=0:
                    raise ValueError('restore timing boundary or verification missing')
    return dict(validation_status='pass',primary_model='qwen3_8b',fallback_model='gpt2',runs=summary,
                comparison_groups={'ours':'raw-83','mindspore_native_save':'filesystem-84','bytecheckpoint_host':'filesystem-84','none':'training-only'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:result=validate(read(a.manifest))
    except Exception as error:result=dict(validation_status='fail',error=repr(error))
    a.out.write_text(json.dumps(result,indent=2)+'\n');return int(result['validation_status']!='pass')
if __name__=='__main__':raise SystemExit(main())
