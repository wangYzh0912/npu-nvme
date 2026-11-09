"""Rebuild delivered PMU/HBM tables using only retained native inputs."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile

from .training_jit_campaign import RESULTS


def close(actual,expected):
    if expected is None:return actual is None
    return math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-9)


def replay_rank(run,rank):
    from .reduce import export_pmu
    from .hbm import export as export_hbm
    target=run/f'rank{rank}'
    windows=json.loads((run/'sample-evidence.json').read_text())['ranks'][rank]['windows']
    with tempfile.TemporaryDirectory(prefix='training-counter-replay-') as temporary:
        device=Path(temporary)/'device';(device/'sqlite').mkdir(parents=True)
        for pattern in ('info.json.*','start_info.*','sample.json'):
            for p in target.glob(pattern):shutil.copy2(p,device/p.name)
        counts={}
        for name in ('aicore.db','ai_vector_core.db'):
            connection=sqlite3.connect(device/'sqlite'/name)
            connection.execute('create table AICoreOriginalData(timestamp NUMERIC,coreid INTEGER,task_cyc INTEGER,replayid INTEGER)')
            with gzip.open(target/(name+'.replay0.csv.gz'),'rt') as stream:
                # SQLite NUMERIC restores integer or half-nanosecond REAL
                # timestamps exactly as in the original native table.
                values=[(r['timestamp'],int(r['coreid']),int(r['task_cyc']),int(r['replayid'])) for r in csv.DictReader(stream)]
            connection.executemany('insert into AICoreOriginalData values(?,?,?,?)',values)
            connection.commit();connection.close();counts[name]=len(values)
        pmu,_=export_pmu(device,Path(temporary)/'pmu',windows)
        saved=list(csv.DictReader((target/'pmu/core_timeseries.csv').open()))
        if len(pmu)!=len(saved):raise ValueError('PMU replay row count differs')
        for actual,expected in zip(pmu,saved):
            for key in ('window_start_ns','window_end_ns'):
                if actual[key]!=int(expected[key]):raise ValueError('PMU replay time differs')
            if actual['valid']!=(expected['valid']=='True'):raise ValueError('PMU replay quality differs')
            for key in ('aic_activity_pct','aiv_activity_pct','aic_coverage','aiv_coverage'):
                value=float(expected[key]) if expected[key] else None
                if not close(actual[key],value):raise ValueError('PMU replay value differs: '+key)
        with gzip.open(target/'hbm.db.gz','rb') as source,(device/'sqlite/hbm.db').open('wb') as destination:
            shutil.copyfileobj(source,destination)
        clock=json.loads((target/'hbm-clock.json').read_text())
        hbm=export_hbm(device,Path(temporary)/'hbm',windows,clock_offset_ns=clock['offset_ns'])
        expected=json.loads((target/'hbm/hbm-summary.json').read_text())
        if len(hbm['windows'])!=len(expected['windows']):raise ValueError('HBM replay row count differs')
        for actual,saved in zip(hbm['windows'],expected['windows']):
            for key in ('coverage','read_bytes','write_bytes'):
                if not close(actual[key],saved[key]):raise ValueError('HBM replay value differs: '+key)
    return dict(run=run.name,rank=rank,native_counter_rows=counts,pmu_bins=len(pmu),
                hbm_windows=len(hbm['windows']),passed=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=RESULTS)
    args=parser.parse_args();rows=[]
    for run in sorted((args.root/'training/runs').glob('*')):
        if not (run/'sample-evidence.json').is_file():continue
        for rank in range(4):
            row=replay_rank(run,rank);rows.append(row)
        print('replayed',run.name,flush=True)
    (args.root/'counter-replay-verification.json').write_text(json.dumps(dict(
        status='pass',rank_runs=len(rows),rows=rows,
        scope='Archive-only native input reconstruction; same metric implementation, not an independent hardware calibration'),indent=2)+'\n')


if __name__=='__main__':main()
