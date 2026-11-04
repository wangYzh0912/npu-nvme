"""Finish the applicability matrix without changing an active run's sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .scope import ROOT,RAW,save


def main():
    p=argparse.ArgumentParser();p.add_argument('--screen-pid',type=int);p.add_argument('--finalize-only',action='store_true')
    p.add_argument('--after-confirm',action='store_true')
    p.add_argument('--remaining-stages',nargs='+',choices=['version','confirm','blocktable','extend','measure','corun','correctness','correctness-sync','hybrid'])
    a=p.parse_args()
    if not a.finalize_only and a.screen_pid is None:p.error('--screen-pid is required')
    begin=time.monotonic()
    while not a.finalize_only and Path(f'/proc/{a.screen_pid}').exists():
        if time.monotonic()-begin>5400:raise TimeoutError('screen did not finish')
        print('WAITING_SCREEN',len(list((ROOT/'completed').glob('*.json'))),flush=True)
        time.sleep(45)
    if not a.finalize_only:
        stages=a.remaining_stages or (('blocktable','extend','measure','corun','correctness','hybrid') if a.after_confirm else ('version','confirm','blocktable','extend','measure','corun','correctness','hybrid'))
        for stage in stages:
            print('STAGE',stage,flush=True)
            subprocess.run([sys.executable,'-u','-m','tools.aic_kv.scope','--stage',stage],check=True)
        subprocess.run([sys.executable,'-u','-m','tools.aic_kv.scope','--stage','summarize'],check=True)
    # Reduce independent PMU/HBM runs using the previously calibrated reader.
    from .reduce import process
    system=[]
    for run in sorted(RAW.glob('official-system-*')):
        if not run.is_dir() or not (ROOT/'completed'/f'{run.name}.json').exists():continue
        case=run/'b1-l128-o32'
        target=ROOT/'system-analysis'/run.name/case.name
        cached=target/'summary.json'
        system.extend(json.loads(cached.read_text()) if cached.exists() else process(case,target))
    save(ROOT/'system-summary.json',system)
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    for path in RAW.glob('*.log'):shutil.copy2(path,logs/path.name)
    for failure in (ROOT/'failures').glob('*.json'):
        run=RAW/failure.stem
        if not run.exists():continue
        target=ROOT/'failure-evidence'/run.name
        for path in run.rglob('*.json'):
            if any(part in ('trace','system') for part in path.relative_to(run).parts):continue
            dest=target/path.relative_to(run);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    from .correctness_evidence import summarize as summarize_correctness
    summarize_correctness(ROOT)
    for path in RAW.glob('corun-r*'):
        if path.is_dir():shutil.copytree(path,ROOT/'corun'/path.name,ignore=shutil.ignore_patterns('PROF_*','FRAMEWORK','logs'),dirs_exist_ok=True)
    from .corun_evidence import extract as extract_corun
    extract_corun(ROOT)
    source=ROOT/'source'/'final'
    shutil.copytree(Path(__file__).parent,source/'aic_kv',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    env=dict(os.environ,PYTHONPATH='/tmp/user7-aic-aiv-20261031/plotdeps'+os.pathsep+os.environ.get('PYTHONPATH',''))
    subprocess.run([sys.executable,'-m','tools.aic_kv.scope_report'],env=env,check=True)
    acceptance=json.loads((ROOT/'acceptance.json').read_text())
    if not acceptance['investigation_complete']:
        raise RuntimeError('planned evidence is unaccounted for; see acceptance.json')
    entries=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.name in ('SHA256SUMS.json','supervisor.log','finalize.log'):continue
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        entries.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest))
    total=sum(r['bytes'] for r in entries)
    if total>2*1024**3:raise RuntimeError('evidence package exceeds 2GiB budget')
    save(ROOT/'SHA256SUMS.json',dict(total_bytes=total,budget_bytes=2*1024**3,
        excluded_live_logs=['supervisor.log','finalize.log'],files=entries))
    print('SCOPE_ARCHIVED',acceptance['status'],total,'bytes',flush=True)


if __name__=='__main__':main()
