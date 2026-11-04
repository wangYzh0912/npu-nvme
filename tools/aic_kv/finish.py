"""Finish the authorized experiment stages after the running capacity matrix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .campaign import ROOT,RAW,CONTAINER


def main():
    p=argparse.ArgumentParser();p.add_argument('--capacity-pid',type=int)
    p.add_argument('--finalize-only',action='store_true')
    p.add_argument('--micro-run',default='micro')
    a=p.parse_args()
    if not a.finalize_only and a.capacity_pid is None:
        p.error('--capacity-pid is required unless --finalize-only is used')
    deadline=time.monotonic()+7200
    def run(command,**kwargs):
        print('FOLLOWUP', ' '.join(map(str,command)),flush=True)
        subprocess.run(command,check=True,**kwargs)
    next_status=0
    while not a.finalize_only and Path(f'/proc/{a.capacity_pid}').exists():
        if time.monotonic()>deadline:raise TimeoutError('capacity matrix did not finish in two hours')
        if time.monotonic()>=next_status:
            print('WAITING_FOR_CAPACITY',len(list(RAW.glob('capacity-*-export.log'))),'exports started',flush=True)
            next_status=time.monotonic()+60
        time.sleep(20)
    completed=[p for p in RAW.glob('capacity-r*-n*') if p.is_dir() and (p/'completed.json').exists()]
    if len(completed)!=9:raise RuntimeError(f'capacity campaign incomplete: {len(completed)}/9')
    if not a.finalize_only:
        run([sys.executable,'-u','-m','tools.aic_kv.campaign','--stage','metadata','--tag','native'])
        run([sys.executable,'-u','-m','tools.aic_kv.campaign','--stage','work','--metadata-run','metadata-a2'])
        run([sys.executable,'-u','-m','tools.aic_kv.campaign','--stage','micro','--metadata-run','metadata-native'])
    run([sys.executable,'-u','-m','tools.aic_kv.reduce','--raw',str(RAW),'--output',str(ROOT/'analysis')])
    # Analysis provenance and the operator replay are small enough to retain.
    shutil.copytree(RAW/a.micro_run,ROOT/'micro',ignore=shutil.ignore_patterns('PROF_*','FRAMEWORK','logs'))
    (ROOT/'micro'/'run-identity.json').write_text(json.dumps(dict(raw_run=a.micro_run),indent=2)+'\n')
    shutil.copytree(RAW/'metadata-native',ROOT/'metadata-native')
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    for path in RAW.glob('*.log'):
        shutil.copy2(path,logs/path.name)
    sources=ROOT/'source'/'final-offline'
    shutil.copytree(Path(__file__).parent,sources/'aic_kv',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(Path(__file__).parents[1]/'aic_aiv_followup.py',sources/'aic_aiv_followup.py')
    env=dict(os.environ,PYTHONPATH='/tmp/user7-aic-aiv-20261031/plotdeps'+os.pathsep+os.environ.get('PYTHONPATH',''))
    run([sys.executable,'-m','tools.aic_kv.report','--root',str(ROOT)],env=env)
    acceptance=json.loads((ROOT/'acceptance.json').read_text())
    if not acceptance.get('required_evidence_passed'):
        raise RuntimeError('final evidence acceptance failed; see acceptance.json')
    run(['docker','stop','--time','10',CONTAINER])
    entries=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.name in ('SHA256SUMS.json','finish.log'):continue
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
        entries.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=h.hexdigest()))
    total=sum(x['bytes'] for x in entries)
    if total>1024**3:raise RuntimeError(f'durable evidence budget exceeded: {total}')
    (ROOT/'SHA256SUMS.json').write_text(json.dumps(dict(total_bytes=total,files=entries,
        excluded_live_supervisor_log='finish.log'),indent=2)+'\n')
    print('FOLLOWUP_COMPLETE',total,'bytes',flush=True)


if __name__=='__main__':main()
