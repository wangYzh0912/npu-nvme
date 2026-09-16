#!/usr/bin/env python3
"""Serial E1/F1 pilot phases, with explicit reports and safe failure retention."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.run_campaign import members,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['e1','f1'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--seed',type=int,default=41);p.add_argument('--model',default='gpt2',choices=['gpt2','gpt2_xl'])
    p.add_argument('--shm-base',type=int,required=True);p.add_argument('--timeout',type=int,default=3600)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    phases=['format','baseline','source','restore'] if a.kind=='e1' else ['source','restore']
    rows=[];result=dict(status='running',runs=rows)
    for index,phase in enumerate(phases):
        script='tests/hardware/e1_live_training.py' if a.kind=='e1' else 'tests/hardware/f1_training.py'
        cmd=[sys.executable,'scripts/run_user_environment.py','--manifest',str(a.manifest),'--profile','old','--','python',script,
             '--phase',phase,'--out',str(a.out),'--seed',str(a.seed),'--shm-id',str(a.shm_base+index)]
        if a.kind=='e1':cmd+=['--model',a.model]
        with (a.out/(phase+'.log')).open('w') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row=dict(phase=phase,pid=child.pid,status='running');rows.append(row);write(a.out/'result.json',result)
            try:rc=child.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                row.update(status='retained',alive=members(child.pid));result['status']='retained';write(a.out/'result.json',result);threading.Event().wait()
            if members(child.pid):
                row.update(status='retained',alive=members(child.pid));result['status']='retained';write(a.out/'result.json',result);threading.Event().wait()
        try:
            report=json.loads((a.out/(phase+'.json')).read_text())
            row.update(status='pass' if rc==0 and report['status']=='pass' else 'fail',exit_code=rc)
        except (OSError,ValueError,KeyError) as error:row.update(status='fail',exit_code=rc,error=repr(error))
        if row['status']!='pass':result['status']='fail';write(a.out/'result.json',result);return 1
    result['status']='pass';write(a.out/'result.json',result);return 0
if __name__=='__main__':raise SystemExit(main())
