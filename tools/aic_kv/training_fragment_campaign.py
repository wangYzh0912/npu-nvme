"""Serialize optimizer fragments through the same hardware ownership protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .training_jit_campaign import RAW,RESULTS,PYTHON


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--resident-gradients',action='store_true');args=p.parse_args()
    worker=Path(__file__).with_name('training_optimizer_fragment.py')
    frozen=RESULTS/'source'/('training_optimizer_fragment_resident.py' if args.resident_gradients else 'training_optimizer_fragment.py')
    if frozen.exists() and frozen.read_bytes()!=worker.read_bytes():
        raise ValueError('fragment source snapshot changed')
    frozen.parent.mkdir(exist_ok=True);shutil.copy2(worker,frozen)
    (RAW/'fragments').mkdir(parents=True,exist_ok=True)
    for shape in ('128','4096','4096x3072'):
        for mode in ('adamw-o0','adamw-o1','fused-o0'):
            for rep in range(3):
                name=f'{mode}-{shape}-r{rep}'
                suite='resident-fragments' if args.resident_gradients else 'fragments'
                raw=RAW/suite/name
                raw.parent.mkdir(parents=True,exist_ok=True)
                durable=RESULTS/suite/'runs'/name;durable.mkdir(parents=True,exist_ok=True)
                command=[PYTHON,str(args.source/'scripts/run_user_environment.py'),'--manifest',
                    '/models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/environments.json',
                    '--profile','candidate','--','python',str(frozen),'--output',str(raw),'--mode',mode,'--shape',shape]
                if rep==0 and shape=='4096x3072':command.append('--trace')
                if args.resident_gradients:command.append('--resident-gradients')
                if raw.exists():
                    if not (raw/'result.json').is_file() or not json.loads((raw/'result.json').read_text())['oracle']['passed']:
                        raise RuntimeError('existing fragment is incomplete or failed: '+name)
                else:
                    env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES='4',PYTHONDONTWRITEBYTECODE='1')
                    with (durable/'run.log').open('x') as log:
                        result=subprocess.run(command,env=env,cwd=raw.parent,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                    if result.returncode:
                        (durable/'failure.json').write_text(json.dumps({'command':command,'returncode':result.returncode}))
                        raise RuntimeError('fragment failed: '+name)
                shutil.copy2(raw/'result.json',durable/'result.json')
                if rep==0:shutil.copy2(raw/'oracle.npz',durable/'oracle.npz')
                for task in raw.glob('trace/**/kernel_details.csv'):
                    import gzip
                    target=durable/('kernel_details-'+str(len(list(durable.glob('kernel_details-*.gz'))))+'.csv.gz')
                    with task.open('rb') as src,gzip.open(target,'wb') as dst:shutil.copyfileobj(src,dst)
                for pattern,label in [('trace/**/ascend_task.db','ascend_task.db.gz'),
                                      ('trace/**/task_time*.csv','task_time.csv.gz')]:
                    sources=list(raw.glob(pattern))
                    if len(sources)>1:raise ValueError('ambiguous profiler task source')
                    if sources:
                        import gzip
                        with sources[0].open('rb') as src,gzip.open(durable/label,'wb') as dst:
                            shutil.copyfileobj(src,dst)
                record=dict(name=name,command=command,replicate=rep,source_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),status='complete')
                (durable/'identity.json').write_text(json.dumps(record,indent=2)+chr(10))
                print(json.dumps(record),flush=True)


if __name__=='__main__':main()
