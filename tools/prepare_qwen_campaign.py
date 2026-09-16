#!/usr/bin/env python3
"""Prepare an explicit serial four-method campaign and its acceptance manifest."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_campaign import write
from tools.validate_qwen_baselines import METHODS


def prepare(checkout, out, environment, shm_base=80000):
    checkout=Path(checkout).resolve();out=Path(out).resolve();environment=Path(environment).resolve()
    stages=[];acceptance=dict(methods={},timing={})
    def stage(method, suffix, source=None, dependency=None):
        name=f'{method}-{suffix}';destination=out/name
        argv=['/usr/bin/python3','train.py','qwen-method','--method',method,'--out',str(destination),
              '--manifest',str(environment),'--shm-base',str(shm_base+10*len(stages))]
        if source:argv+=['--source-run',str(source)]
        stages.append(dict(id=name,cwd=str(checkout),argv=argv,report=str(destination/'result.json'),
                           timeout_seconds=7500,depends_on=[dependency] if dependency else []))
        return destination,name
    for method in METHODS:
        groups=[];timed=[]
        for repeat in range(3):
            source,name=stage(method,f'source-{repeat+1}')
            group=dict(source=str(source),restores=[]);groups.append(group)
            if method=='none':continue
            # First correctness restore also warms up; all measured restores
            # still perform mandatory integrity verification and continuation.
            if repeat==0:
                warmup,_=stage(method,'restore-warmup',source,name)
                group['restores'].append(str(warmup))
            restored,_=stage(method,f'restore-{repeat+1}',source,name)
            group['restores'].append(str(restored));timed.append(str(restored))
        acceptance['methods'][method]=groups
        if method!='none':acceptance['timing'][method]=dict(warmup=str(warmup),measured=timed)
    manifest=out/'acceptance-manifest.json'
    stages.append(dict(id='acceptance',cwd=str(checkout),argv=['/usr/bin/python3','tools/validate_qwen_baselines.py',
        '--manifest',str(manifest),'--out',str(out/'acceptance.json')],report=str(out/'acceptance.json'),
        timeout_seconds=300,depends_on=[row['id'] for row in stages]))
    return dict(inputs=[str(environment),str(manifest)],stages=stages),acceptance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--shm-base',type=int,default=80000)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    plan,acceptance=prepare(a.checkout,a.out,a.manifest,a.shm_base)
    write(a.out/'acceptance-manifest.json',acceptance);write(a.out/'plan.json',plan)
if __name__=='__main__':main()
