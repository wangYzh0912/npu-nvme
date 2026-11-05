"""Collect graph dumps and independent sample PMU in an isolated training source.

Apply training/dependency-graph-observation.patch to the frozen v4 observation
source before invoking this module. It never patches an active training tree.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from .coordination import ROOT


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--raw',type=Path,default=Path('/dev/shm/user7-aic-aiv-coordination-training'))
    args=p.parse_args()
    workload=args.source/'python/npu_nvme/workloads/qwen.py'
    if 'AIC_AIV_SAVE_DEPENDENCY_GRAPHS' not in workload.read_text():
        raise ValueError('source lacks the explicit dependency-graph observation hook')
    args.raw.mkdir(exist_ok=True)
    for length in (128,512):
        config=json.loads((args.root/'training/dependency-run-config.json').read_text())
        config['seq_length']=length
        path=args.root/'configs'/f'train-{length}-dependency.json'
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(config,indent=2)+'\n')
        command=['/models/npu_nvme_exp/user7-stack/conda/bin/python','-m',
                 'tools.aic_activity.train_sample','--source',str(args.source),
                 '--config',str(path),'--output',str(args.raw/f'train-{length}-dependency')]
        (args.root/'commands'/f'train-{length}-dependency.json').write_text(json.dumps(command,indent=2)+'\n')
        with (args.raw/f'train-{length}-dependency.supervisor.log').open('x') as log:
            subprocess.run(command,cwd=args.source,
                env=dict(os.environ,PYTHONPATH='.',PYTHONDONTWRITEBYTECODE='1',
                         AIC_AIV_SAVE_DEPENDENCY_GRAPHS='1'),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        print('TRAIN_COMPLETED',length,flush=True)

if __name__=='__main__':main()
