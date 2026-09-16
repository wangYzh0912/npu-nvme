#!/usr/bin/env python3
"""Freeze a committed C1 configuration, run software and hardware, then join evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from npu_nvme.cli.contracts import source_identity,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--config',type=Path,default=ROOT/'config/c1/gpt2_frozen.json');p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    if not args.dry_run and os.geteuid()!=0: p.error('hardware acceptance requires the documented root old-environment shell')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.config.read_text());identity=source_identity()
    config['identity'].update(expected_commit=identity['observed_commit'],allow_dirty=False)
    write(out/'frozen_config.json',config)
    commands=[
        [sys.executable,str(ROOT/'tools/run_gate.py'),'--profile','C1','--out',str(out/'software')],
        [sys.executable,str(ROOT/'train.py'),'benchmark','--config',str(out/'frozen_config.json'),'--out',str(out/'hardware')],
        [sys.executable,str(ROOT/'tools/validate_c1_acceptance.py'),'--software',str(out/'software'),'--hardware',str(out/'hardware'),'--out',str(out/'acceptance.json')],
    ]
    if args.dry_run:
        commands=[[sys.executable,str(ROOT/'train.py'),'preflight','--config',str(out/'frozen_config.json'),'--out',str(out/'preflight'),'--dry-run']]
    phases=[]
    for command in commands:
        env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'python')+':'+os.environ.get('PYTHONPATH',''))
        if command[1]==str(ROOT/'tools/run_gate.py'):
            env['PYTHONPATH']+=':/home/user7/.local/lib/python3.9/site-packages'
        rc=subprocess.call(command,cwd=ROOT,env=env)
        phases.append(dict(argv=command,returncode=rc,pythonpath=env['PYTHONPATH']));write(out/'phases.json',phases)
        if rc: return rc
    return 0

if __name__=='__main__':raise SystemExit(main())
