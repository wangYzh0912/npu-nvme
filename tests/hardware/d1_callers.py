#!/usr/bin/env python3
"""Fresh-process smoke acceptance for the migrated single-card and Ours callers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]


def write(path,value): path.write_text(json.dumps(value,indent=2,default=str)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['matrix','prepare','source','restore'],default='matrix')
    args=p.parse_args(); out=args.out.resolve()
    if os.geteuid()!=0: p.error('root required')
    if args.phase!='matrix':
        from experiments.baselines.repro.runner import prepare,run,restore
        config=json.loads((out/'ours-config.json').read_text())
        if args.phase=='prepare': prepare(config,out/'ours/prepare')
        elif args.phase=='source': run(config,'ours',out/'ours/run')
        else:
            result=restore(config,'ours',out/'ours/run')
            if result['status']!='pass': raise AssertionError(result)
        return 0
    out.mkdir(parents=True,exist_ok=False)
    config=json.loads((ROOT/'experiments/baselines/repro/configs/gpt2_30step.json').read_text())
    config.update(project_root=str(ROOT),results_root=str(out/'ours'),fs_test_dir=str(out/'ours/files'),
                  warmup_steps=1,formal_steps=2,checkpoint_every=2,continue_steps=3,spdk_shm_id=46100)
    write(out/'ours-config.json',config)
    source={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for directory in ('python','src','include','tests','experiments')
            for f in sorted((ROOT/directory).rglob('*')) if f.is_file() and f.suffix in ('.py','.c','.h')}
    write(out/'sources.json',source)
    phases=[]
    commands=[('single-card',[sys.executable,str(ROOT/'experiments/benchmarks/run_single_card_full.py'),
        '--run-dir',str(out/'single-card'),'--smoke','--total-steps','5','--checkpoint-steps','2',
        '--mode','serial','--shm-id','45100','--timeout','180'])]
    commands += [(phase,[sys.executable,str(Path(__file__).resolve()),'--out',str(out),'--phase',phase])
                 for phase in ('prepare','source','restore')]
    for phase,argv in commands:
        print('D1 caller '+phase,flush=True)
        with (out/(phase+'.log')).open('w') as log:
            process=subprocess.run(argv,cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        phases.append(dict(phase=phase,argv=argv,returncode=process.returncode))
        write(out/'phases.json',phases)
        if process.returncode:
            write(out/'result.json',dict(status='fail',phases=phases)); return 1
    changed=[k for k,v in source.items() if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v]
    write(out/'result.json',dict(status='pass' if not changed else 'invalid',phases=phases,changed_sources=changed,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        library_sha256=hashlib.sha256((ROOT/'build_out/lib/libnpu_nvme.so').read_bytes()).hexdigest(),
        scope='single-card serial frozen FULL and Ours source/fresh restore; seed41, GPT2'))
    return int(bool(changed))

if __name__=='__main__': raise SystemExit(main())
