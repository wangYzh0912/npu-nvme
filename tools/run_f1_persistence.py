#!/usr/bin/env python3
"""Serial cross-environment R0 persistence and failed-flush Host-array fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text());runs=[];counter=0
    def run(profile,operation,label,extra=()):
        nonlocal counter
        out=a.out/label;counter+=1
        command=[sys.executable,'scripts/run_user_environment.py','--manifest',str(a.manifest),'--profile',profile,'--',
            'python','tests/hardware/f1_persistent_probe.py',operation,'--out',str(out),'--library',manifest['profiles'][profile]['library'],
            '--shm-id',str(70000+counter),*extra]
        with (a.out/(label+'.log')).open('w') as log:child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        row=dict(profile=profile,operation=operation,pid=child.pid,out=str(out),status='running');runs.append(row)
        (a.out/'result.json').write_text(json.dumps(dict(status='running',runs=runs),indent=2))
        try:rc=child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            row['status']='retained';(a.out/'result.json').write_text(json.dumps(dict(status='retained',runs=runs),indent=2));raise
        row['exit']=rc;row['status']='pass' if rc==0 else 'fail'
        if rc:raise RuntimeError('F1 operation failed: '+str(out))
        result=json.loads((out/'result.json').read_text())
        if result['status']!='pass' or result.get('closed') is not True:raise ValueError('F1 owner not closed')
        print(label,'pass',flush=True)
        return result
    try:
        inspected=run('old','inspect','inspect')
        header=(a.out/'inspect/header-before.bin').read_bytes()
        if header[:8]!=b'NPUNVM3\0':run('old','initialize','initialize',['--expected-header-sha256',inspected['header_sha256']])
        for profile in ('old','candidate'):
            for fault in (False,True):
                label=profile+('-flush-fault' if fault else '-normal')
                run(profile,'source',label+'-source',['--inject-flush'] if fault else [])
                run(profile,'restore',label+'-restore',['--source-run',str(a.out/(label+'-source'))])
    except BaseException as error:
        (a.out/'result.json').write_text(json.dumps(dict(status='fail',error=repr(error),runs=runs),indent=2));raise
    (a.out/'result.json').write_text(json.dumps(dict(status='pass',scope='real NVMe Host-array fixture; not H06 training acceptance',runs=runs),indent=2))
if __name__=='__main__':main()
