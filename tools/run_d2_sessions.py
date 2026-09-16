#!/usr/bin/env python3
"""Serial 2/4-rank save/exit/restore fixtures in old and candidate runtimes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--shm-base',type=int,default=68000)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text());runs=[];counter=0
    sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for directory in ('python','tools','tests/hardware') for f in (ROOT/directory).rglob('*.py')}
    (a.out/'sources.json').write_text(json.dumps(sources,indent=2))
    def write():
        (a.out/'result.json').write_text(json.dumps(dict(status='running',runs=runs),indent=2))
    def launch(profile,args,log):
        command=[sys.executable,str(ROOT/'scripts/run_user_environment.py'),'--manifest',str(a.manifest),
                 '--profile',profile,'--','python',*args]
        stream=log.open('w');proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT);stream.close()
        return proc
    for profile in ('old','candidate'):
        for world in (2,4):
            group=a.out/f'{profile}-tp{world}';group.mkdir()
            identity=group/'identity.json';identity.write_text(json.dumps(dict(scope='D2 fresh process fixture',world_size=world)))
            generation=None
            for operation in ('save','restore'):
                smi=subprocess.check_output(['npu-smi','info'],text=True)
                for rank in range(world):
                    if f'No running processes found in NPU {rank}' not in smi:raise RuntimeError('fixture rank occupied')
                out=group/operation;out.mkdir();(out/'npu-before.txt').write_text(smi)
                epoch=uuid.uuid4().hex;path='/tmp/d2-'+epoch+'.sock';shm=a.shm_base+counter*10;counter+=1
                lib=manifest['profiles'][profile]['library']
                args=['tools/d2_session_owner.py','--operation',operation,'--out',str(out/'owner'),
                    '--socket',path,'--library',lib,'--identity',str(identity),'--epoch',epoch,
                    '--request-id',epoch,'--world-size',str(world),'--shm-id',str(shm),'--timeout','300']
                if generation is not None:args+=['--generation',str(generation)]
                owner=launch(profile,args,out/'owner.log');children=[owner]
                record=dict(profile=profile,world_size=world,operation=operation,pids=[owner.pid],status='running');runs.append(record);write()
                deadline=time.monotonic()+360
                while not (out/'owner/ready.json').exists():
                    if owner.poll() is not None:raise RuntimeError('owner startup failed; see '+str(out/'owner.log'))
                    if time.monotonic()>deadline:raise TimeoutError('owner ready timeout; owner retained')
                    time.sleep(.1)
                for rank in range(world):
                    args=['tests/hardware/d2_session_rank.py','--operation',operation,'--socket',path,
                        '--epoch',epoch,'--library',lib,'--identity',str(identity),'--rank',str(rank),
                        '--shm-id',str(shm+rank+1),'--out',str(out/f'rank-{rank}')]
                    child=launch(profile,args,out/f'rank-{rank}.log');children.append(child);record['pids'].append(child.pid)
                write()
                try:
                    exits=[child.wait(timeout=max(.001,deadline-time.monotonic())) for child in children]
                except subprocess.TimeoutExpired:
                    record.update(status='retained',alive=[c.pid for c in children if c.poll() is None]);write();return 3
                record['exits']=exits
                if any(exits):record['status']='fail';write();return 1
                owner_result=json.loads((out/'owner/result.json').read_text())
                if owner_result['status']!='pass' or owner_result.get('closed') is not True:raise ValueError('owner lacks success/close')
                if operation=='save':generation=owner_result['receipt']['generation']
                record.update(status='pass',generation=generation);write();print(profile,world,operation,'pass',flush=True)
    changed=[name for name,digest in sources.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
    result=dict(status='fail' if changed else 'pass',runs=runs,changed_sources=changed,
                scope='HW small deterministic states; separate source and restore processes; no training acceptance')
    (a.out/'result.json').write_text(json.dumps(result,indent=2))
    return int(bool(changed))
if __name__=='__main__':raise SystemExit(main())
