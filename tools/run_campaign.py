#!/usr/bin/env python3
"""Serial, resumable campaign: process exit never substitutes for validation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def members(group):
    result=[]
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rsplit(')',1)[1].split()
            if int(fields[2])==group and fields[0]!='Z':result.append(dict(pid=int(path.parent.name),start_ticks=int(fields[19])))
        except (OSError,ValueError,IndexError):continue
    return result

def source_snapshot(plan):
    """Hash tracked working bytes as well as HEAD; never accept checkout drift."""
    roots = set()
    for stage in plan['stages']:
        probe = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=stage['cwd'],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            roots.add(probe.stdout.strip())
    result = {}
    for root in sorted(roots):
        paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).split(b'\0')
        digest = hashlib.sha256()
        for name in sorted(filter(None, paths)):
            path = Path(root) / os.fsdecode(name)
            digest.update(name + b'\0')
            digest.update(hashlib.sha256(path.read_bytes()).digest() if path.is_file() else b'MISSING')
        result[root] = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                            cwd=root, text=True).strip(), tracked_sha256=digest.hexdigest())
    return result


def validate_plan(plan):
    seen=set()
    for row in plan['stages']:
        name=row['id']
        if not name or '/' in name or name in ('.','..') or name in seen:raise ValueError('stage identity')
        if not set(row.get('depends_on',[]))<=seen:raise ValueError('dependency must precede stage')
        if not row['argv'] or not all(isinstance(x,str) for x in row['argv']):raise ValueError('argv')
        if not 0<float(row['timeout_seconds'])<float('inf'):raise ValueError('stage deadline')
        if not Path(row['cwd']).is_absolute() or not Path(row['report']).is_absolute():raise ValueError('absolute stage paths required')
        seen.add(name)

def run(plan,out,lock_path):
    validate_plan(plan);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    lock_path=Path(lock_path);lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        inputs={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in plan['inputs']}
        sources=source_snapshot(plan)
        fingerprint=signature(dict(plan=plan,inputs=inputs,sources=sources));state_path=out/'result.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else dict(fingerprint=fingerprint,stages={})
        if state['fingerprint']!=fingerprint:raise ValueError('campaign input/config changed; use a new run directory')
        state['status']='running';state['sources']=sources;write(state_path,state)
        # Durable lease prevents a different campaign from reusing the device
        # after a supervisor timeout/crash. The executable also inherits flock.
        lease_path=lock_path.with_suffix('.lease.json')
        if lease_path.exists():
            lease=json.loads(lease_path.read_text())
            if members(lease['process_group']):raise RuntimeError('previous campaign process group retained')
        for stage in plan['stages']:
            name=stage['id'];old=state['stages'].get(name,{})
            if source_snapshot(plan)!=sources:
                state.update(status='fail',error='tracked source changed between stages; new frozen attempt required')
                write(state_path,state);return 1
            if old.get('status')=='pass':
                report=Path(stage['report'])
                if not report.exists() or hashlib.sha256(report.read_bytes()).hexdigest()!=old['report_sha256']:
                    raise ValueError('accepted stage artifact changed')
                continue
            if old.get('status') in ('running','retained'):
                # A crashed supervisor cannot establish successful completion.
                old.update(status='retained' if members(old['process_group']) else 'fail',error='supervisor interrupted; explicit new attempt required')
                state['status']=old['status'];write(state_path,state);return 3
            if old.get('status')=='fail':state['status']='fail';write(state_path,state);return 1
            if any(state['stages'][dep]['status']!='pass' for dep in stage.get('depends_on',[])):
                state['stages'][name]=dict(status='blocked',reason='dependency validation failed');continue
            directory=out/name;directory.mkdir(exist_ok=True)
            with (directory/'stdout.log').open('a') as stdout,(directory/'stderr.log').open('a') as stderr:
                child=subprocess.Popen(stage['argv'],cwd=stage['cwd'],stdout=stdout,stderr=stderr,
                    start_new_session=True,pass_fds=(lock.fileno(),))
                start=time.monotonic();row=dict(status='running',pid=child.pid,process_group=child.pid,started_at=time.time())
                state['stages'][name]=row;write(lease_path,dict(campaign=str(out.resolve()),**row));write(state_path,state)
                while child.poll() is None:
                    row['heartbeat_at']=time.time();row['members']=members(child.pid);write(state_path,state)
                    if time.monotonic()-start>stage['timeout_seconds']:
                        row.update(status='retained',error='deadline exceeded; no automatic process/DMA cleanup')
                        state['status']='retained';write(state_path,state);return 3
                    time.sleep(.2)
                row.update(exit_code=child.returncode,seconds=time.monotonic()-start,members=members(child.pid))
                if row['members']:
                    row.update(status='retained',error='descendants still alive');state['status']='retained';write(state_path,state);return 3
            try:
                report=Path(stage['report']);raw=report.read_bytes();result=json.loads(raw)
                passed=child.returncode==0 and result.get(stage.get('verdict_key','validation_status'))=='pass'
                row.update(status='pass' if passed else 'fail',report_sha256=hashlib.sha256(raw).hexdigest(),report=str(report))
            except (OSError,ValueError) as error:row.update(status='fail',error=repr(error))
            lease_path.unlink(missing_ok=True)
            if source_snapshot(plan)!=sources:
                row.update(status='fail',error='tracked source changed during execution; result not accepted')
                state['status']='fail';write(state_path,state);return 1
            write(state_path,state)
        state['status']='pass' if all(v['status']=='pass' for v in state['stages'].values()) else 'fail'
        write(state_path,state);return int(state['status']!='pass')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--lock',type=Path,required=True)
    args=parser.parse_args();return run(json.loads(args.plan.read_text()),args.out,args.lock)
if __name__=='__main__':raise SystemExit(main())
