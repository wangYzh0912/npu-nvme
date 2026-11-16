"""Wait for the rejected pilot's safe exit, then freeze and start corrected E1."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/tmp/npu-nvme-checkpoint-io-closure')
BASE=Path('/home/user7/npu-nvme-evidence')
OLD=BASE/'aic-aiv-io-20261116'
NEW=BASE/'aic-aiv-io-20261116-r1'
STATE=BASE/'aic-aiv-io-20261116-restart.json'
EXPECTED='0bac9dfec10cc75fd1f29b962bb90bd437bfc8e3'
TARGET=sys.argv[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT)]
from npu_nvme.runtime.training_catalog import read_checked


def save(value):
    value=dict(value,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+chr(10));temporary.replace(STATE)


def process(pid):
    try:fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except (FileNotFoundError,ProcessLookupError):return None
    return fields[0],int(fields[19])


def main():
    launcher=json.loads(OLD.with_suffix('.launcher.json').read_text())
    pid=launcher['pid'];identity=process(pid)
    if launcher['source_commit']!=EXPECTED:raise RuntimeError('pilot identity differs')
    while identity is not None:
        current=process(pid)
        if current is None or current[0]=='Z' or current[1]!=identity[1]:break
        result=json.loads((OLD/'result.json').read_text())
        save(dict(status='waiting_for_pilot_safe_exit',pilot_pid=pid,target_commit=TARGET,
                  pilot_status=result['status'],pilot_phase=result['phase']))
        time.sleep(15)
    result=json.loads((OLD/'result.json').read_text())
    if result['status']!='failed' or result.get('error')!="ValueError('resource samples missing rank processes')":
        raise RuntimeError('pilot outcome differs from reproduced sampling defect: '+str(result.get('error')))
    raw=Path(result['active_run']['raw']);fit=json.loads((raw/'result.json').read_text())
    if fit['validation_status']!='pass' or fit['execution_status']!='completed':
        raise RuntimeError('pilot training did not close successfully')
    end=fit['end_to_end']
    if end['children_remaining'] or end['retained'] or not end['lease_released']:
        raise RuntimeError('pilot child/retention/lease proof missing')
    for rank in range(4):
        report=read_checked(raw/f'rank_{rank}/training.json')
        if report['status']!='pass' or not report.get('closed') or report['final_step']!=32:
            raise RuntimeError('pilot rank safe completion differs')
        rank_process=process(report['pid'])
        if rank_process is not None and rank_process[0]!='Z':raise RuntimeError('pilot rank still alive')
    if Path('/models/npu_nvme_exp/user7-stack/hardware-campaign.lease.json').exists():
        raise RuntimeError('hardware lease still present; do not override')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    if head!=EXPECTED or dirty:raise RuntimeError('execution worktree changed during pilot')
    save(dict(status='pilot_closed_switching_frozen_source',target_commit=TARGET))
    subprocess.run(['git','merge','--ff-only',TARGET],cwd=ROOT,check=True)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=TARGET:
        raise RuntimeError('source fast-forward identity differs')
    command=['/usr/bin/python3',str(ROOT/'tools/checkpoint_io_closure.py'),'queue']
    with NEW.with_suffix('.queue.log').open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH='python:.',PYTHONUNBUFFERED='1'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,command=command,cwd=str(ROOT),source_commit=TARGET,
        launched_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),results=str(NEW))
    NEW.with_suffix('.launcher.json').write_text(json.dumps(record,indent=2)+chr(10))
    save(dict(status='corrected_queue_launched',**record))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        save(dict(status='failed',error=repr(error),target_commit=TARGET));raise
