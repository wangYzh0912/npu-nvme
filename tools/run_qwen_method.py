#!/usr/bin/env python3
"""One fixed TP4 baseline source or fresh restore with a persisted verdict."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import threading
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from tools.run_campaign import members,write


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',required=True,choices=['none','mindspore_native_save','ours','bytecheckpoint_host'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source-run',type=Path)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--shm-base',type=int,default=75000)
    p.add_argument('--timeout',type=float,default=7200)
    a=p.parse_args(argv);a.out=a.out.resolve()
    if a.source_run:a.source_run=a.source_run.resolve()
    if a.method=='ours':
        from tools.run_qwen_d2 import main as run
        args=['--out',str(a.out),'--manifest',str(a.manifest),'--shm-base',str(a.shm_base),'--timeout',str(int(a.timeout))]
        if a.source_run:args+=['--source-run',str(a.source_run)]
        rc=run(args)
        launch=json.loads((a.out/'launch.json').read_text())
        if launch.get('status')=='retained':
            write(a.out/'result.json',dict(execution_status='retained',validation_status='fail',alive=launch.get('alive',[])))
            threading.Event().wait()
        result=json.loads((a.out/'d2-acceptance.json').read_text()) if (a.out/'d2-acceptance.json').exists() else dict(status='fail')
        write(a.out/'result.json',dict(execution_status='completed',validation_status=result['status'],method=a.method,exit_code=rc))
        return rc
    a.out.mkdir(parents=True,exist_ok=False)
    result=dict(execution_status='running',validation_status='not_run',method=a.method)
    write(a.out/'result.json',result)
    smi=subprocess.check_output(['npu-smi','info'],text=True)
    if any(f'No running processes found in NPU {r}' not in smi for r in range(4)):
        raise RuntimeError('TP4 NPU occupied')
    (a.out/'npu-before.txt').write_text(smi)
    profile=json.loads(a.manifest.read_text())['profiles']['candidate']
    identity=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        library_sha256=hashlib.sha256(Path(profile['library']).read_bytes()).hexdigest(),profile=profile)
    write(a.out/'environment.lock.json',identity)
    s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close()
    command=[sys.executable,'scripts/run_user_environment.py','--manifest',str(a.manifest),'--profile','candidate','--','python',
        '-m','mindspore.parallel.cluster.run','--worker_num=4','--local_worker_num=4',
        '--master_addr=127.0.0.1','--master_port='+str(port),'--join=True','--cluster_time_out=300',
        '--log_dir='+str(a.out/'msrun-log'),'python',str(ROOT/'experiments/training/train_qwen3_full_restart.py'),
        '--output',str(a.out),'--checkpoint-step','8','--source-stop-step','11','--lr-horizon','11',
        '--seq-length','128','--deterministic','--checkpoint-method',a.method]
    if a.source_run:command+=['--resume-run',str(a.source_run)]
    env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES='0,1,2,3',RUN_MODE='finetune',PYTHONUNBUFFERED='1',
             HCCL_CONNECT_TIMEOUT='300',MS_COMPILER_CACHE_PATH=str(a.out/'compiler-cache'))
    write(a.out/'command.json',command)
    with (a.out/'training.log').open('w') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        result['pid']=child.pid;write(a.out/'result.json',result)
        try:rc=child.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            result.update(execution_status='retained',validation_status='fail',alive=members(child.pid))
            write(a.out/'result.json',result);threading.Event().wait()
    if members(child.pid):
        result.update(execution_status='retained',validation_status='fail',alive=members(child.pid))
        write(a.out/'result.json',result);threading.Event().wait()
    try:
        if rc:raise RuntimeError('training process failed')
        if a.method=='bytecheckpoint_host':
            write(a.out/'owner/result.json',dict(status='pass',closed=True,receipt=dict(generation=8),
                scope='Host-port global manifest after all four exited rank-local upstream workers'))
            from experiments.training.check_qwen_d2_run import audit
            checked=audit(a.out,a.source_run,backend='bytecheckpoint_host')
            write(a.out/'d2-acceptance.json',checked)
        elif a.method=='mindspore_native_save':
            check=[sys.executable,'experiments/training/check_qwen_training_run.py','--output',str(a.out)]
            if a.source_run:check+=['--source-run',str(a.source_run)]
            subprocess.run(check,cwd=ROOT,check=True)
            if not a.source_run:
                from tools.prepare_qwen_restart import prepare
                write(a.out/'restart_contract.json',prepare(a.out,8,11))
        else:
            for rank in range(4):
                r=json.loads((a.out/f'rank_{rank}/acceptance.json').read_text())
                if r['checkpoint_backend']!='none' or r['status']!='training_pass_restart_not_tested' or len(r['losses'])!=11 or r['checkpoint_files']:
                    raise ValueError('none baseline wrote checkpoint or failed training')
        result.update(execution_status='completed',validation_status='pass',exit_code=rc)
    except Exception as error:
        result.update(execution_status='completed',validation_status='fail',exit_code=rc,error=repr(error))
    write(a.out/'result.json',result)
    return int(result['validation_status']!='pass')
if __name__=='__main__':raise SystemExit(main())
