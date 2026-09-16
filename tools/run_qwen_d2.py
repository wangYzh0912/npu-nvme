#!/usr/bin/env python3
"""Launch explicit blocking TP4 D2 source or fresh-process continuation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time
import uuid
ROOT=Path(__file__).resolve().parents[1]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--source-run',type=Path)
    p.add_argument('--timeout',type=int,default=7200);p.add_argument('--shm-base',type=int,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv);a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.read_text());profile=manifest['profiles']['candidate']
    library=profile['library'];epoch=uuid.uuid4().hex;socket_path='/tmp/qwen-d2-'+epoch+'.sock'
    identity=dict(model='Qwen3-8B',topology=dict(tp=4,dp=1,pp=1),seed=42,sequence_length=128,
                  state_scope='full_state',params_dtype='float32',compute_dtype='bfloat16',dropout=0,lr_horizon=11)
    connection=dict(socket=socket_path,epoch=epoch,identity=identity,capture='blocking',world_size=4,
        rank_shm_base=a.shm_base+1,timeout_seconds=a.timeout,chunk_bytes=4<<20,depth=4,
        host_tensor_budget_bytes=2<<30,step=8)
    identity_path=a.out/'identity.json';identity_path.write_text(json.dumps(identity,indent=2))
    connection_path=a.out/'connection.json';connection_path.write_text(json.dumps(connection,indent=2))
    strategy=ROOT/'config/qwen_runtime_schema.json'
    sys.path.insert(0,str(ROOT/'python'))
    from npu_nvme.d2.tp_schema import validate,digest
    strategy_value=json.loads(strategy.read_text());validate(strategy_value)
    identity['strategy_sha256']=digest(strategy_value)
    connection['strategy_path']=str(strategy)
    operation='restore' if a.source_run else 'save'
    identity_path.write_text(json.dumps(identity,indent=2))
    connection_path.write_text(json.dumps(connection,indent=2))
    prefix=[sys.executable,str(ROOT/'scripts/run_user_environment.py'),'--manifest',str(a.manifest),'--profile','candidate','--','python']
    owner_args=['tools/d2_session_owner.py','--operation',operation,'--out',str(a.out/'owner'),'--socket',socket_path,
        '--library',library,'--strategy',str(strategy),'--identity',str(identity_path),'--epoch',epoch,'--request-id',epoch,'--shm-id',str(a.shm_base),'--timeout',str(a.timeout)]
    if a.source_run:
        source=a.source_run.resolve();contract=json.loads((source/'d2_restart_contract.json').read_text())
        owner_args+=['--generation',str(contract['generation'])]
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    train_args=['-m','mindspore.parallel.cluster.run','--worker_num=4','--local_worker_num=4','--master_addr=127.0.0.1',
        '--master_port='+str(port),'--join=True','--cluster_time_out=300','--tail_worker_log=4','--log_dir='+str(a.out/'msrun-log'),
        'python',str(ROOT/'experiments/training/train_qwen3_full_restart.py'),'--output',str(a.out),
        '--checkpoint-step','8','--source-stop-step','11','--lr-horizon','11','--seq-length','128','--deterministic',
        '--d2-connection',str(connection_path)]
    if a.source_run:train_args+=['--resume-run',str(source)]
    record=dict(status='planned',owner=prefix+owner_args,training=prefix+train_args,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest(),
        library_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest())
    def save():(a.out/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    if a.dry_run:return 0
    smi=subprocess.check_output(['npu-smi','info'],text=True);(a.out/'npu-before.txt').write_text(smi)
    for rank in range(4):
        if f'No running processes found in NPU {rank}' not in smi:raise RuntimeError('Qwen rank occupied')
    env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES='0,1,2,3',RUN_MODE='finetune',PYTHONUNBUFFERED='1',
             HCCL_CONNECT_TIMEOUT='300',MS_COMPILER_CACHE_PATH=str(a.out/'compiler-cache'))
    with (a.out/'owner.log').open('w') as log:owner=subprocess.Popen(prefix+owner_args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    record.update(status='running',owner_pid=owner.pid);save();deadline=time.monotonic()+a.timeout
    while not (a.out/'owner/ready.json').exists():
        if owner.poll() is not None:raise RuntimeError('D2 owner initialization failed')
        if time.monotonic()>deadline:raise TimeoutError('D2 owner retained after ready deadline')
        time.sleep(.1)
    with (a.out/'training.log').open('w') as log:training=subprocess.Popen(prefix+train_args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    record['training_pid']=training.pid;save()
    try:
        training_exit=training.wait(timeout=max(.001,deadline-time.monotonic()))
        if training_exit and owner.poll() is None:
            # Python SIGINT unwinds accept/receive through the owner's checked close.
            owner.send_signal(signal.SIGINT)
        exits=[training_exit,owner.wait(timeout=max(.001,deadline-time.monotonic()))]
    except subprocess.TimeoutExpired:
        record.update(status='retained',alive=[c.pid for c in (owner,training) if c.poll() is None]);save();return 3
    record.update(status='pass' if exits==[0,0] else 'fail',exits=exits);save()
    if exits!=[0,0]:return 1
    command=[sys.executable,'experiments/training/check_qwen_d2_run.py','--out',str(a.out)]
    if a.source_run:command+=['--source-run',str(source)]
    return subprocess.call(command,cwd=ROOT)
if __name__=='__main__':raise SystemExit(main())
