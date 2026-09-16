"""Unified Qwen3 baseline entry; no framework import in the supervisor."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from npu_nvme.runtime import training_catalog as catalog
from npu_nvme.runtime.qwen_config import resolve,METHODS

ROOT=Path(__file__).resolve().parents[3]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def preflight(config):
    from user_environment import read_profile,with_python_identity,isolated_environment,verify_library_paths,verify_runtime_abi
    profile=with_python_identity(read_profile(config['environment_manifest'],'candidate',ROOT))
    env=isolated_environment(profile,ROOT)
    resolved=verify_library_paths(profile,env);abi=verify_runtime_abi(profile,env)
    code="from npu_nvme.framework import _address_native; import mindspore; assert _address_native.framework_version == mindspore.__version__; print(mindspore.__version__)"
    subprocess.run([profile['python'],'-c',code],env=env,check=True,capture_output=True,text=True,timeout=60)
    model=Path(config['model']);index=json.loads((model/'model.safetensors.index.json').read_text())
    shards=[]
    for name in sorted(set(index['weight_map'].values())):
        path=(model/name).resolve()
        if model not in path.parents:raise ValueError('model shard escapes model directory')
        shards.append(dict(name=name,bytes=path.stat().st_size,sha256=catalog.digest_file(path)))
    config['identity']['weights_sha256']=hashlib.sha256(catalog.canonical(shards)).hexdigest()
    template=Path(profile['python']).parent.parent/'lib/python3.11/site-packages/configs/qwen3/finetune_qwen3.yaml'
    config['identity']['framework_template_sha256']=catalog.digest_file(template)
    config['identity']['environment_id']=profile['environment_id']
    smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=30)
    idle=all(f'No running processes found in NPU {r}' in smi for r in range(4))
    lease=Path(config['hardware_lock']).with_suffix('.lease.json')
    return dict(status='pass',profile=profile,resolved_libraries=resolved,runtime_abi=abi,
        model_shards=shards,npu_info=smi,devices_idle=idle,pending_lease=str(lease) if lease.exists() else None)


def run_fit(config, output, *, restore='latest', resume=False):
    from tools.run_campaign import members,source_snapshot
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    config=dict(config,output=str(output),run_id=uuid.uuid4().hex,start_step=0)
    config.setdefault('checkpoint_root',str(output/'checkpoints'))
    result=dict(execution_status='running',validation_status='not_run',method=config['method'])
    write(output/'result.json',result)
    children=[];lease=None;lease_owned=False
    with ExitStack() as stack:
        restore_pin=stack.enter_context(ExitStack())
        try:
            lock_path=Path(config['hardware_lock']);lock_path.parent.mkdir(parents=True,exist_ok=True)
            lock=stack.enter_context(lock_path.open('a+'));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            lease=lock_path.with_suffix('.lease.json')
            if lease.exists():raise RuntimeError('unreconciled hardware lease: '+str(lease))
            snapshot_plan=dict(stages=[dict(cwd=str(ROOT))])
            sources=source_snapshot(snapshot_plan)
            extensions={str(path.relative_to(ROOT)):catalog.digest_file(path)
                        for path in (ROOT/'python/npu_nvme/framework').glob('_address_native*.so')}
            write(output/'source-identity.json',dict(sources=sources,extensions=extensions))
            audit=preflight(config);write(output/'preflight.json',audit)
            if not audit['devices_idle']:raise RuntimeError('TP4 devices are occupied')
            def select_restore(media_receipts=None):
                if not resume:return
                if config['method']=='none':raise ValueError('none has no persistent checkpoint')
                def validate(path,value):
                    if value['identity']!=config['identity'] or value['method']!=config['method']:
                        raise ValueError('checkpoint identity differs')
                    if config['method']=='ours':
                        receipt=value['ranks'][0]['backend']['receipt']
                        if not any(row['generation']==receipt['generation'] and row['identity']==config['identity']
                                   for row in media_receipts):
                            raise ValueError('raw media generation is no longer retained')
                    for rank in value['ranks']:catalog.verify_files(path/f"rank_{rank['rank']}",rank['files'])
                rejected=[]
                path,value=restore_pin.enter_context(catalog.selected(config['checkpoint_root'],restore,validate=validate,rejected=rejected))
                config.update(restore_checkpoint=str(path),start_step=value['step'])
                write(output/'restore-selection.json',dict(path=str(path),generation=value['generation'],rejected=rejected))
            prefix=[sys.executable,str(ROOT/'scripts/run_user_environment.py'),'--manifest',config['environment_manifest'],
                    '--profile','candidate','--','python']
            env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES='0,1,2,3',RUN_MODE='finetune',PYTHONUNBUFFERED='1',
                HCCL_CONNECT_TIMEOUT='300',MS_COMPILER_CACHE_PATH=str(output/'compiler-cache'))
            deadline=time.monotonic()+config['timeout_seconds']
            owned_lease=dict(status='running',run=str(output),pid=os.getpid(),children=[])
            write(lease,owned_lease);lease_owned=True
            def launch(argv,name):
                log=stack.enter_context((output/f'{name}.log').open('w'))
                child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=True,pass_fds=(lock.fileno(),))
                children.append(child);owned_lease['children'].append(dict(pid=child.pid,process_group=child.pid))
                write(lease,owned_lease);write(output/f'{name}-command.json',argv)
                return child
            owner=None
            if config['method']=='ours':
                if os.geteuid()!=0:raise PermissionError('raw SPDK owner requires root execution')
                epoch=uuid.uuid4().hex
                connection=dict(socket='/tmp/qwen-training-'+epoch+'.sock',epoch=epoch,
                    identity=config['identity'],library=audit['profile']['library'],
                    region=json.loads(Path(config['region']).read_text()),strategy=json.loads(Path(config['strategy']).read_text()),
                    retention=config['retention'],owner_shm_id=config['shm_base'],rank_shm_base=config['shm_base']+1,
                    timeout_seconds=config['timeout_seconds'],operation_timeout_seconds=config['operation_timeout_seconds'],
                    copy_timeout_ms=config['copy_timeout_ms'],depth=4,chunk_bytes=4<<20,host_tensor_budget_bytes=2<<30)
                config['d2_connection']=str(output/'connection.json');write(config['d2_connection'],connection)
                owner=launch(prefix+[str(ROOT/'tools/training_owner.py'),'--connection',config['d2_connection'],'--out',str(output/'owner')],'owner')
                while not (output/'owner/ready.json').exists():
                    if owner.poll() is not None:raise RuntimeError('owner initialization failed')
                    if time.monotonic()>deadline:raise TimeoutError('owner readiness timeout')
                    time.sleep(.1)
            try:
                receipts=json.loads((output/'owner/ready.json').read_text())['receipts'] if owner else None
                select_restore(receipts)
                if config['start_step']>=config['stop_step']:raise ValueError('stop_step must exceed restored step')
            except BaseException:
                if owner and owner.poll() is None:
                    owner.send_signal(signal.SIGINT)
                    owner.wait(timeout=config['operation_timeout_seconds'])
                raise
            write(output/'run-config.json',config)
            sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
            training=launch(prefix+['-m','mindspore.parallel.cluster.run','--worker_num=4','--local_worker_num=4',
                '--master_addr=127.0.0.1','--master_port='+str(port),'--join=True','--cluster_time_out=300',
                '--log_dir='+str(output/'msrun-log'),'python','-m','npu_nvme.workloads.qwen','--config',str(output/'run-config.json')],'training')
            released=not resume
            while training.poll() is None:
                if resume and not released and all((output/f'restore-ready-{r}.json').exists() for r in range(4)):
                    for rank in range(4):
                        ready=catalog.read_checked(output/f'restore-ready-{rank}.json')
                        if ready!=dict(rank=rank,run_id=config['run_id'],step=config['start_step']):
                            raise ValueError('restore ready identity differs')
                    restore_pin.close();released=True
                    catalog.write_checked(output/'restore-pin-released.json',dict(run_id=config['run_id']))
                if time.monotonic()>=deadline:raise subprocess.TimeoutExpired(training.args,config['timeout_seconds'])
                time.sleep(.1)
            if training.returncode and owner and owner.poll() is None:owner.send_signal(signal.SIGINT)
            if owner:owner.wait(timeout=max(.001,deadline-time.monotonic()))
            if any(child.returncode or members(child.pid) for child in children):raise RuntimeError('training or owner failed / descendants retained')
            reports=[catalog.read_checked(output/f'rank_{r}/training.json') for r in range(4)]
            if any(row['status']!='pass' or row['final_step']!=config['stop_step'] for row in reports):
                raise RuntimeError('rank training completion differs')
            if owner:
                owner_report=json.loads((output/'owner/result.json').read_text())
                if owner_report['status']!='pass' or not owner_report['closed']:raise RuntimeError('owner close not verified')
            if source_snapshot(snapshot_plan)!=sources or any(catalog.digest_file(ROOT/name)!=digest for name,digest in extensions.items()):
                raise RuntimeError('training source or allocation extension changed during execution')
            result.update(execution_status='completed',validation_status='pass',final_step=config['stop_step'])
        except BaseException as error:
            result.update(execution_status='completed',validation_status='fail',error=repr(error))
        finally:
            alive=[dict(pid=child.pid,members=members(child.pid)) for child in children
                   if child.poll() is None or members(child.pid)]
            retained=[]
            for path in output.rglob('*status.json'):
                try:
                    if json.loads(path.read_text()).get('status')=='retained':retained.append(str(path))
                except (ValueError,OSError):pass
            if alive or retained:
                result.update(execution_status='retained',alive=alive,retained=retained)
                write(output/'result.json',result)
                # Retain the lock and mappings until deliberate device recovery.
                threading.Event().wait()
            if lease_owned:lease.unlink(missing_ok=True)
            write(output/'result.json',result)
    return int(result['validation_status']!='pass')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('preflight','fit','inspect','verify-restart','benchmark'):
        item=sub.add_parser(command);item.add_argument('--config',type=Path,required=True)
        item.add_argument('--output',type=Path)
        item.add_argument('--method',choices=METHODS)
        if command=='benchmark':item.add_argument('--profile',choices=('all','multicycle','repeated'),default='all')
        if command in ('fit','verify-restart'):
            item.add_argument('--resume',action='store_true');item.add_argument('--generation',default='latest')
            item.add_argument('--stop-step',type=int)
            item.add_argument('--checkpoint-root',type=Path)
            if command=='verify-restart':item.add_argument('--oracle-run',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        value=json.loads(args.config.read_text())
        for name in ('method','stop_step','checkpoint_root'):
            item=getattr(args,name,None)
            if item is not None:value[name]=str(item) if isinstance(item,Path) else item
        config=resolve(value,ROOT)
        if args.command=='preflight':
            result=preflight(config)
            if args.output:write(args.output,result)
            print(json.dumps(result,indent=2));return 0
        if args.command=='inspect':
            rejected=[];rows=catalog.committed(config['checkpoint_root'],rejected=rejected)
            print(json.dumps(dict(checkpoints=[dict(generation=g,path=str(p),step=v['step']) for g,p,v in rows],rejected=rejected),indent=2));return 0
        if args.command in ('fit','verify-restart'):
            if args.output is None:raise ValueError('--output required')
            rc=run_fit(config,args.output,restore=args.generation,resume=args.resume or args.command=='verify-restart')
            if rc or args.command=='fit':return rc
            from npu_nvme.runtime.training_validation import compare
            write(args.output/'restart-verification.json',compare(args.oracle_run,args.output));return 0
        if args.command=='benchmark':
            if args.output is None:raise ValueError('--output required')
            from npu_nvme.runtime.training_benchmark import campaign
            return campaign(config,args.output,profile=args.profile)
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as error:
        parser.exit(2,str(error)+'\n')
