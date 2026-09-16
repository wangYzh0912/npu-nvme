"""C1 supervisor. Only child phases import the training framework."""
import argparse
import fcntl
from contextlib import contextmanager
import json
import os
from pathlib import Path
import resource
import re
import signal
import subprocess
import sys
import time
import uuid

from .contracts import ROOT, METHODS, CapabilityBlocked, canonical, check_identity, comparison, digest, load_config, outcome, source_identity, write, validate_restore_report


_device_fd = None

@contextmanager
def device_lock():
    global _device_fd
    path=Path('/models/npu_nvme_exp/user7-stack/c1-device-83.lock')
    with path.open('a+') as stream:
        try: fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error: raise CapabilityBlocked('another C1 process owns device admission') from error
        _device_fd=stream.fileno()
        try: yield
        finally: _device_fd=None


class PhaseFailure(RuntimeError):
    def __init__(self,code,message):
        self.code=code if code in (1,2,3,4,5) else 1
        super().__init__(message)


def resource_sample(pid):
    proc=subprocess.run(['ps','-eo','pid=,ppid=,rss='],capture_output=True,text=True,timeout=10,check=True)
    rows=[tuple(map(int,line.split())) for line in proc.stdout.splitlines()]
    descendants={pid}
    while True:
        more={p for p,parent,rss in rows if parent in descendants}
        if more<=descendants: break
        descendants|=more
    host=sum(rss*1024 for p,parent,rss in rows if p in descendants)
    npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=10,check=True)
    lines=npu.stdout.splitlines();hbm=None
    for i,line in enumerate(lines):
        if re.search(r'\|\s*7\s+910B3',line):
            match=re.search(r'(\d+)\s*/\s*65536',lines[i+1])
            if match: hbm=int(match[1])*1024**2
    if hbm is None: raise RuntimeError('cannot measure NPU7 HBM; resource budget unknown')
    return dict(monotonic_ns=time.monotonic_ns(),host_bytes=host,hbm_bytes=hbm)


def flat_config(c, out, seed):
    w,k,r,s,m=(c[x] for x in ('workload','checkpoint','runtime','storage','measurement'))
    return dict(project_root=str(ROOT), project_commit=c['identity']['expected_commit'], model=w['model_id'],
        model_revision=w['model_revision'], seed=seed,input_tokens=w['seq_len'],batch_size=w['batch'],dropout=w['dropout'],
        transport=k['transport'],warmup_steps=m['warmup'],formal_steps=m['formal_steps'],checkpoint_every=k['interval'],capture_mode=k['capture'],
        continue_steps=m['continue_steps'],loss_rtol=m['loss_rtol'],loss_atol=m['loss_atol'],npu_device=7,
        numa_node=4,training_numa_node=2,io_numa_node=4,training_cpu_set='48-71',io_cpu_set='96-119',
        fs_test_dir=str(Path(s['fs_root'])/(out.parent.name+'-'+canonical(str(out))[:12])/out.name),raw_pci=s['pci'],raw_test_authorized=True,
        worker_python_bytecheckpoint='/home/user7/npu-nvme-baseline-envs/bytecheckpoint/bin/python',
        worker_python_fastpersist='/home/user7/npu-nvme-baseline-envs/fastpersist/bin/python',
        upstream_root='/home/user7/npu-nvme-baseline-upstreams',results_root=str(out),same_physical_storage_verified=False,
        timeout_seconds=r['wait_ms']/1000,admission_timeout_seconds=r['admission_ms']/1000,drain_timeout_seconds=r['drain_ms']/1000,close_timeout_seconds=r['close_ms']/1000,
        phase_timeout_seconds=r['phase_ms']/1000,chunk_bytes=r['chunk_bytes'],pipeline_depth=r['dma_depth'],
        max_inflight=1,ours_max_inflight=1,keep_last_n=2,ours_keep_last_n=2,slot_size_gb=s['slot_size_gb'],
        spdk_shm_id=s['shm_id'],ours_chunk_bytes=r['chunk_bytes'], host_budget_bytes=r['host_budget_bytes'],
        hbm_budget_bytes=r['hbm_budget_bytes'])


def preflight(c, dry=False):
    if not dry and c['checkpoint']['transport']!='async':
        raise CapabilityBlocked('this implementation uses async; run legacy_sync on the frozen B2 entry checkout')
    if (Path('/models/npu_nvme_exp/user7-stack/c1-device-83.quarantine.json')).exists() and not dry:
        raise RuntimeError('previous C1 device access outcome unknown; reconcile quarantine first')
    identity=source_identity()
    check_identity(c,identity)
    s=c['storage']
    auth=json.loads(Path(s['authorization_file']).read_text())
    if 'ours' in c['checkpoint']['methods'] and (auth.get('write_authorized') is not True or auth.get('pci_addr') != s['pci'] or auth.get('protected_pci_addr') != '0000:84:00.0'):
        raise ValueError('raw authorization does not match selected device')
    capabilities={name:dict(state_scopes=[] if name=='none' else ['full_state'],capture=['frozen'],
        transport=c['checkpoint']['transport'] if name=='ours' else 'framework',requires_device=True,upstream_core_invoked=name in ('mindspore_native_save','bytecheckpoint_host'),
        identity='host-adapted-semantic-port' if name=='bytecheckpoint_host' else name) for name in c['checkpoint']['methods']}
    result=dict(execution_status='planned' if dry else 'completed',validation_status='not_applicable',
                config_digest=canonical(c),identity=identity,capabilities=capabilities,device_initialized=False)
    if not dry:
        # Child imports cannot initialize NPU state in this supervisor.
        commands=[['npu-smi','info'], ['lspci','-s','83:00.0','-nnk'],['lspci','-s','84:00.0','-nnk'],
                  ['findmnt','-no','TARGET,SOURCE,FSTYPE,OPTIONS','-T',s['fs_root']]]
        result['probes']=[]
        for argv in commands:
            proc=subprocess.run(argv,capture_output=True,text=True,timeout=30)
            result['probes'].append(dict(argv=argv,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
            if proc.returncode: raise RuntimeError(f'preflight probe failed: {argv}')
        if not re.search(r'No running processes found in NPU\s+7',result['probes'][0]['stdout']):
            raise RuntimeError('NPU7 already has a running process')
        if 'uio_pci_generic' not in result['probes'][1]['stdout'] or 'Kernel driver in use: nvme' not in result['probes'][2]['stdout']:
            raise RuntimeError('unexpected raw/protected disk drivers')
        for name in c['checkpoint']['methods']:
            if name=='bytecheckpoint_host':
                env=dict(os.environ)
                env['PYTHONPATH']='/home/user7/npu-nvme-baseline-upstreams/ByteCheckpoint:/home/user7/.local/lib/python3.9/site-packages'
                proc=subprocess.run(['/home/user7/npu-nvme-baseline-envs/bytecheckpoint/bin/python','-c',
                    'import torch, bytecheckpoint.workflow.state_dict; print(torch.__version__)'],env=env,capture_output=True,text=True,timeout=60)
                if proc.returncode: raise RuntimeError('ByteCheckpoint dependency blocked: '+proc.stderr[-2000:])
                capabilities[name]['import_probe']=proc.stdout
        lib=Path(os.environ.get('NPU_NVME_LIBRARY_PATH', ROOT/'build_out/lib/libnpu_nvme.so'))
        if 'ours' in capabilities: result['library_sha256']=digest(lib)
    return result


def run_phase(phase, config_path, run_dir, method, out, *, generation='latest-committed', mode='verify', output=None):
    argv=[sys.executable,str(ROOT/'train.py'),'_phase','--phase',phase,'--flat-config',str(config_path),
          '--run-dir',str(run_dir),'--method',method,'--generation',str(generation),'--mode',mode]
    if output: argv += ['--output',str(output)]
    flat=json.loads(Path(config_path).read_text())
    log=out/f'{phase}-{uuid.uuid4().hex[:8]}'
    begun=time.monotonic()
    with log.with_suffix('.stdout.log').open('w') as stdout,log.with_suffix('.stderr.log').open('w') as stderr:
        proc=subprocess.Popen(argv,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)+':'+str(ROOT/'python')+':'+os.environ.get('PYTHONPATH','')),stdout=stdout,stderr=stderr,start_new_session=True,pass_fds=(_device_fd,) if _device_fd is not None else ())
        samples=[]
        try:
            while proc.poll() is None:
                if time.monotonic()-begun > flat['phase_timeout_seconds']: raise subprocess.TimeoutExpired(argv,flat['phase_timeout_seconds'])
                sample=resource_sample(proc.pid)
                samples.append(sample)
                if sample['host_bytes'] > flat['host_budget_bytes'] or sample['hbm_bytes'] > flat['hbm_budget_bytes']:
                    raise subprocess.TimeoutExpired(argv,flat['phase_timeout_seconds'])
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired: pass
            rc=proc.returncode
        except subprocess.TimeoutExpired:
            # No forced DMA-buffer reclamation. Keep process ownership explicit,
            # stop this campaign and require external stop proof before reuse.
            record=dict(phase=phase,argv=argv,pid=proc.pid,returncode=4,outcome='unknown',
                        quarantine=True,elapsed_seconds=time.monotonic()-begun)
            write(log.with_suffix('.process.json'),record)
            write(Path('/models/npu_nvme_exp/user7-stack/c1-device-83.quarantine.json'),record)
            raise TimeoutError(f'phase deadline; retained process {proc.pid}; no further device admission')
    record=dict(phase=phase,argv=argv,pid=proc.pid,returncode=rc,elapsed_seconds=time.monotonic()-begun,
                stdout=str(log.with_suffix('.stdout.log')),stderr=str(log.with_suffix('.stderr.log')),resources=samples)
    write(log.with_suffix('.process.json'),record)
    if rc==4:
        write(Path('/models/npu_nvme_exp/user7-stack/c1-device-83.quarantine.json'),dict(record,quarantine=True))
    if rc: raise PhaseFailure(rc, f'{phase}/{method} failed ({rc}); see {log}')
    return record


def seal(out,result,identity):
    current=source_identity()
    if current['source_digest'] != identity['source_digest']:
        result.update(execution_status='aborted',validation_status='invalid',error='source changed during run')
    result['source_digest']=identity['source_digest']
    result['observed_commit']=identity['observed_commit']
    write(out/'result.json',result)
    write(out/'evidence_manifest.json',dict(schema_version=1,artifacts=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=digest(p))
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='evidence_manifest.json']))
    return result


def execute(c,out,command,source_run=None,generation='latest-committed'):
    try: pre=preflight(c)
    except Exception as error: raise CapabilityBlocked(str(error)) from error
    write(out/'preflight.json',pre)
    identity=pre['identity'];write(out/'source_manifest.json',identity)
    result=dict(execution_status='completed',validation_status='fail',command=command,runs=[])
    try:
        if command=='verify-restart':
            source=Path(source_run).resolve()
            flat=source.parent/'flat-config.json'
            if not flat.is_file(): raise ValueError('source run requires sibling flat-config.json')
            saved=json.loads(flat.read_text()); method=json.loads((source/'source.json').read_text())['adapter']
            if saved['project_commit']!=identity['observed_commit'] or method not in c['checkpoint']['methods']: raise ValueError('source identity/method mismatch')
            if saved['model_revision']!=c['workload']['model_revision'] or saved['input_tokens']!=c['workload']['seq_len'] or saved['capture_mode']!=c['checkpoint']['capture']:
                raise ValueError('source workload differs from restore configuration')
            output=out/'restore.json'
            process=run_phase('restore',flat,source,method,out,generation=generation,output=output)
            row=json.loads(output.read_text())
            validate_restore_report(row,json.loads((source/'source.json').read_text()),c['measurement']['continue_steps'])
            result['runs']=[dict(method=method,restore=row,process=process)]
        elif command=='inspect':
            # Read media through the existing strict mount; no catalog writes.
            flat=flat_config(c,out,c['workload']['seeds'][0]);write(out/'flat-config.json',flat)
            run_phase('inspect',out/'flat-config.json',out,'ours',out,generation=generation)
            result['inspection']=json.loads((out/'inspection.json').read_text())
        else:
            for index,seed in enumerate(c['workload']['seeds']):
                seed_dir=out/f'seed-{seed}';seed_dir.mkdir()
                flat=flat_config(c,seed_dir,seed);write(seed_dir/'flat-config.json',flat)
                run_phase('prepare',seed_dir/'flat-config.json',seed_dir/'prepare','none',seed_dir)
                fixture=json.loads((seed_dir/'prepare/fixture.json').read_text())
                logical=fixture['bytes']
                capacity=dict(logical_state_bytes=logical,slot_bytes=c['storage']['slot_size_gb']*1024**3,
                    host_budget_bytes=c['runtime']['host_budget_bytes'],hbm_budget_bytes=c['runtime']['hbm_budget_bytes'],
                    conservative_host_bytes=8*logical+8*1024**3,conservative_hbm_bytes=4*logical+8*1024**3,
                    retained_generations=2,pending_generations=1,estimates_not_allocator_proofs=True)
                write(seed_dir/'capacity_plan.json',capacity)
                if capacity['conservative_host_bytes']>capacity['host_budget_bytes'] or capacity['conservative_hbm_bytes']>capacity['hbm_budget_bytes'] or logical*1.25>capacity['slot_bytes']:
                    raise RuntimeError('fixture exceeds conservative C1 resource preflight')
                methods=list(c['checkpoint']['methods']);methods=methods[index%len(methods):]+methods[:index%len(methods)]
                for method in methods:
                    path=seed_dir/method;path.mkdir()
                    row=dict(seed=seed,method=method,comparison=comparison(method),processes=[])
                    result['runs'].append(row)
                    print(f'C1 seed={seed} method={method} source',flush=True)
                    row['processes'].append(run_phase('source',seed_dir/'flat-config.json',path,method,seed_dir))
                    source=json.loads((path/'source.json').read_text())
                    row['source']=str(path/'source.json')
                    if source.get('status')!='trend_measured': raise RuntimeError('source did not complete')
                    if command=='fit' or method=='none':
                        row['restore_status']='not_applicable'
                        continue
                    output=path/'restore.json'
                    row['processes'].append(run_phase('restore',seed_dir/'flat-config.json',path,method,seed_dir,output=output))
                    restored=json.loads(output.read_text());row['restore']=str(output)
                    validate_restore_report(restored,source,c['measurement']['continue_steps'])
                    samples=[]
                    for repetition in range(c['measurement']['restore_repetitions']+1):
                        target=path/f'timing-{repetition}.json'
                        row['processes'].append(run_phase('restore',seed_dir/'flat-config.json',path,method,seed_dir,mode='timing',output=target))
                        timing=json.loads(target.read_text())
                        if timing.get('status')!='pass' or timing.get('mandatory_integrity') is not True: raise RuntimeError('timing omitted mandatory integrity')
                        t=timing['timing'];samples.append((t['state_ready_ns']-t['restore_begin_ns'])/1e9)
                    row['restore_seconds']=dict(warmup=samples[0],raw_values=samples[1:],sample_count=len(samples)-1,
                        mean=sum(samples[1:])/(len(samples)-1),min=min(samples[1:]),max=max(samples[1:]),statistics='observed mean/range; no tail inference')
                    write(out/'progress.json',result)
        result['validation_status']='not_applicable' if command in ('fit','inspect') else 'pass'
    except Exception as error:
        code=getattr(error,'code',4 if isinstance(error,TimeoutError) else 1)
        result.update(error=repr(error),execution_status='blocked' if code==3 else 'aborted' if code==4 else 'completed',
                      validation_status='invalid' if code in (3,4) else 'fail',exit_code=code)
    return seal(out,result,identity)


def child(args):
    c=json.loads(Path(args.flat_config).read_text())
    if c.get('raw_pci')!='0000:83:00.0' or c.get('npu_device')!=7 or c.get('project_root')!=str(ROOT):
        raise ValueError('child configuration differs from C1 device/source boundary')
    from experiments.baselines.repro.runner import prepare,run,restore
    begun=time.monotonic()
    if args.phase=='prepare': result=prepare(c,args.run_dir)
    elif args.phase=='source': result=run(c,args.method,args.run_dir)
    elif args.phase=='restore': result=restore(c,args.method,args.run_dir,generation=args.generation,mode=args.mode,output_path=args.output)
    else:
        from experiments.baselines.repro.adapters.ours import OursAdapter
        adapter=OursAdapter(c,args.run_dir)
        try: result=dict(catalog=adapter.ckpt.meta_dict)
        finally: adapter.close()
        write(Path(args.run_dir)/'inspection.json',result)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    mappings=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if '/' in line and any(name in line for name in ('libnpu_nvme','libascendcl','libruntime.so','libhccl'))})
    metrics=dict(pid=os.getpid(),elapsed_seconds=time.monotonic()-begun,max_rss_bytes=usage.ru_maxrss*1024,
                 loaded_libraries={name:digest(name) for name in mappings if Path(name).is_file()},python=sys.executable,
                 cpu_affinity=sorted(os.sched_getaffinity(0)))
    write(Path(args.output).with_suffix('.resources.json') if args.output else Path(args.run_dir)/f'{args.phase}.resources.json',metrics)
    if metrics['max_rss_bytes']>c['host_budget_bytes']: raise RuntimeError('Host budget exceeded')
    return 0 if result.get('status') in (None,'pass','trend_measured') else result.get('error_code',1)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('preflight','fit','verify-restart','benchmark','inspect'):
        p=sub.add_parser(name);p.add_argument('--config',required=True);p.add_argument('--out');p.add_argument('--dry-run',action='store_true')
        if name=='verify-restart': p.add_argument('--source-run',required=True);p.add_argument('--generation',default='latest-committed')
    p=sub.add_parser('_phase');p.add_argument('--phase',choices=('prepare','source','restore','inspect'),required=True)
    p.add_argument('--flat-config',required=True);p.add_argument('--run-dir',required=True);p.add_argument('--method',required=True,choices=METHODS)
    p.add_argument('--generation',default='latest-committed');p.add_argument('--mode',default='verify',choices=('verify','timing'));p.add_argument('--output')
    args=parser.parse_args(argv)
    out=None
    try:
        if args.command=='_phase': return child(args)
        c=load_config(args.config)
        if args.command=='fit' and len(c['checkpoint']['methods'])!=1: raise ValueError('fit requires exactly one method; use benchmark for a matrix')
        report=preflight(c,dry=True)
        out=Path(args.out or Path(c['output']['root'])/(args.command+'-'+uuid.uuid4().hex)).resolve()
        out.mkdir(parents=True,exist_ok=False)
        write(out/'resolved_config.json',c)
        if args.dry_run or args.command=='preflight':
            report=preflight(c,dry=args.dry_run);write(out/'result.json',report)
            print(json.dumps(dict(out=str(out),execution_status=report['execution_status'],validation_status=report['validation_status'])))
            return 0
        with device_lock():
            result=execute(c,out,args.command,getattr(args,'source_run',None),getattr(args,'generation','latest-committed'))
        print(json.dumps(dict(out=str(out),execution_status=result['execution_status'],validation_status=result['validation_status'],error=result.get('error'))))
        return 0 if result['validation_status'] in ('pass','not_applicable') else result.get('exit_code',4 if result['execution_status']=='aborted' else 1)
    except CapabilityBlocked as error:
        if out is not None: write(out/'result.json',dict(execution_status='blocked',validation_status='invalid',error=str(error)))
        print(str(error),file=sys.stderr);return 3
    except (ValueError,TypeError,KeyError,FileExistsError) as error:
        print(str(error),file=sys.stderr);return 2
    except TimeoutError as error:
        print(str(error),file=sys.stderr);return 4
    except (OSError,RuntimeError) as error:
        print(str(error),file=sys.stderr);return getattr(error,'exit_code',3 if args.command=='preflight' else 1)

if __name__=='__main__': raise SystemExit(main())
