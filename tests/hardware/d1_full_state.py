#!/usr/bin/env python3
"""D1 GPT-2 strict save/fresh restore; a parent runs the fixed three-seed matrix."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
from experiments.training.full_fixture import build_training, train_range, state_digest, control_state, write_state_oracle, compare_state_oracle
from experiment_evidence import sha256_file


def write(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n')


def identity(args,ms):
    return dict(workload=args.model, seq_len=args.seq_len, dropout=args.dropout_rate,
                loss_scale=args.loss_scale, optimizer='AdamWeightDecay', framework=ms.__version__,
                world_size=1, deterministic=args.deterministic)


def open_store(args):
    from npu_nvme import StrictCheckpoint
    return StrictCheckpoint(nvme_addr=args.pci, npu_device_id=args.npu, requested_chunk_size=1 << 20, slot_size_gb=10, pipeline_depth=4, spdk_shm_id=args.shm_id, profiling_dir=str(args.out / 'profiling'))


def save(args):
    from npu_nvme.framework.training_state import encode_control_value
    from npu_nvme.framework.full_state import training_spec
    from npu_nvme.storage.bindings import load_backend
    ms,model,optimizer,cell=build_training(args)
    train_range(ms,cell,1,args.save_step,args.seq_len)
    controls=control_state(ms,optimizer,args.save_step,args)
    spec=training_spec({'model':model,'optimizer':optimizer},controls,identity(args,ms),
                       framework=ms,acl=load_backend().acl_lib,npu=args.npu)
    initial=state_digest(model,optimizer)
    store=open_store(args)
    try:
        start=time.monotonic()
        handle=store.save_state({'model':model,'optimizer':optimizer},controls,args.save_step,
                                expected_spec=spec)
        handle.wait(120)
        save_seconds=time.monotonic()-start
        losses,_=train_range(ms,cell,args.save_step+1,args.save_step+args.continue_steps,args.seq_len)
        write_state_oracle(args.out/'oracle',model,optimizer)
        write(args.out/'save.json',dict(status='pass',spec=spec,state=initial,losses=losses,
            final_state=state_digest(model,optimizer),generation=handle.generation,request_id=handle.request_id,
            controls_sha256={k:encode_control_value(v)[1]['sha256'] for k,v in controls.items()},
            receipt=handle.as_dict(),save_seconds=save_seconds,transport='bounded-host-existing-C-ABI',capture='frozen'))
    finally: store.close()


def restore(args):
    import numpy as np
    from npu_nvme.framework.training_state import encode_control_value
    from npu_nvme.framework.full_state import MindSporeRestoreTarget
    from npu_nvme.storage.bindings import load_backend
    saved=json.loads((args.out/'save.json').read_text())
    # Load graph/runtime dependencies before SPDK attaches. This throwaway
    # materialization is not the restoration target and receives no payload.
    bootstrap=build_training(args); ms=bootstrap[0]; del bootstrap; gc.collect()
    store=open_store(args)
    made=[]
    def factory(spec):
        framework,model,optimizer,cell=build_training(args,initialized=True)
        target=MindSporeRestoreTarget(framework=framework,acl=load_backend().acl_lib,npu=args.npu,
            model=model,optimizer=optimizer,cell=cell,identity=identity(args,framework))
        try: target.train_step()
        except RuntimeError: pass
        else: raise AssertionError('unready target allowed training')
        made.append(target)
        return target
    try:
        start=time.monotonic()
        target,receipt=store.restore_full_state(factory,saved['spec'],args.save_step,deadline=time.monotonic()+180)
        restore_seconds=time.monotonic()-start
        actual=state_digest(target.model,target.optimizer)
        if actual!=saved['state']: raise AssertionError(f'initial restore bytes differ: {actual} != {saved["state"]}')
        control_hashes={k:encode_control_value(v)[1]['sha256'] for k,v in target.controls.items()}
        if control_hashes!=saved['controls_sha256']: raise AssertionError('restored controls differ from source')
        if target.controls['data_cursor'] != {'epoch':0,'sample':args.save_step}: raise AssertionError('cursor differs')
        losses,_=train_range(ms,target.cell,args.save_step+1,args.save_step+args.continue_steps,args.seq_len)
        comparison=compare_state_oracle(args.out/'oracle',target.model,target.optimizer,1e-5,1e-6)
        result=dict(status='pass',gate='H01-D1',seed=args.seed,loaded_state_byte_exact=True,
                    controls_byte_exact=True,ready=target.ready,continuation_steps=args.continue_steps,
                    losses=losses,expected_losses=saved['losses'],final_state_comparison=comparison,
                    receipt=receipt.__dict__,restore_seconds=restore_seconds)
        if not comparison['allclose'] or not np.allclose(losses,saved['losses'],rtol=1e-5,atol=1e-6):
            result['status']='fail'
        write(args.out/'result.json',result)
        if result['status']!='pass': raise AssertionError('strict continuation exceeds frozen tolerance')
    finally: store.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['matrix','save','restore'],default='matrix')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[41,42,43])
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--pci',default='0000:83:00.0'); p.add_argument('--npu',type=int,default=7)
    p.add_argument('--shm-id',type=int,default=39100)
    args=p.parse_args()
    args.model='gpt2';args.seq_len=129;args.save_step=2;args.continue_steps=3
    args.loss_scale=1.;args.dropout_rate=0.;args.deterministic='ON'
    args.out=args.out.resolve()
    if os.geteuid()!=0 or args.pci!='0000:83:00.0': p.error('requires root and authorized 83 namespace')
    for pci,driver in [('0000:83:00.0','uio_pci_generic'),('0000:84:00.0','nvme')]:
        if Path('/sys/bus/pci/devices',pci,'driver').resolve().name!=driver: p.error('device preflight failed')
    if args.phase!='matrix':
        write(args.out/(args.phase+'-source.json'),dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            library_sha256=sha256_file(ROOT/'build_out/lib/libnpu_nvme.so'),pid=os.getpid()))
        try: (save if args.phase=='save' else restore)(args)
        except BaseException:
            write(args.out/(args.phase+'-error.json'),dict(status='fail',traceback=traceback.format_exc()))
            raise
        return
    args.out.mkdir(parents=True,exist_ok=False)
    sources={str(f.relative_to(ROOT)):sha256_file(f) for directory in ('python','src','include','tests','experiments')
             for f in sorted((ROOT/directory).rglob('*')) if f.is_file() and f.suffix in ('.py','.c','.h')}
    write(args.out/'sources.json',sources)
    phases=[]
    for seed in args.seeds:
        directory=args.out/f'seed-{seed}';directory.mkdir()
        for phase in ('save','restore'):
            argv=[sys.executable,str(Path(__file__).resolve()),'--phase',phase,'--out',str(directory),
                  '--seed',str(seed),'--pci',args.pci,'--npu',str(args.npu),'--shm-id',str(args.shm_id+len(phases))]
            start=time.monotonic()
            print(f'D1 seed={seed} {phase}',flush=True)
            with (directory/(phase+'.log')).open('w') as log:
                process=subprocess.run(argv,cwd=directory,stdout=log,stderr=subprocess.STDOUT,timeout=360)
            phases.append(dict(seed=seed,phase=phase,argv=argv,returncode=process.returncode,seconds=time.monotonic()-start))
            write(args.out/'phases.json',phases)
            if process.returncode:
                write(args.out/'result.json',dict(status='fail',phases=phases));return 1
    changed=[name for name,digest in sources.items() if sha256_file(ROOT/name)!=digest]
    write(args.out/'result.json',dict(status='pass' if not changed else 'invalid',seeds=args.seeds,
                                    phases=phases,changed_sources=changed))
    return int(bool(changed))


if __name__=='__main__': sys.exit(main())
