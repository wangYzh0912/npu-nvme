#!/usr/bin/env python3
"""Real GPT-2/XL split-optimizer live source and strict fresh restore pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import numpy as np
from c1_training_state_restart import (build_training,batch_for_step,state_digest,control_state,write_state_oracle,compare_state_oracle)
from d1_full_state import identity


def write(path,value):path.write_text(json.dumps(value,indent=2,default=str)+'\n')

def build(args,initialized=False):
    from npu_nvme.framework.cells import LiveForwardBackwardCell,LiveOptimizerCell
    ms,model,optimizer,_=build_training(args,initialized)
    fb=LiveForwardBackwardCell(model,optimizer);update=LiveOptimizerCell(optimizer)
    loss,grads=fb(*batch_for_step(ms,0,args.seq_len));ms.hal.synchronize();update(*grads);ms.hal.synchronize()
    return ms,model,optimizer,fb,update

def run_step(ms,fb,update,guard,step,args):
    loss,grads=fb(*batch_for_step(ms,step,args.seq_len))
    ms.hal.synchronize();value=float(np.asarray(loss.asnumpy()).reshape(()))
    if not np.isfinite(value):raise ValueError('nonfinite loss')
    guard.update(lambda:update(*grads))
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['format','baseline','source','restore'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--model',choices=['gpt2','gpt2_xl'],default='gpt2')
    p.add_argument('--seed',type=int,default=41);p.add_argument('--shm-id',type=int,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    a.npu=7;a.seq_len=129;a.dropout_rate=0.;a.loss_scale=1.;a.deterministic='ON';a.pci='0000:83:00.0';a.save_step=3;a.continue_steps=3
    os.environ['SPDK_SHM_ID']=str(a.shm_id)
    from npu_nvme.storage.bindings import load_backend
    from npu_nvme.storage.full_transport import FullTransport
    if a.phase=='format':
        from npu_nvme.storage.layout import make_layout,META_SLOT_A_OFFSET,META_SLOT_B_OFFSET,SUPERBLOCK_OFFSET
        from npu_nvme.storage.format import pack_superblock,pack_metadata
        transport=FullTransport(load_backend(),pci=a.pci,npu=7,depth=4,chunk_size=16<<20,profiling_dir=a.out)
        try:
            layout=make_layout(total_bytes=transport.total_bytes,full_slot_bytes=32<<30,full_slot_count=3,delta_slot_bytes=256<<20,delta_slot_count=128)
            if layout.full_end>128<<30 or layout.delta_end>256<<30:raise ValueError('E1 extent overlaps Qwen')
            (a.out/'v2-header-before.bin').write_bytes(transport.read(0,1<<20))
            raw=pack_metadata(dict(strict_contract='D1',catalog_revision=0,checkpoints={}),0)
            transport.write(META_SLOT_A_OFFSET,raw);transport.write(META_SLOT_B_OFFSET,raw);transport.flush()
            transport.write(SUPERBLOCK_OFFSET,pack_superblock(layout));transport.flush()
            write(a.out/'format.json',dict(status='pass',full_slot_bytes=32<<30,full_end=layout.full_end))
        finally:transport.close(120)
        return
    from npu_nvme.framework.update_guard import UpdateGuard
    from npu_nvme.framework.full_state import training_spec,MindSporeRestoreTarget
    from npu_nvme.live_checkpoint import LiveCheckpoint
    from training_state import encode_control_value
    store=None
    if a.phase=='restore':
        import mindspore as ms
        from experiments.common import init_env
        ms.set_context(deterministic='ON');init_env(device_id=7,seed=a.seed)
        guard=UpdateGuard(timeout_seconds=600,synchronize=ms.hal.synchronize)
    else:
        ms,model,optimizer,fb,update=build(a)
        guard=UpdateGuard(timeout_seconds=600,synchronize=ms.hal.synchronize)
    try:
        if a.phase!='baseline':
            store=LiveCheckpoint(guard,nvme_addr=a.pci,npu_device_id=7,requested_chunk_size=16<<20,
                pipeline_depth=4,spdk_shm_id=a.shm_id,slot_size_gb=32,profiling_dir=str(a.out/'profiling'))
        if a.phase=='restore':
            saved=json.loads((a.out/'source.json').read_text());constructed=[]
            def factory(spec):
                framework,model,optimizer,fb,update=build(a,True)
                constructed.append((model,optimizer,fb,update))
                return MindSporeRestoreTarget(framework=framework,acl=load_backend().acl_lib,npu=7,
                    model=model,optimizer=optimizer,cell=fb,identity=identity(a,framework))
            target,receipt=store.restore_full_state(factory,saved['spec'],a.save_step,deadline=time.monotonic()+1200)
            model,optimizer,fb,update=constructed[0]
            assert state_digest(model,optimizer)==saved['checkpoint_state']
            assert {k:encode_control_value(v)[1]['sha256'] for k,v in target.controls.items()}==saved['controls_sha256']
            losses=[run_step(ms,fb,update,guard,step,a) for step in range(a.save_step+1,a.save_step+a.continue_steps+1)]
            comparison=compare_state_oracle(a.out/'source-oracle',model,optimizer,1e-5,1e-6)
            assert comparison['allclose'] and np.allclose(losses,saved['losses'][a.save_step:],rtol=1e-5,atol=1e-6)
            write(a.out/'restore.json',dict(status='pass',losses=losses,comparison=comparison,ready=target.ready,loaded_byte_exact=True,controls_byte_exact=True))
        else:
            losses=[];result={}
            for step in range(1,a.save_step+a.continue_steps+1):
                losses.append(run_step(ms,fb,update,guard,step,a))
                if a.phase=='source' and step==a.save_step:
                    controls=control_state(ms,optimizer,step,a)
                    spec=training_spec({'model':model,'optimizer':optimizer},controls,identity(a,ms),framework=ms,acl=load_backend().acl_lib,npu=7)
                    result.update(checkpoint_state=state_digest(model,optimizer),spec=spec,
                        controls_sha256={k:encode_control_value(v)[1]['sha256'] for k,v in controls.items()})
                    handle=store.save_state({'model':model,'optimizer':optimizer},controls,step,expected_spec=spec,timeout=600)
            guard.before_optimizer_update();write_state_oracle(a.out/(a.phase+'-oracle'),model,optimizer)
            result.update(status='pass',phase=a.phase,losses=losses,events=guard.events,final_state=state_digest(model,optimizer),
                          capture='live_borrowed' if a.phase=='source' else 'none',fence='checkpoint terminal before optimizer dispatch')
            if a.phase=='source':
                baseline=json.loads((a.out/'baseline.json').read_text());assert np.allclose(losses,baseline['losses'],rtol=1e-5,atol=1e-6)
                assert compare_state_oracle(a.out/'baseline-oracle',model,optimizer,1e-5,1e-6)['allclose']
                result['receipt']=handle.as_dict()
            write(a.out/(a.phase+'.json'),result)
    finally:
        if store:
            try:store.close(600)
            except BaseException as error:
                if store.store.transport.ctx:
                    write(a.out/'retained.json',dict(pid=os.getpid(),error=repr(error)))
                    __import__('threading').Event().wait()
                raise
if __name__=='__main__':main()
