#!/usr/bin/env python3
"""Real single-rank GPT-2 R0 source and fresh-process continuation pilot."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import numpy as np
from c1_training_state_restart import (build_training,train_range,state_digest,control_state,apply_control_state,write_state_oracle,compare_state_oracle)


def write(path,value):path.write_text(json.dumps(value,indent=2,default=str)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['source','restore'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=41);p.add_argument('--shm-id',type=int,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    a.npu=7;a.model='gpt2';a.seq_len=129;a.dropout_rate=0.;a.loss_scale=1.;a.deterministic='ON';a.save_step=3;a.continue_steps=3
    os.environ['SPDK_SHM_ID']=str(a.shm_id)
    ms,model,optimizer,cell=build_training(a)
    from npu_nvme.storage.bindings import load_backend
    from npu_nvme.storage.full_transport import FullTransport
    from npu_nvme.d2.backend import RegisteredBackend
    from npu_nvme.d2.format import Region
    from npu_nvme.framework.incremental import IncrementalState
    from training_state import encode_control_value
    cfg=json.loads((ROOT/'config/f1_region.json').read_text())
    for key in ('protected_qwen_region','authorization_source','required_feature'):cfg.pop(key)
    transport=FullTransport(load_backend(),pci=cfg['pci_addr'],npu=7,depth=4,chunk_size=4<<20,profiling_dir=a.out)
    try:
        backend=RegisteredBackend(transport,cfg,region_id=cfg['region_id']);region=Region(backend,offset=cfg['offset'],length=cfg['length'],retention=3);region.mount()
        if region.header['region_id']!=cfg['region_id']:raise ValueError('F1 identity differs')
        adapter=IncrementalState(ms,{'model':model,'optimizer':optimizer},region,host_budget_bytes=32<<30)
        if a.phase=='source':
            receipts=[]
            for step in range(1,4):
                train_range(ms,cell,step,step,a.seq_len)
                receipts.append(adapter.save(control_state(ms,optimizer,step,a),step=step))
            checkpoint=state_digest(model,optimizer)
            controls=control_state(ms,optimizer,3,a);hashes={k:encode_control_value(v)[1]['sha256'] for k,v in controls.items()}
            losses,_=train_range(ms,cell,4,6,a.seq_len);write_state_oracle(a.out/'oracle',model,optimizer)
            write(a.out/'source.json',dict(status='pass',receipts=receipts,state=checkpoint,controls_sha256=hashes,losses=losses,state_bytes=adapter.state_bytes,
                capture='blocking Host oracle',max_delta_chain=2,manifest_sha256=adapter.manifest.digest))
        else:
            saved=json.loads((a.out/'source.json').read_text())
            if adapter.manifest.digest!=saved['manifest_sha256']:raise ValueError('R0 manifest differs')
            def controls(value,step):apply_control_state(ms,optimizer,value,step,a)
            def verify(value,step):
                assert step==3 and state_digest(model,optimizer)==saved['state']
                observed=control_state(ms,optimizer,step,a)
                assert {k:encode_control_value(v)[1]['sha256'] for k,v in observed.items()}==saved['controls_sha256']
            result=adapter.restore(controls,verify,generation=saved['receipts'][-1]['generation'])
            assert adapter.ready
            losses,_=train_range(ms,cell,4,6,a.seq_len)
            comparison=compare_state_oracle(a.out/'oracle',model,optimizer,1e-5,1e-6)
            assert comparison['allclose'] and np.allclose(losses,saved['losses'],rtol=1e-5,atol=1e-6)
            write(a.out/'restore.json',dict(status='pass',ready=True,loaded_byte_exact=True,controls_byte_exact=True,comparison=comparison,losses=losses,generation=result['generation']))
    finally:
        try:transport.close(120)
        except BaseException as error:
            write(a.out/'retained.json',dict(pid=os.getpid(),error=repr(error)));__import__('threading').Event().wait()
if __name__=='__main__':main()
