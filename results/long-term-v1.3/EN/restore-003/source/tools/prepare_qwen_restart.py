#!/usr/bin/env python3
"""Validate exited TP4 source and hash the selected generation before restoration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from audit_safetensors import inspect


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def prepare(run,step,horizon):
    run=Path(run).resolve();ranks=[]
    for rank in range(4):
        directory=run/f'rank_{rank}';report=json.loads((directory/'acceptance.json').read_text())
        if report['status']!='training_pass_restart_not_tested' or Path('/proc',str(report['pid'])).exists():
            raise ValueError('source must finish and exit before restart')
        if report['parallel']!={'tensor_parallel':4,'data_parallel':1,'pipeline_parallel':1}:
            raise ValueError('source topology differs')
        if report['schedule']['checkpoint_step']!=step or report['schedule']['lr_horizon']!=horizon:
            raise ValueError('source step/horizon differs')
        control=json.loads((directory/'control-checkpoint.json').read_text())
        if control['logical_optimizer_step']!=step or control['next_data_row']!=step:
            raise ValueError('source cursor differs')
        checkpoint=run/f'training/checkpoint/rank_{rank}/qwen3_rank_{rank}-{step}_1.safetensors'
        audit=inspect(checkpoint)
        scalars={t['name']:t.get('scalar_value') for t in audit['tensors']}
        if scalars.get('global_step')!=step or scalars.get('step_num')!=step:
            raise ValueError('checkpoint step differs')
        artifacts={str(path.relative_to(run)):dict(bytes=path.stat().st_size,sha256=digest(path))
                   for path in [directory/'control-checkpoint.json',directory/'state-checkpoint.json',directory/'input_ids.npy',directory/'resolved_config.json']}
        artifacts[str(checkpoint.relative_to(run))]=dict(bytes=audit['bytes'],sha256=audit['sha256'])
        (directory/'checkpoint-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        ranks.append(dict(rank=rank,source_pid=report['pid'],data_sha256=report['data_sha256'],config_sha256=report['config_sha256'],artifacts=artifacts))
        print('Prepared rank',rank,flush=True)
    if len({r['data_sha256'] for r in ranks})!=1:raise ValueError('rank input tokens differ')
    return dict(schema_version=1,source_run=str(run),checkpoint_step=step,lr_horizon=horizon,
                topology=dict(tp=4,dp=1,pp=1),ranks=ranks,execution_status='completed',
                validation_status='not_applicable',scope='preflight only; does not prove restoration')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-run',required=True,type=Path);p.add_argument('--checkpoint-step',type=int,default=8);p.add_argument('--lr-horizon',type=int,default=32)
    a=p.parse_args();result=prepare(a.source_run,a.checkpoint_step,a.lr_horizon)
    output=a.source_run/'restart_contract.json'
    with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
