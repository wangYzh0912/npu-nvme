#!/usr/bin/env python3
"""Audit actual D2 Qwen reports; no Native checkpoint or synthetic acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def audit(out,source=None,backend='d2'):
    out=Path(out);owner=json.loads((out/'owner/result.json').read_text())
    if owner['status']!='pass' or owner.get('closed') is not True:raise ValueError('D2 owner failed or retained')
    rows=[];losses=None
    for rank in range(4):
        root=out/f'rank_{rank}';report=json.loads((root/'acceptance.json').read_text())
        expected='restored_and_continued' if source else 'training_pass_restart_not_tested'
        if report['status']!=expected or report.get('checkpoint_backend')!=backend:raise ValueError('D2 rank did not finish')
        receipt=json.loads((root/('d2-restore.json' if source else 'd2-save.json')).read_text())
        if receipt['receipt']['generation']!=owner['receipt']['generation']:raise ValueError('rank/owner generations differ')
        if receipt['state_bytes']<=0 or receipt['capture']!=('blocking' if backend=='d2' else 'blocking_host_snapshot'):raise ValueError('missing explicit state capture')
        if not report['losses'] or any(x['overflow'] or not math.isfinite(x['loss']) for x in report['losses']):raise ValueError('invalid training losses')
        if losses is not None and report['losses']!=losses:raise ValueError('TP4 losses differ')
        losses=report['losses']
        if source:
            original=Path(source)/f'rank_{rank}'
            old=json.loads((original/'acceptance.json').read_text())
            step=report['schedule']['checkpoint_step']
            continued=[x for x in old['losses'] if x['step']>step]
            if len(continued)!=len(losses) or old['data_sha256']!=report['data_sha256']:raise ValueError('continuation data/count differs')
            for a,b in zip(losses,continued):
                if a['step']!=b['step'] or not math.isclose(a['loss'],b['loss'],rel_tol=1e-5,abs_tol=1e-6):raise ValueError('continuation loss differs')
            for target,reference in [('restored-state.json','state-checkpoint.json'),('restored-control.json','control-checkpoint.json'),('state-final.json','state-final.json'),('control-final.json','control-final.json')]:
                if json.loads((root/target).read_text())!=json.loads((original/reference).read_text()):raise ValueError('state/control bytes differ: '+target)
        artifacts={f'rank_{rank}/{name}':hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('state-checkpoint.json','control-checkpoint.json','resolved_config.json','input_ids.npy','d2-save.json','d2-partitions.json')} if not source else {}
        if not source and backend=='bytecheckpoint_host':
            for path in sorted((root/'bytecheckpoint').rglob('*')):
                if path.is_file():
                    hasher=hashlib.sha256()
                    with path.open('rb') as stream:
                        for block in iter(lambda:stream.read(8<<20),b''):hasher.update(block)
                    artifacts[str(path.relative_to(out))]=hasher.hexdigest()
        rows.append(dict(rank=rank,pid=report['pid'],artifacts=artifacts,state_bytes=receipt['state_bytes']))
    if not source:
        contract=dict(backend=backend,world_size=4,step=report['schedule']['checkpoint_step'],lr_horizon=report['schedule']['lr_horizon'],
                      generation=owner['receipt']['generation'],ranks=rows)
        (out/'d2_restart_contract.json').write_text(json.dumps(contract,indent=2))
    return dict(status='pass',scope='Qwen TP4 blocking full_state D2 '+('restart and exact continuation' if source else 'source only'),ranks=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source-run',type=Path)
    a=p.parse_args()
    try:result=audit(a.out,a.source_run)
    except BaseException as error:result=dict(status='fail',error=repr(error))
    (a.out/'d2-acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    return int(result['status']!='pass')
if __name__=='__main__':raise SystemExit(main())
