#!/usr/bin/env python3
"""Audit observed outputs without treating missing experiments as passing."""
import argparse
import json
from pathlib import Path
import sys
import statistics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from npu_nvme.runtime.training_catalog import read_checked
from npu_nvme.experiments.stages import summarise


def audit(root):
    results=[];failed=[]
    paths=[]
    for parent in sorted((root/'runs').iterdir()):
        if (parent/'suite').exists():
            result=parent/'result.json'
            if result.exists() and json.loads(result.read_text()).get('validation_status')=='pass':
                paths.extend(sorted((parent/'suite').iterdir()))
        else:paths.append(parent)
    for path in paths:
        outcome=path/'result.json'
        if not outcome.exists() or json.loads(outcome.read_text()).get('validation_status')!='pass':continue
        ranks=[read_checked(path/f'rank_{rank}'/'training.json') for rank in range(4)]
        role='auxiliary' if path.name.startswith('auxiliary-') else 'main'
        experiment=ranks[0]['incremental'];group=experiment['group'];probe=any(s.get('probe') for s in experiment['steps'])
        row=dict(run=str(path.relative_to(root/'runs')),role=role,group=group,canonical='canonical' in path.name,
                 suite=experiment.get('suite',False),
                 profiled=experiment.get('profiled',False),probe=probe,status='pass')
        try:
            if any(r['status']!='pass' or len(r['losses'])!=20+experiment.get('warmup_steps',4) or not r['initial_full_restore']['verified'] for r in ranks):
                raise ValueError('rank interval or initial state invalid')
            completions=[]
            for rank in range(4):
                completions.extend(json.loads(p.read_text()) for p in (path/f'rank_{rank}'/'incremental-completions').glob('step-*.json'))
            owner=root/'owners'/path.name
            if group!='B0' or owner.exists():
                catalog=json.loads((owner/'raw-catalog.json').read_text());closed=json.loads((owner/'result.json').read_text())
                if closed.get('status')!='pass' or not closed.get('closed') or len(completions)!=80:
                    raise ValueError('raw commit/drain incomplete')
                if [v['step'] for v in catalog['commits']]!=list(range(1,21)):raise ValueError('raw step sequence differs')
                row['physical_frame_bytes']=sum(v['frame_bytes']+4096 for c in catalog['commits'] for v in c['ranks'])
                row['media_readback_verified']=all(c.get('media_verified') for c in catalog['commits'])
            row.update(summarise(ranks,completions))
            row['losses']=[v['loss'] for v in ranks[0]['losses'][experiment.get('warmup_steps',4):]]
            row['peak_framework_hbm_bytes']=[v['incremental']['memory_peak_bytes'] for v in ranks]
            if row['canonical']:
                curves=[json.loads(p.read_text()) for p in sorted((path/'fidelity').glob('step-*/result.json'))]
                if len(curves)!=20 or any(v['loss_difference_status']!='measured' for v in curves):
                    raise ValueError('fidelity/loss series incomplete')
                row['fidelity']=curves
        except Exception as error:
            row.update(status='failed',error=repr(error));failed.append(row)
        results.append(row)
    for role in ('main','auxiliary'):
        baseline=next((r for r in results if r['role']==role and r['group']=='B0' and not r['probe'] and r['status']=='pass'),None)
        if baseline:
            for row in results:
                if row['role']==role and 'losses' in row:
                    row['training_numerics_exact']=row['losses']==baseline['losses']
                    row['training_loss_max_absolute_difference']=max(abs(a-b) for a,b in zip(row['losses'],baseline['losses']))
    document=dict(runs=results,failed=failed,quality_budget=None,
                  quality_conclusion='No task-specific quality threshold supplied; report measured curves without pass claim.')
    (root/'validation-summary.json').write_text(json.dumps(document,indent=2)+'\n')
    return document


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('campaign',type=Path);args=parser.parse_args()
    result=audit(args.campaign);print(json.dumps(dict(runs=len(result['runs']),failed=len(result['failed']))))
