#!/usr/bin/env python3
"""Audit completed formal runs, keeping conditional gates and exclusions explicit."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',type=Path,default=Path('/models/npu_nvme_exp/user7-stack/graph-topk-20260917-001'))
    parser.add_argument('--output',type=Path,default=Path('results/graph-topk-20260917'))
    args=parser.parse_args()
    measurements=json.loads((args.output/'measurements.json').read_text())['runs']
    formal=[r for r in measurements if not r['diagnostic'] and r['steps']==20 and r['warmup_steps']==12]
    expected={'main':14,'auxiliary':5}
    checks=[]
    for role,expected_count in expected.items():
        runs=[r for r in formal if r['role']==role]
        if len(runs)!=expected_count:raise ValueError(f'{role}: expected {expected_count} valid runs, got {len(runs)}')
        baseline=next(r for r in runs if r['level']==0)
        for row in runs:
            config=json.loads((args.campaign/row['name']/'run-config.json').read_text())
            if config['layout']!='serial':raise ValueError('unverified parallel timing entered report')
            per_rank=[]
            for rank in range(4):
                report=json.loads((args.campaign/row['name']/f'rank_{rank}/result.json').read_text())
                base=json.loads((args.campaign/baseline['name']/f'rank_{rank}/result.json').read_text())
                assert report['status']=='pass'
                assert len(report['losses'])==20
                assert report['restore']['identity']==base['restore']['identity']
                assert report['data_sha256']==base['data_sha256']
                loss_delta=max(abs(a['loss']-b['loss']) for a,b in zip(report['losses'],base['losses']))
                assert loss_delta==0.0
                if row['level']:
                    assert report['version_end']-report['version_begin']==20
                    assert report['score_oracle']['status']=='pass'
                if row['level']>=6:
                    assert report['topk_diagnostic']['order_violations']==0
                    assert report['topk_diagnostic']['threshold_gap']>=-1e-5
                per_rank.append(dict(rank=rank,loss_delta=loss_delta,output_check='pass' if row['level'] else 'not_applicable'))
            checks.append(dict(run=row['name'],role=role,status='pass',ranks=per_rank))
    dependency=json.loads((args.campaign/'main-g6-serial-profile/dependency-gate.json').read_text())
    assert dependency['status']=='pass' and dependency['actual_task_overlap_us']==0
    exclusions=[]
    for run in args.campaign.iterdir():
        if not run.is_dir():continue
        for name in ('validation-exclusion.json','timing-exclusion.json','compile-interruption.json'):
            if (run/name).exists():exclusions.append(dict(run=run.name,evidence=name))
    result=dict(status='completed_conditional_matrix_with_capability_gaps',formal_configurations=len(checks),
        optimizer_steps_per_configuration=20,ranks_per_configuration=4,independent_repeats='one except two G0 runs per model',
        checks=checks,serial_dependency_gate=dependency,exclusions=exclusions,
        parallel_capacity='unresolved: target graph variants did not compile',
        device_consumer='not entered: main-model detection did not pass or approach budget',
        persistence_claim=False,cross_family_claim=False)
    (args.output/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'Passed audit: {len(checks)} configurations, all four ranks, exact training-loss equality')


if __name__=='__main__':main()
