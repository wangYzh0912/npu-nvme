"""Compare a fresh-process continuation against an uninterrupted trajectory."""
import math
from pathlib import Path

from .training_catalog import read_checked


def compare(oracle, run, *, rtol=1e-5,atol=1e-6):
    oracle=Path(oracle);run=Path(run);ranks=[]
    for rank in range(4):
        baseline=read_checked(oracle/f'rank_{rank}/training.json')
        actual=read_checked(run/f'rank_{rank}/training.json')
        if baseline['status']!='pass' or actual['status']!='pass':
            raise ValueError('trajectory requires completed training')
        expected={row['step']:row for row in baseline['losses']}
        if not actual['losses']:raise ValueError('empty continuation')
        for row in actual['losses']:
            reference=expected.get(row['step'])
            if (reference is None or row['overflow'] or reference['overflow'] or
                    not math.isfinite(row['loss']) or
                    not math.isclose(row['loss'],reference['loss'],rel_tol=rtol,abs_tol=atol)):
                raise ValueError(f'loss/optimizer mismatch at rank {rank} step {row["step"]}')
        step=actual['final_step']
        for kind in ('state','control'):
            name=f'{kind}-step-{step}.json'
            if read_checked(oracle/f'rank_{rank}'/name)!=read_checked(run/f'rank_{rank}'/name):
                raise ValueError(f'{kind} differs at rank {rank} step {step}')
        ranks.append(dict(rank=rank,steps=len(actual['losses']),final_step=step))
    return dict(status='pass',ranks=ranks,loss_tolerance=dict(rtol=rtol,atol=atol),
                state_comparison='exact SHA256 and geometry',controls_comparison='exact')
