"""Explicit observational acceptance: completeness, stability and perturbation."""
import json,statistics
from pathlib import Path

def training(run):
    run=Path(run);execution=json.loads((run/'result.json').read_text())
    ranks=[]
    for p in sorted(run.glob('rank_*/training.json')):
        d=json.loads(p.read_text())['payload'];steps=d['losses'];profile=d.get('activity_profile')
        times={r['step']:(r['end_ns']-r['begin_ns'])/1e6 for r in steps}
        start=profile['start_after_step'] if profile else 17
        stop=profile['stop_after_step'] if profile else min(times)
        before=[times[n] for n in range(max(2,start-5),start+1) if n in times]
        measured=[times[n] for n in range(start+1,stop+1) if n in times] if profile else [times[n] for n in range(18,25) if n in times]
        baseline=statistics.median(before);cv=statistics.pstdev(before)/statistics.mean(before)
        overhead=statistics.median(measured)/baseline-1 if profile else None
        rank=dict(rank=d['rank'],status=d['status'],final_step=d.get('final_step'),loss_final=steps[-1]['loss'],
                  warmup_cv=cv,control_median_ms=baseline,measured_median_ms=statistics.median(measured),
                  profile_overhead_fraction=overhead,warmup_stable=cv<=.05)
        ranks.append(rank)
    valid_execution=execution.get('validation_status')=='pass' and len(ranks)==4 and all(r['status']=='pass' for r in ranks)
    reasons=[]
    if not valid_execution:reasons.append('execution_or_rank_validation_failed')
    if not all(r['warmup_stable'] for r in ranks):reasons.append('warmup_cv_above_5pct')
    if any(r['profile_overhead_fraction'] is not None and r['profile_overhead_fraction']>.1 for r in ranks):reasons.append('profiler_overhead_above_10pct')
    quality=('invalid_execution' if not valid_execution else 'diagnostic_unstable' if not all(r['warmup_stable'] for r in ranks) else
             'diagnostic_profiler_overhead' if any(r['profile_overhead_fraction'] is not None and r['profile_overhead_fraction']>.1 for r in ranks) else 'valid')
    return dict(quality=quality,reasons=reasons,execution=execution,ranks=ranks)

def cross_rank_counts(summaries):
    """Flag large count differences for inspection, not as proof of corruption."""
    counts=[s['engine_counts'].get('aic',0)+s['engine_counts'].get('aiv',0)+s['engine_counts'].get('mixed',0) for s in summaries]
    median=statistics.median(counts)
    return dict(counts=counts,median=median,suspect=[i for i,c in enumerate(counts) if median and abs(c-median)/median>.1])
