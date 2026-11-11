"""Host-clock measurements; ranks are observations, processes are repetitions."""
import math
import statistics
from pathlib import Path

from npu_nvme.runtime.training_catalog import read_checked

METHODS=('none','checkpoint_io_sink','ours')
SAVE_STEPS=tuple(range(28,65,4))


def distribution(values, *, minimum_p95=30):
    values=sorted(values)
    if not values or any(not math.isfinite(v) or v<0 for v in values):
        raise ValueError('nonempty finite nonnegative measurements required')
    return dict(n=len(values),median=statistics.median(values),
                p95=values[math.ceil(.95*len(values))-1] if len(values)>=minimum_p95 else None,
                p95_status='reported' if len(values)>=minimum_p95 else 'insufficient_samples',
                minimum=min(values),maximum=max(values))


def process_metrics(run):
    ranks=run['ranks']
    if sorted(rank['rank'] for rank in ranks)!=list(range(4)):
        raise ValueError('exactly four ranks required')
    durations={};computes={};saves={};timestamps={}
    for rank in ranks:
        observations={v['step']:v for v in rank['loss_steps']}
        if sorted(observations)!=list(range(28,65)):
            raise ValueError('formal steps must be exactly 28..64')
        for step,row in observations.items():
            begin,end,resume=row['begin_ns'],row['end_ns'],row['resume_ns']
            if not 0<begin<=end<=resume:raise ValueError('invalid host-clock step interval')
            durations.setdefault(step,[]).append((resume-begin)/1e9)
            computes.setdefault(step,[]).append((end-begin)/1e9)
            timestamps.setdefault(step,[]).append((begin,resume))
        checkpoints={v['step']:v for v in rank['checkpoints']}
        if run['group']=='none':
            if checkpoints:raise ValueError('T0 must not save')
        elif tuple(sorted(checkpoints))!=SAVE_STEPS:
            raise ValueError('ten formal saves required per process')
        for step,row in checkpoints.items():
            saves.setdefault(step,[]).append(row['blocking_seconds'])
    windows=[dict(step=step,seconds=max(durations[step]),
                  compute_seconds=max(computes[step]),
                  blocking_seconds=max(saves[step]) if step in saves else 0)
             for step in SAVE_STEPS]
    elapsed=(max(v[1] for v in timestamps[64])-min(v[0] for v in timestamps[28]))/1e9
    return dict(name=run['name'],group=run['group'],replicate=run['replicate'],
                windows=windows,aligned_step_seconds=distribution([v['seconds'] for v in windows]),
                regular_step_seconds=distribution([max(v) for step,v in durations.items() if step%4]),
                formal_elapsed_seconds=elapsed,formal_tokens=37*128,
                formal_tokens_per_second=37*128/elapsed,
                rank_step_seconds={str(rank['rank']):distribution([
                    (v['resume_ns']-v['begin_ns'])/1e9 for v in rank['loss_steps']]) for rank in ranks})


def analyse(record):
    processes=[process_metrics(run) for run in record['runs']]
    groups={}
    for method in METHODS:
        rows=[row for row in processes if row['group']==method]
        if sorted(row['replicate'] for row in rows)!=[0,1,2]:
            raise ValueError('three independent processes required for '+method)
        groups[method]=dict(processes=3,aligned_step_seconds=distribution([
            v['seconds'] for row in rows for v in row['windows']]),
            tokens_per_second_by_process=[row['formal_tokens_per_second'] for row in rows])
    paired=[]
    for replicate in range(3):
        rows={row['group']:row for row in processes if row['replicate']==replicate}
        paired.append(dict(replicate=replicate,
            sink_minus_t0_seconds=[b['seconds']-a['seconds'] for a,b in zip(rows['none']['windows'],rows['checkpoint_io_sink']['windows'])],
            full_minus_sink_seconds=[b['seconds']-a['seconds'] for a,b in zip(rows['checkpoint_io_sink']['windows'],rows['ours']['windows'])]))
    return dict(status='host_timing_measured',processes=processes,groups=groups,paired=paired,
        quantile='nearest-rank P95, only with at least 30 aligned windows; rank maxima per window',
        limitations=['T2 minus T1 includes socket, owner, validation, storage, flush and publication; it is not NVMe service time.',
                    'Formal throughput covers steps 28..64; compilation, initial validation, final T0 oracle hashing and drain are excluded.',
                    'Host-clock observations do not establish device engine overlap, PMU activity or peak memory.'])


def probe_overhead(record,diagnostic_root):
    """Screen matched steps only; one trace process cannot establish acceptance."""
    methods=[]
    for method in METHODS:
        baseline=next(run for run in record['runs'] if run['group']==method and run['replicate']==0)
        traced=[read_checked(Path(diagnostic_root)/method/f'rank_{rank}/training.json')
                for rank in range(4)]
        def windows(ranks):
            return [max((row['resume_ns']-row['begin_ns'])/1e9 for rank in ranks
                        for row in rank['loss_steps'] if row['step']==step) for step in (28,32)]
        ordinary=windows(baseline['ranks'])
        detailed=windows([dict(loss_steps=row['losses']) for row in traced])
        methods.append(dict(method=method,baseline_seconds=ordinary,diagnostic_seconds=detailed,
            median_ratio=statistics.median(detailed)/statistics.median(ordinary)))
    return dict(status='screening_only',methods=methods,perturbation_limit=1.10,
                independent_diagnostic_processes_per_method=1,
                limitation='Two aligned windows from one diagnostic process per method do not establish the 10% observation-perturbation gate.')
