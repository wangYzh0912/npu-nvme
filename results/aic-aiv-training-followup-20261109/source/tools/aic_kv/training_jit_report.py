"""Reduce training observations per physical card without pooling replicas."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics

from .training_jit_campaign import RESULTS


def longest_joint_intervals(bins,window_ids):
    maxima=dict.fromkeys(window_ids,0);length=0;previous_end=None;previous_id=None
    for row in bins:
        start=int(row['window_start_ns']);end=int(row['window_end_ns']);identity=row['window_id']
        if row['valid']=='True' and float(row['aic_activity_pct'])<=10 and float(row['aiv_activity_pct'])>=50:
            length=length+end-start if start==previous_end and identity==previous_id else end-start
            maxima[identity]=max(maxima[identity],length)
        else:length=0
        previous_end=end;previous_id=identity
    return maxima


def summarize(run):
    quality=json.loads((run/'sample-evidence.json').read_text())
    config=json.loads((run/'run-config.json').read_text())
    rows=[]
    for rank in quality['ranks']:
        target=run/f"rank{rank['rank']}"
        bins=list(csv.DictReader((target/'pmu/core_timeseries.csv').open()))
        valid=[b for b in bins if b['valid']=='True']
        if not valid:raise ValueError('no valid PMU bins: '+str(target))
        durations=[int(b['window_end_ns'])-int(b['window_start_ns']) for b in valid]
        seconds=sum(durations)/1e9
        means={e:sum(float(b[e+'_activity_pct'])*w for b,w in zip(valid,durations))/sum(durations) for e in ('aic','aiv')}
        fractions={f'aic_le_{low}_aiv_ge_{high}':sum(w for b,w in zip(valid,durations) if float(b['aic_activity_pct'])<=low and float(b['aiv_activity_pct'])>=high)/sum(durations)
                   for low in (5,10,20) for high in (30,50,70)}
        timing=rank['control_median_ms']
        low_time=sum(w for b,w in zip(valid,durations) if float(b['aic_activity_pct'])<=10)
        high_time=sum(w for b,w in zip(valid,durations) if float(b['aiv_activity_pct'])>=50)
        joint_time=sum(w for b,w in zip(valid,durations) if float(b['aic_activity_pct'])<=10 and float(b['aiv_activity_pct'])>=50)
        longest=longest_joint_intervals(bins,[w['id'] for w in rank['windows']])
        accepted=(quality['measurement_quality']=='valid' and len(valid)/len(bins)>=.95 and seconds>=30 and abs(rank['profile_overhead_fraction'])<=.1)
        hbm_path=target/'hbm/hbm-summary.json'
        hbm={}
        if hbm_path.is_file():
            windows=json.loads(hbm_path.read_text())['windows']
            covered=[w for w in windows if w['coverage']>=.95]
            hbm_seconds=sum(w['end_ns']-w['begin_ns'] for w in covered)/1e9
            hbm=dict(hbm_complete_step_count=len(covered),hbm_total_step_count=len(windows),
                     hbm_min_coverage=min(w['coverage'] for w in windows),
                     hbm_complete_step_seconds=hbm_seconds,
                     hbm_measurement_accepted=hbm_seconds>=30 and len(covered)/len(windows)>=.95,
                     hbm_read_bytes_per_complete_step=statistics.mean(w['read_bytes'] for w in covered) if covered else None,
                     hbm_write_bytes_per_complete_step=statistics.mean(w['write_bytes'] for w in covered) if covered else None)
        match=re.search(r'-r(\d+)(?:-|$)',run.name)
        if match is None:raise ValueError('run lacks independent replicate identity')
        rows.append(dict(run=run.name,replicate=int(match[1]),model_process_run_id=config['run_id'],
                         jit_level=config['jit_level'],sequence_length=config['seq_length'],micro_batch=config['micro_batch'],rank=rank['rank'],physical_device=rank['rank']+4,
                         control_ms=timing,measured_ms=rank['measured_median_ms'],sample_seconds=seconds,measurement_accepted=accepted,
                         warmup_cv=rank['warmup_cv'],profile_overhead_fraction=rank['profile_overhead_fraction'],
                         aic_mean_pct=means['aic'],aiv_mean_pct=means['aiv'],joint_fraction=fractions['aic_le_10_aiv_ge_50'],
                         low_aic_fraction=low_time/sum(durations),high_aiv_fraction=high_time/sum(durations),
                         low_aic_given_high_aiv=joint_time/high_time if high_time else None,
                         joint_state_sampled_ms_per_step=joint_time/1e6/len(rank['windows']),
                         mean_longest_joint_interval_ms=statistics.mean(longest.values())/1e6,
                         capacity_busy_aic_ms_per_step=sum(float(b['aic_activity_pct'])*w for b,w in zip(valid,durations))/1e8/len(rank['windows']),
                         capacity_busy_aiv_ms_per_step=sum(float(b['aiv_activity_pct'])*w for b,w in zip(valid,durations))/1e8/len(rank['windows']),
                         pmu_complete_window_count=len(rank['windows']),
                         pmu_valid_time_fraction_of_step_windows=sum(durations)/sum(w['end_ns']-w['begin_ns'] for w in rank['windows']),
                         tokens_per_second=config['seq_length']*config['micro_batch']*1000/timing,
                         valid_bin_fraction=len(valid)/len(bins),threshold_sensitivity=fractions,**hbm))
    return rows


def recurrence(rows,mode):
    expected={(s,b) for s in (128,512,2048) for b in (1,2)}
    result=[]
    for device in (4,5,6,7):
        cells=[]
        for seq,batch in sorted(expected):
            selected=[r for r in rows if r['jit_level']==mode and r['physical_device']==device and r['sequence_length']==seq and r['micro_batch']==batch]
            complete=(len(selected)==3 and len({r['model_process_run_id'] for r in selected})==3 and
                      {r['replicate'] for r in selected}=={0,1,2} and
                      all(r['measurement_accepted'] for r in selected))
            cells.append(dict(sequence_length=seq,micro_batch=batch,replicates=len(selected),complete=complete,
                              low_aic_majority_in_every_repeat=complete and all(r['low_aic_fraction']>.5 for r in selected),
                              majority_in_every_repeat=complete and all(r['joint_fraction']>.5 for r in selected)))
        result.append(dict(physical_device=device,cells=cells,
                           matrix_complete=all(x['complete'] for x in cells),
                           low_aic_alone_covers_matrix=all(x['low_aic_majority_in_every_repeat'] for x in cells),
                           low_aic_high_aiv_covers_matrix=all(x['majority_in_every_repeat'] for x in cells)))
    return result


def fragment_tasks(root):
    import gzip
    from collections import Counter
    result=[]
    for run in sorted((root/'fragments/runs').glob('*')):
        for task_file in run.glob('kernel_details-*.gz'):
            tasks=list(csv.DictReader(gzip.open(task_file,'rt')))
            result.append(dict(run=run.name,source=str(task_file),diagnostic_iterations=8,
                tasks=len(tasks),tasks_per_update=len(tasks)/8,
                accelerator_core_counts=dict(Counter(t['Accelerator Core'] for t in tasks)),
                tensor_move_tasks_per_update=sum('TensorMove' in t['Name'] for t in tasks)/8,
                assign_tasks_per_update=sum('/Assign' in t['Name'] for t in tasks)/8,
                kernel_counts=dict(Counter(t['Name'] for t in tasks)),
                duration_sum_us=sum(float(t['Duration(us)']) for t in tasks),
                interpretation='diagnostic device tasks only; not normal critical-path or overlap proof'))
    (root/'fragment-task-evidence.json').write_text(json.dumps(result,indent=2)+chr(10))


def numerical_diagnostics(root,rows):
    """Repeatability is distinct from cross-JIT numerical equivalence."""
    groups={}
    for row in rows:
        rank=root/'training/runs'/row['run']/f"rank{row['rank']}"
        payload=json.loads((rank/'training.json').read_text())['payload']
        losses={v['step']:v['loss'] for v in payload['losses']}
        initial=json.loads((rank/'state-initial.json').read_text())['sha256']
        final=json.loads((rank/f"state-step-{payload['final_step']}.json").read_text())['sha256']
        groups.setdefault((row['jit_level'],row['sequence_length'],row['micro_batch'],row['rank']),[]).append(
            dict(run=row['run'],losses=losses,initial=initial,final=final,
                 overflow=any(v['overflow'] for v in payload['losses'])))
    repeated=[];cross=[]
    for key,runs in sorted(groups.items()):
        repeated.append(dict(jit_level=key[0],sequence_length=key[1],micro_batch=key[2],rank=key[3],
            runs=[r['run'] for r in runs],all_initial_hashes_identical=len({r['initial'] for r in runs})==1,
            all_final_hashes_identical=len({r['final'] for r in runs})==1,
            loss_trajectories_identical=all(r['losses']==runs[0]['losses'] for r in runs),
            all_losses_finite=all(math.isfinite(v) for r in runs for v in r['losses'].values()),
            any_overflow=any(r['overflow'] for r in runs)))
    cells=sorted({k[1:] for k in groups})
    for cell in cells:
        modes=sorted(k[0] for k in groups if k[1:]==cell)
        for i,left in enumerate(modes):
            for right in modes[i+1:]:
                a=groups[(left,*cell)][0];b=groups[(right,*cell)][0]
                common=sorted(set(a['losses'])&set(b['losses']))
                differences=[abs(a['losses'][s]-b['losses'][s]) for s in common]
                cross.append(dict(left=a['run'],right=b['run'],rank=cell[2],
                    common_step_count=len(common),common_last_step=max(common),
                    initial_hashes_identical=a['initial']==b['initial'],
                    max_loss_abs_difference=max(differences),losses_identical=not any(differences),
                    full_model_numerical_equivalence_accepted=False,
                    scope='descriptive diagnostics; no post-hoc acceptance tolerance'))
    (root/'numerical-diagnostics.json').write_text(json.dumps(dict(
        within_mode_repeats=repeated,cross_jit_common_prefix=cross),indent=2)+chr(10))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=RESULTS);args=parser.parse_args()
    root=args.root;rows=[]
    fragment_tasks(root)
    for run in sorted((root/'training/runs').glob('*')):
        if (run/'sample-evidence.json').is_file():rows.extend(summarize(run))
    numerical_diagnostics(root,rows)
    (root/'measurements.json').write_text(json.dumps(rows,indent=2)+chr(10))
    fields=[k for k in rows[0] if k!='threshold_sensitivity'] if rows else []
    with (root/'measurements.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    (root/'recurrence.json').write_text(json.dumps({mode:recurrence(rows,mode) for mode in ('O1','O2')},indent=2)+chr(10))
    print(json.dumps(dict(run_count=len({r['run'] for r in rows}),device_rows=len(rows),accepted_rows=sum(r['measurement_accepted'] for r in rows))),flush=True)


if __name__=='__main__':main()
