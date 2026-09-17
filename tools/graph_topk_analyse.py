#!/usr/bin/env python3
"""Same-run device task overlap. Never equate task envelopes with core occupancy."""
import argparse
import csv
from collections import Counter
import json
import re
import statistics
from pathlib import Path


def merge(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]: result[-1][1] = max(result[-1][1], b)
        else: result.append([a, b])
    return result


def duration(intervals): return sum(b - a for a, b in merge(intervals))


def intersection(left, right):
    a, b = merge(left), merge(right)
    i = j = 0; total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]: i += 1
        else: j += 1
    return total


def analyse(run):
    reports = []
    configuration = json.loads((run / 'run-config.json').read_text())
    serial = configuration['layout'] == 'serial'
    schema = json.loads(Path(configuration['strategy']).read_text())
    model_parameters = sum(t['role']=='model' for t in schema['tensors'])
    candidate_parameters = len(json.loads((run/'rank_0/result.json').read_text()).get('geometry',{}).get('parameters',[]))
    # AdamW: three Assign nodes per model parameter; later Assign nodes are
    # auxiliary output buffers despite inheriting optimizer scope in profiling.
    def is_state_write(task):
        match=re.search(r'optimizer-AdamW/Assign-op(\d+)/',task['name'])
        return bool(match and int(match.group(1)) < 3*model_parameters)
    for rank in range(4):
        profile = run / f'rank_{rank}/profiler'
        with next(profile.rglob('kernel_details.csv')).open() as f:
            rows = list(csv.DictReader(f))
        pmu = {}
        for core in sorted(set(r['Accelerator Core'] for r in rows)):
            selected = [r for r in rows if r['Accelerator Core']==core]
            metrics = {}
            for key in ('Block Dim','Mix Block Dim','aic_mac_ratio','aiv_vec_ratio','cube_utilization(%)'):
                values = []
                for r in selected:
                    try: values.append(float(r[key]))
                    except (ValueError,KeyError): pass
                if values:metrics[key]=dict(minimum=min(values),median=statistics.median(values),maximum=max(values))
            pmu[core]=dict(task_count=len(selected),metrics=metrics)
        tasks = []
        for row in rows:
            name = row['Name']; start = float(row['Start Time(us)'])
            # Morph expansion strips scopes from these minimal-probe tasks.
            minimal_morph = configuration['level']==1 and ('Default/StridedSlice-op' in name or 'Default/ReduceSum-op' in name)
            scanner_match = re.search(r'KernelLaunch::Default/(Sub|Mul|ReduceSum|UnsortedSegmentSum)-op(\d+)/',name)
            scanner_morph = False
            if configuration['level']>=2 and scanner_match:
                kind,number=scanner_match[1],int(scanner_match[2])
                scanner_morph = (1<=number<=candidate_parameters if kind in ('Sub','Mul')
                                 else 0<=number<candidate_parameters)
            auxiliary_assign = bool(re.search(r'optimizer-AdamW/Assign-op(\d+)/', name)) and not is_state_write(dict(name=name))
            phase = ('auxiliary' if (minimal_morph or scanner_morph or auxiliary_assign or any(s in name for s in ('chain-DetectionChain', 'ParameterScan'))) else
                     'optimizer' if 'optimizer-' in name else
                     'backward' if 'Gradients/' in name else
                     'forward' if 'network-' in name else 'other')
            tasks.append(dict(name=name, phase=phase, start=start, end=start + float(row['Duration(us)']),
                              stream=row['Stream ID'], type=row['Type'], core=row['Accelerator Core'],
                              output_shape=row['Output Shapes'], output_dtype=row['Output Data Types']))
        with next(profile.rglob('op_summary*.csv')).open() as f:
            bounds = sorted(set(float(r['Task Start Time(us)'].strip()) for r in csv.DictReader(f) if r['OP Type'] == 'GetNext'))
        # In the accepted serial layout, compiler TensorMove copies between the
        # last Adam write and the named chain's end belong to auxiliary capture.
        # Scanner identity above is independent of this time-window attribution.
        if serial and configuration['level']>=2:
            for a,b in zip(bounds,bounds[1:]+[max(t['end'] for t in tasks)+1]):
                current=[t for t in tasks if a<=t['start']<b]
                ready=max((t['end'] for t in current if is_state_write(t)),default=None)
                end=max((t['end'] for t in current if t['phase']=='auxiliary'),default=None)
                if ready is not None and end is not None:
                    for t in current:
                        if ready<=t['start']<end and t['type']=='TensorMove':
                            t['phase']='auxiliary';t['attribution']='post-update compiler capture copy'
        steps = []
        for index, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
            selected = [t for t in tasks if a <= t['start'] < b]
            aux = [t for t in selected if t['phase'] == 'auxiliary']
            train = [t for t in selected if t['phase'] in ('forward', 'backward')]
            updates = [t for t in selected if is_state_write(t)]
            ai = [(t['start'], t['end']) for t in aux]
            ti = [(t['start'], t['end']) for t in train]
            next_updates = [t for t in tasks if t['start'] >= b and is_state_write(t)]
            deadline = min((t['start'] for t in (next_updates if serial else updates)), default=None)
            source_updates = updates if serial else [t for t in tasks if t['end'] <= a and is_state_write(t)]
            ready = max((t['end'] for t in source_updates), default=None)
            first = min((t['start'] for t in aux), default=None)
            last = max((t['end'] for t in aux), default=None)
            steps.append(dict(step=index, interval_us=b-a, auxiliary_tasks=len(aux),
                auxiliary_busy_union_us=duration(ai), training_fb_busy_union_us=duration(ti),
                auxiliary_training_intersection_us=intersection(ai, ti),
                auxiliary_envelope_us=max(t['end'] for t in aux)-min(t['start'] for t in aux) if aux else 0,
                auxiliary_streams=sorted(set(t['stream'] for t in aux)), training_streams=sorted(set(t['stream'] for t in train)),
                source_ready_us=ready, first_auxiliary_start_us=first, starts_after_source_update=(first >= ready) if first is not None and ready is not None else None,
                last_auxiliary_end_us=last, earliest_next_writing_optimizer_assign_us=deadline,
                finished_before_all_optimizer_assigns=(last <= deadline) if last is not None and deadline is not None else None))
        aux_all = [t for t in tasks if t['phase'] == 'auxiliary']
        if not aux_all:
            raise ValueError('no named auxiliary tasks: cannot prove execution or overlap')
        overlap = sum(s['auxiliary_training_intersection_us'] for s in steps)
        phase_intervals={phase:[[t['start'],t['end']] for t in tasks if t['phase']==phase] for phase in ('forward','backward','optimizer','auxiliary')}
        grouped = {}
        for phase in ('forward','backward','optimizer','auxiliary','other'):
            totals = Counter()
            counts = Counter()
            for t in tasks:
                if t['phase']==phase:
                    totals[t['type']]+=t['end']-t['start'];counts[t['type']]+=1
            grouped[phase]=[dict(type=kind,sum_task_duration_us=total,count=counts[kind])
                            for kind,total in totals.most_common(20)]
        reports.append(dict(rank=rank, steps=steps, auxiliary_task_count=len(aux_all), phase_intervals=phase_intervals,
                            task_pmu_statistics=pmu,top_task_types_by_phase=grouped,
                            actual_task_overlap_us=overlap,
                            observed_overlap='present' if overlap > 0 else 'not_observed',
                            tasks=tasks))
    return dict(run=str(run), ranks=reports,
                clock='same kernel_details device timestamp domain',
                limits=['Task envelopes are not active-core counts or simultaneous instruction measurements.',
                        'Block Dim is launch geometry, not time-resolved active-core occupancy. PMU ratios are per-task distributions, not instruction-interval overlap.',
                        'HCCL rows with stream N/A are collective envelopes including rank waiting, not individual arithmetic kernels; their duration is not pure link-transfer time.',
                        'Expanded scanner Sub/Mul op indices 1..candidate_parameters and ReduceSum/UnsortedSegmentSum 0..candidate_parameters-1 identify the cumulative G6 scanner. Serial post-update TensorMove copies are separately attributed by time window.',
                        'AdamW state writes use Assign-op indices below 3 times model parameter count; later auxiliary Assign nodes may inherit optimizer scope. Validate this mapping against each compiled graph.',
                        'Only complete GetNext-to-GetNext intervals are used; final auxiliary drain is excluded from overlap statistics.',
                        'No window-specific HBM bandwidth claim. Profiling is excluded from performance comparison.'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path)
    args = parser.parse_args(); report = analyse(args.run)
    output = args.run / 'device-overlap.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    summary=dict(report, ranks=[{k:v for k,v in r.items() if k!='tasks'} for r in report['ranks']])
    (args.run / 'device-overlap-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(output)

if __name__ == '__main__': main()
