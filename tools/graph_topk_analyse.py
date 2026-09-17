#!/usr/bin/env python3
"""Same-run device task overlap. Never equate task envelopes with core occupancy."""
import argparse
import csv
import json
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
    for rank in range(4):
        profile = run / f'rank_{rank}/profiler'
        with next(profile.rglob('kernel_details.csv')).open() as f:
            rows = list(csv.DictReader(f))
        tasks = []
        for row in rows:
            name = row['Name']; start = float(row['Start Time(us)'])
            phase = ('auxiliary' if any(s in name for s in ('chain-DetectionChain', 'ParameterScan')) else
                     'optimizer' if 'optimizer-' in name else
                     'backward' if 'Gradients/' in name else
                     'forward' if 'network-' in name else 'other')
            tasks.append(dict(name=name, phase=phase, start=start, end=start + float(row['Duration(us)']),
                              stream=row['Stream ID'], type=row['Type'], core=row['Accelerator Core']))
        with next(profile.rglob('op_summary*.csv')).open() as f:
            bounds = sorted(set(float(r['Task Start Time(us)'].strip()) for r in csv.DictReader(f) if r['OP Type'] == 'GetNext'))
        steps = []
        for index, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
            selected = [t for t in tasks if a <= t['start'] < b]
            aux = [t for t in selected if t['phase'] == 'auxiliary']
            train = [t for t in selected if t['phase'] in ('forward', 'backward')]
            updates = [t for t in selected if t['phase'] == 'optimizer' and 'Assign' in t['name']]
            ai = [(t['start'], t['end']) for t in aux]
            ti = [(t['start'], t['end']) for t in train]
            next_updates = [t for t in tasks if t['start'] >= b and t['phase']=='optimizer' and 'Assign' in t['name']]
            deadline = min((t['start'] for t in (next_updates if serial else updates)), default=None)
            source_updates = updates if serial else [t for t in tasks if t['end'] <= a and t['phase']=='optimizer' and 'Assign' in t['name']]
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
        reports.append(dict(rank=rank, steps=steps, auxiliary_task_count=len(aux_all),
                            actual_task_overlap_us=overlap,
                            observed_overlap='present' if overlap > 0 else 'not_observed',
                            tasks=tasks))
    return dict(run=str(run), ranks=reports,
                clock='same kernel_details device timestamp domain',
                limits=['Task envelopes are not active-core counts or simultaneous instruction measurements.',
                        'Earliest optimizer Assign includes optimizer state; sufficient but not necessary source protection deadline.',
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
