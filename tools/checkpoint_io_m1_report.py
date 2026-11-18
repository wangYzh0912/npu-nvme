#!/usr/bin/env python3
"""Extract compact checkpoint and AIC/AIV evidence for the M1 report."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return values[low]
    return values[low] * (high - index) + values[high] * (index - low)


def write_control(control_json, database, out_dir):
    result = read_json(control_json)
    conn = sqlite3.connect(database)
    conn.execute('PRAGMA temp_store=MEMORY')
    ids = {value: key for key, value in conn.execute('SELECT id,value FROM STRING_IDS')}
    windows = result['windows']
    rows = []
    for window in windows:
        begin = window['realtime_begin_ns']
        end = window['realtime_end_ns']
        item = dict(mode=window['mode'], repeat=window['repeat'],
                    elapsed_ms=window['elapsed_ms'], aic_iterations=window['aic_iterations'],
                    aiv_iterations=window['aiv_iterations'])
        for core_type, label in ((ids['AIC'], 'aic'), (ids['AIV'], 'aiv')):
            values = [r[0] for r in conn.execute(
                'SELECT usage FROM SAMPLE_PMU_TIMELINE '
                'WHERE timestampNs BETWEEN ? AND ? AND coreType=?',
                (begin, end, core_type))]
            item[f'{label}_pmu_samples'] = len(values)
            item[f'{label}_pmu_mean_pct'] = sum(values) / len(values) if values else None
            item[f'{label}_pmu_p95_pct'] = percentile(values, .95)
            item[f'{label}_pmu_max_pct'] = max(values) if values else None
        for metric, label in ((ids['read'], 'hbm_read'), (ids['write'], 'hbm_write')):
            values = [r[0] / (1024 * 1024) for r in conn.execute(
                'SELECT SUM(bandwidth) FROM HBM '
                'WHERE timestampNs BETWEEN ? AND ? AND type=? GROUP BY timestampNs',
                (begin, end, metric))]
            item[f'{label}_samples'] = len(values)
            item[f'{label}_mean_mib_s'] = sum(values) / len(values) if values else None
            item[f'{label}_p95_mib_s'] = percentile(values, .95)
            item[f'{label}_max_mib_s'] = max(values) if values else None
        rows.append(item)

    fields = list(rows[0])
    with (out_dir / 'control-window-summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    session_start = conn.execute('SELECT startTimeNs FROM SESSION_TIME_INFO').fetchone()[0]
    bucket_ns = 10_000_000
    phases = []
    for row, window in zip(rows, windows):
        phases.append((window['realtime_begin_ns'], window['realtime_end_ns'],
                       f"{window['mode']}-{window['repeat']}"))
    buckets = defaultdict(lambda: defaultdict(list))
    query_start = min(a for a, _, _ in phases)
    query_end = max(b for _, b, _ in phases)
    for core_type, metric in ((ids['AIC'], 'aic_pmu_pct'), (ids['AIV'], 'aiv_pmu_pct')):
        for timestamp, value in conn.execute(
                'SELECT timestampNs,usage FROM SAMPLE_PMU_TIMELINE '
                'WHERE timestampNs BETWEEN ? AND ? AND coreType=?',
                (query_start, query_end, core_type)):
            bucket = (timestamp - session_start) // bucket_ns
            buckets[bucket][metric].append(value)
    for metric_id, metric in ((ids['read'], 'card_hbm_read_mib_s'), (ids['write'], 'card_hbm_write_mib_s')):
        for timestamp, value in conn.execute(
                'SELECT timestampNs,SUM(bandwidth) FROM HBM '
                'WHERE timestampNs BETWEEN ? AND ? AND type=? GROUP BY timestampNs',
                (query_start, query_end, metric_id)):
            bucket = (timestamp - session_start) // bucket_ns
            buckets[bucket][metric].append(value / (1024 * 1024))

    time_rows = []
    for bucket, metrics in sorted(buckets.items()):
        timestamp = session_start + bucket * bucket_ns
        phase = next((name for a, b, name in phases if a <= timestamp < b), 'idle_or_boundary')
        item = {'profile_offset_ms': (timestamp - session_start) / 1e6, 'phase': phase}
        for metric in ('aic_pmu_pct', 'aiv_pmu_pct',
                       'card_hbm_read_mib_s', 'card_hbm_write_mib_s'):
            values = metrics.get(metric, [])
            item[metric] = sum(values) / len(values) if values else None
            item[metric + '_samples'] = len(values)
        time_rows.append(item)
    fields = list(time_rows[0])
    with (out_dir / 'control-timeseries-10ms.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(time_rows)
    conn.close()
    return rows, time_rows


def merge(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if not merged or begin > merged[-1][1]:
            merged.append([begin, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def overlap_duration(left, right):
    total = 0.0
    i = j = 0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def distribute(intervals, origin, width):
    bins = defaultdict(float)
    for begin, end in intervals:
        first = max(0, int((begin - origin) // width))
        last = int(math.ceil((end - origin) / width))
        for index in range(first, last):
            left = origin + index * width
            right = left + width
            bins[index] += max(0.0, min(end, right) - max(begin, left))
    return bins


def write_e4_activity(e4_root, timeline_path, out_dir):
    sources = sorted(e4_root.glob('rank_*/device-profile/**/kernel_details.csv.gz'))
    per_device = defaultdict(lambda: {'aic': [], 'aiv': []})
    for source in sources:
        rank = int(source.parts[-5].split('_')[-1])
        for row in csv.DictReader(gzip.open(source, 'rt')):
            family = row.get('Accelerator Core')
            if family in ('AI_CORE', 'MIX_AIC'):
                key = 'aic'
            elif family in ('AI_VECTOR_CORE', 'MIX_AIV'):
                key = 'aiv'
            else:
                continue
            try:
                begin = float(row['Start Time(us)'].strip())
                duration = float(row['Duration(us)'].strip())
            except (TypeError, ValueError):
                continue
            if duration > 0:
                per_device[rank][key].append((begin, begin + duration))

    merged = {rank: {key: merge(intervals) for key, intervals in values.items()}
              for rank, values in per_device.items()}
    all_intervals = [interval for values in merged.values()
                     for intervals in values.values() for interval in intervals]
    origin = min(a for a, _ in all_intervals)
    end = max(b for _, b in all_intervals)
    width = 10_000.0
    rows = []
    summaries = {}
    for rank, values in sorted(merged.items()):
        aic, aiv = values['aic'], values['aiv']
        busy_aic = sum(b - a for a, b in aic)
        busy_aiv = sum(b - a for a, b in aiv)
        concurrent = overlap_duration(aic, aiv)
        summaries[rank] = {
            'capture_span_seconds': (end - origin) / 1e6,
            'aic_task_busy_union_seconds': busy_aic / 1e6,
            'aiv_task_busy_union_seconds': busy_aiv / 1e6,
            'aic_aiv_task_overlap_seconds': concurrent / 1e6,
            'aic_busy_fraction': busy_aic / (end - origin),
            'aiv_busy_fraction': busy_aiv / (end - origin),
            'overlap_fraction': concurrent / (end - origin),
        }
        common = []
        i = j = 0
        while i < len(aic) and j < len(aiv):
            left = max(aic[i][0], aiv[j][0])
            right = min(aic[i][1], aiv[j][1])
            if right > left:
                common.append((left, right))
            if aic[i][1] < aiv[j][1]:
                i += 1
            else:
                j += 1
        bin_count = math.ceil((end - origin) / width)
        aic_bins = distribute(aic, origin, width)
        aiv_bins = distribute(aiv, origin, width)
        common_bins = distribute(common, origin, width)
        for index in range(bin_count):
            begin = origin + index * width
            finish = min(begin + width, end)
            aic_us = aic_bins[index]
            aiv_us = aiv_bins[index]
            both_us = common_bins[index]
            rows.append({'rank': rank, 'physical_npu': rank + 4,
                         'offset_ms': (begin - origin) / 1000,
                         'bin_ms': (finish - begin) / 1000,
                         'aic_task_active_ms': aic_us / 1000,
                         'aiv_task_active_ms': aiv_us / 1000,
                         'aic_aiv_overlap_ms': both_us / 1000})
    with (out_dir / 'e4-task-activity-10ms.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    timeline = read_json(timeline_path)
    checkpoint = next(item for item in timeline['checkpoints'] if item['step'] == 28)
    stages = []
    for name, data in checkpoint['stages'].items():
        elapsed = data.get('elapsed_seconds', {})
        stages.append({'step': 28, 'stage': name, 'intervals': data.get('intervals'),
                       'union_seconds': data.get('union_seconds'),
                       'cumulative_seconds': data.get('cumulative_seconds'),
                       'median_seconds': elapsed.get('median'),
                       'p95_seconds': elapsed.get('p95'),
                       'p95_status': elapsed.get('p95_status')})
    with (out_dir / 'e4-save-stage-summary-step28.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stages[0]))
        writer.writeheader()
        writer.writerows(stages)
    return summaries, checkpoint, timeline


def write_continuation(raw_root, out_dir):
    stage_rows = []
    restore_seconds = []
    save_seconds = []
    boundary_seconds = []
    for rank_dir in sorted(raw_root.glob('rank_*')):
        rank = int(rank_dir.name.split('_')[-1])
        training = read_json(rank_dir / 'training.json')['payload']
        boundary = next(item for item in training['boundaries']
                        if item.get('operation') == 'save' and item.get('step') == 56)
        boundary_begin = boundary['begin_ns']
        boundary_seconds.append((boundary['end_ns'] - boundary_begin) / 1e9)
        for stage in boundary['stages']:
            stage_rows.append({
                'rank': rank,
                'stage': stage['name'],
                'offset_from_save_begin_seconds': (stage['begin_ns'] - boundary_begin) / 1e9,
                'duration_seconds': (stage['end_ns'] - stage['begin_ns']) / 1e9,
            })
        for path in sorted(rank_dir.glob('d2-operation-*.json')):
            operation = read_json(path)
            if operation['operation'] == 'restore':
                restore_seconds.append(operation['seconds'])
            elif operation['operation'] == 'save':
                save_seconds.append(operation['seconds'])
    with (out_dir / 'continuation-save-stage-step56.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)
    return {
        'restore_operation_seconds_by_rank': restore_seconds,
        'restore_operation_median_seconds': percentile(restore_seconds, .5),
        'save_operation_seconds_by_rank': save_seconds,
        'save_operation_median_seconds': percentile(save_seconds, .5),
        'save_boundary_seconds_by_rank': boundary_seconds,
        'save_stage_rows': len(stage_rows),
        'restore_stage_detail': 'operation durations here; chunk/readiness events analyzed by checkpoint_io_restore_events.py',
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-result', type=Path, required=True)
    parser.add_argument('--control-db', type=Path, required=True)
    parser.add_argument('--e4-root', type=Path, required=True)
    parser.add_argument('--e4-timeline', type=Path, required=True)
    parser.add_argument('--continuation-raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    control_rows, control_series = write_control(args.control_result, args.control_db, args.output)
    e4_activity, checkpoint, timeline = write_e4_activity(
        args.e4_root, args.e4_timeline, args.output)
    continuation = write_continuation(args.continuation_raw, args.output)
    e4_profiles = sorted(args.e4_root.glob('rank_*/device-profile/**/kernel_details.csv.gz'))
    continuation_inputs = sorted(args.continuation_raw.glob('rank_*/training.json'))
    continuation_inputs += sorted(args.continuation_raw.glob('rank_*/d2-operation-*.json'))
    continuation_inputs.append(args.continuation_raw / 'owner' / 'result.json')
    conn = sqlite3.connect(args.control_db)
    metric_ids = {value: str(key) for key, value in conn.execute('SELECT id,value FROM STRING_IDS')}
    conn.close()
    summary = {
        'control_status': 'pass',
        'control_windows': control_rows,
        'control_timeseries_10ms_rows': len(control_series),
        'control_core_type_mapping': {metric_ids[name]: name for name in ('AIC', 'AIV')},
        'control_hbm_type_mapping': {metric_ids[name]: name for name in ('read', 'write')},
        'e4_profiler_scope': 'rank-local task intervals, save-focused steps 28-32',
        'e4_task_activity_by_rank': e4_activity,
        'e4_checkpoint_step28': {
            'capture_to_resume_seconds': checkpoint['capture_to_resume_seconds'],
            'full_boundary_by_rank': [x for x in timeline['full_boundaries']
                                      if x['step'] == 28],
            'stage_overlap_note': timeline['interpretation'],
        },
        'continuation_step52_to_56': continuation,
        'input_sha256': {
            'control_result': sha256(args.control_result),
            'control_db': sha256(args.control_db),
            'e4_timeline': sha256(args.e4_timeline),
            'e4_kernel_details': {
                str(path.relative_to(args.e4_root)): sha256(path) for path in e4_profiles
            },
            'continuation_records': {
                str(path.relative_to(args.continuation_raw)): sha256(path)
                for path in continuation_inputs
            },
        },
    }
    (args.output / 'analysis-summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
