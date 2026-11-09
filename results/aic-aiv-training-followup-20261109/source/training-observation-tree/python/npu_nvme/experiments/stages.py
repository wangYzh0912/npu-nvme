"""Pure aggregation of checkpoint stage times and completion boundaries."""
import statistics


def summarise(rank_reports, completions):
    stages=('score_ns','select_ns','capture_ns','checksum_ns','wait_previous_ns')
    stage_values={name:[] for name in stages}
    payload=0
    for report in rank_reports:
        for row in report['incremental']['steps']:
            payload+=row['payload_bytes']
            for name in stages:stage_values[name].append(row.get(name,0))
    starts=[report['incremental']['formal_begin_ns'] for report in rank_reports]
    finishes=[report['incremental']['formal_end_ns'] for report in rank_reports]
    durable=[row['committed_ns'] for row in completions]
    reference=[row['reference_ready_ns'] for row in completions]
    return dict(payload_bytes=payload,training_seconds=(max(finishes)-min(starts))/1e9,
                all_saved_seconds=(max(durable)-min(starts))/1e9 if durable else None,
                all_reference_ready_seconds=(max(reference)-min(starts))/1e9 if reference else None,
                stage_median_ms={name:statistics.median(values)/1e6 if values else None for name,values in stage_values.items()},
                note='Stage execution times are not additive critical-path costs; ablations required.')
