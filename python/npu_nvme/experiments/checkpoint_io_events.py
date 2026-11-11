"""Correlate host events without treating overlapping intervals as additive."""
import gzip
import json
from pathlib import Path

from .checkpoint_io_analysis import distribution
from .checkpoint_io import FIELDS

PAIRS={
    'source_prepare_and_copy':('source_read_begin','data_ready',('rank','chunk')),
    'rank_hash_before_send':('data_ready','chunk_send_begin',('rank','chunk')),
    'sink_hash_and_consume':('data_ready','sink_consumed_source_safe',('rank','chunk')),
    'bridge_ack_wait':('chunk_send_begin','source_safe_ack',('rank','chunk')),
    'owner_validation':('owner_received','owner_validated',('rank','chunk')),
    'owner_lock_wait':('owner_validated','owner_payload_begin',('rank','chunk')),
    'owner_payload':('owner_payload_begin','owner_payload_end',('rank','chunk')),
    'native_api':('native_submitted','native_completed',('native_request_id',)),
    'payload_flush':('payload_flush_begin','payload_flush_end',()),
    'metadata_write_and_flush':('metadata_begin','metadata_flush_end',()),
    'anchor_write_and_flush':('metadata_flush_end','anchor_flush_end',()),
    'catalog_publication':('catalog_publish_begin','catalog_published',('rank',)),
    'rank_capture':('capture_begin','capture_end',('rank',)),
}


def union_ns(intervals):
    total=0;start=end=None
    for left,right in sorted(intervals):
        if right<left:raise ValueError('event interval reversed')
        if end is None:start,end=left,right
        elif left<=end:end=max(end,right)
        else:total+=end-start;start,end=left,right
    return total+(end-start if end is not None else 0)


def read_events(output):
    output=Path(output)
    paths=sorted((output/'io-events').glob('*.jsonl*'))
    if not paths:raise ValueError('diagnostic events missing: '+str(output))
    rows=[]
    for path in paths:
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt') as stream:
            for line in stream:
                row=json.loads(line)
                if (set(row)!=set(FIELDS) or type(row['timestamp_ns']) is not int or
                        row['timestamp_ns']<=0 or
                        (row['rank'] is not None and (type(row['rank']) is not int or row['rank'] not in range(4)))):
                    raise ValueError('invalid diagnostic event identity/timestamp')
                if row['clock_domain']!='CLOCK_MONOTONIC':
                    raise ValueError('unanchored event clock')
                row['role']='owner' if path.name.startswith('owner.') else 'rank'
                rows.append(row)
    return rows


def timeline(events,physical_npus=(4,5,6,7)):
    steps=sorted({row['step'] for row in events if row['event']=='capture_begin'})
    if not steps:raise ValueError('no checkpoint capture events')
    mapping_errors=[dict(rank=row['rank'],physical_npu=row['physical_npu'],event=row['event'])
        for row in events if row['rank'] is not None and
        row['physical_npu']!=physical_npus[row['rank']]]
    result=[]
    for step in steps:
        rows=[row for row in events if row['step']==step]
        groups={};missing=[]
        for stage,(begin,end,keys) in PAIRS.items():
            if stage=='rank_hash_before_send' and not any(row['event']=='chunk_send_begin' for row in rows):continue
            if stage=='sink_hash_and_consume' and not any(row['event']=='sink_consumed_source_safe' for row in rows):continue
            starts={};ends={}
            for row in rows:
                if row['event'] not in (begin,end):continue
                key=(row['role'],row['raw_request_id'],*(row[field] for field in keys))
                target=starts if row['event']==begin else ends
                if key in target:raise ValueError(f'duplicate {row["event"]} correlation key')
                target[key]=row['timestamp_ns']
            for key in starts.keys()^ends.keys():missing.append(dict(stage=stage,key=key))
            intervals=[(starts[key],ends[key]) for key in starts.keys()&ends.keys()]
            if intervals:
                groups[stage]=dict(intervals=len(intervals),union_seconds=union_ns(intervals)/1e9,
                    sample_unit='correlated intervals within one checkpoint; not independent process repetitions',
                    cumulative_seconds=sum(right-left for left,right in intervals)/1e9,
                    elapsed_seconds=distribution([(right-left)/1e9 for left,right in intervals]))
        begin=[row['timestamp_ns'] for row in rows if row['event']=='capture_begin']
        resume=[row['timestamp_ns'] for row in rows if row['event']=='training_resume']
        ranks=sorted({row['rank'] for row in rows if row['event']=='capture_begin'})
        if ranks!=list(range(4)):missing.append(dict(stage='capture_ranks',observed=ranks))
        committed=[row for row in rows if row['event']=='raw_commit_receipt']
        result.append(dict(step=step,raw_generations=[row['generation'] for row in committed],
            raw_request_ids=sorted({row['raw_request_id'] for row in rows if row['raw_request_id']}),
            capture_to_resume_seconds=(max(resume)-min(begin))/1e9 if len(resume)==4 else None,
            stages=groups,missing_pairs=missing))
    return dict(status='correlated' if not mapping_errors and all(not row['missing_pairs'] for row in result)
                else 'coverage_incomplete',checkpoints=result,
                physical_mapping_error_count=len(mapping_errors),
                physical_mapping_error_examples=mapping_errors[:8],
                clock='host CLOCK_MONOTONIC only',
                interpretation='Stage unions overlap and must not be summed into an end-to-end time. Native API intervals include queueing and host overhead; device-only service is unknown.')
