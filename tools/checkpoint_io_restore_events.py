#!/usr/bin/env python3
"""Extract recorded restore stages without confusing rank waits with DMA time."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.source / 'python'))
    from npu_nvme.experiments.checkpoint_io_events import union_ns
    pairs = {
        'owner_chunk_read_verify_send_ack': ('owner_restore_chunk_begin', 'owner_restore_chunk_end'),
        'rank_chunk_apply': ('restore_apply_begin', 'restore_chunk_applied'),
        'rank_apply_until_copy_safe': ('restore_apply_begin', 'restore_copy_safe'),
        'rank_receive_until_applied': ('restore_chunk_received', 'restore_chunk_applied'),
    }
    events = []
    hashes = {}
    for path in sorted((a.raw / 'io-events').glob('*.jsonl')):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for line in stream:
                h.update(line)
                row = json.loads(line)
                if row['step'] == a.step:
                    if row['clock_domain'] != 'CLOCK_MONOTONIC':
                        raise ValueError('unexpected event clock')
                    events.append(row)
        hashes[path.name] = h.hexdigest()
    owner_begin = [r['timestamp_ns'] for r in events if r['event'] == 'owner_restore_begin']
    owner_ready = [r['timestamp_ns'] for r in events if r['event'] == 'owner_restore_ready']
    if len(owner_begin) != 1 or len(owner_ready) != 1:
        raise ValueError('one restore owner begin/ready pair required')
    origin, end = owner_begin[0], owner_ready[0]
    rows = []
    for stage, (begin_name, end_name) in pairs.items():
        begins, ends = {}, {}
        for row in events:
            key = (row['rank'], row['raw_request_id'], row['chunk'])
            if row['event'] == begin_name:
                if key in begins:
                    raise ValueError('duplicate restore begin')
                begins[key] = row['timestamp_ns']
            if row['event'] == end_name:
                if key in ends:
                    raise ValueError('duplicate restore end')
                ends[key] = row['timestamp_ns']
        if not begins or begins.keys() != ends.keys():
            raise ValueError('restore event pair coverage differs: ' + stage)
        grouped = defaultdict(list)
        for key, begin in begins.items():
            finish = ends[key]
            if finish < begin:
                raise ValueError('reversed restore interval')
            grouped[key[0]].append((begin, finish))
        for rank, intervals in sorted(grouped.items()):
            values = sorted((b-s)/1e6 for s, b in intervals)
            rows.append(dict(stage=stage, rank=rank, chunks=len(values),
                             begin_offset_seconds=(min(s for s, _ in intervals)-origin)/1e9,
                             end_offset_seconds=(max(b for _, b in intervals)-origin)/1e9,
                             span_seconds=(max(b for _, b in intervals)-min(s for s, _ in intervals))/1e9,
                             union_seconds=union_ns(intervals)/1e9,
                             cumulative_seconds=sum(b-s for s, b in intervals)/1e9,
                             median_chunk_ms=values[len(values)//2],
                             p99_chunk_ms=values[(len(values)*99+99)//100-1]))
    ranks = {}
    markers = ('restore_chunk_received', 'restore_chunk_applied', 'restore_controls_verified',
               'restore_local_ready', 'restore_collective_ready')
    for rank in range(4):
        r = [row for row in events if row['rank'] == rank]
        ranks[rank] = {}
        for name in markers:
            times = [row['timestamp_ns'] for row in r if row['event'] == name]
            if not times:
                raise ValueError(f'missing rank {rank} marker: {name}')
            ranks[rank][name] = dict(count=len(times), first_offset_seconds=(min(times)-origin)/1e9,
                                     last_offset_seconds=(max(times)-origin)/1e9)
    all_begins = [r['timestamp_ns'] for r in events if r['event'] == 'owner_restore_chunk_begin']
    all_ends = [r['timestamp_ns'] for r in events if r['event'] == 'owner_restore_chunk_end']
    summary = dict(step=a.step, owner_begin_ns=origin, owner_ready_ns=end,
                   owner_total_seconds=(end-origin)/1e9,
                   setup_until_first_chunk_seconds=(min(all_begins)-origin)/1e9,
                   chunk_stream_span_seconds=(max(all_ends)-min(all_begins))/1e9,
                   after_last_chunk_until_global_ready_seconds=(end-max(all_ends))/1e9,
                   stages=rows, rank_markers=ranks, input_sha256=hashes,
                   scope='host monotonic event intervals; stages overlap, include host/IPC/verification; not device service time')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'restore-event-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (a.output / 'restore-event-stages.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rank_markers', 'input_sha256', 'stages')}))


if __name__ == '__main__':
    main()
