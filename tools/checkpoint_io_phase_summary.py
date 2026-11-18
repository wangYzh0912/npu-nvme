#!/usr/bin/env python3
"""Export actual checkpoint phase PMU/HBM windows, with clock anchors preserved."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captures', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(a.captures.read_text())
    if manifest['status'] != 'pass':
        raise ValueError('both phase collections must finish successfully')
    results = []
    series = []
    for capture in manifest['captures']:
        db = Path(capture['database'])
        conn = sqlite3.connect('file:' + str(db) + '?mode=ro', uri=True)
        conn.execute('PRAGMA temp_store=MEMORY')
        ids = {value: key for key, value in conn.execute('SELECT id,value FROM STRING_IDS')}
        start, end = conn.execute('SELECT startTimeNs,endTimeNs FROM SESSION_TIME_INFO').fetchone()
        anchor_real, anchor_mono = capture['launch_realtime_ns'], capture['launch_monotonic_ns']
        trigger = capture['trigger']['timestamp_ns']
        values = defaultdict(lambda: defaultdict(list))
        stats = dict(phase=capture['phase'], device=4, session_begin_realtime_ns=start,
                     session_end_realtime_ns=end, session_seconds=(end-start)/1e9,
                     launch_realtime_ns=anchor_real, launch_monotonic_ns=anchor_mono,
                     trigger_monotonic_ns=trigger,
                     session_start_after_trigger_seconds=(start-anchor_real+anchor_mono-trigger)/1e9)
        for metric, core in (('aic_usage_pct', ids['AIC']), ('aiv_usage_pct', ids['AIV'])):
            rows = list(conn.execute(
                'SELECT timestampNs,usage FROM SAMPLE_PMU_TIMELINE '
                'WHERE deviceId=4 AND coreType=?', (core,)))
            if not rows:
                raise ValueError('empty engine PMU rows')
            samples = [row[1] for row in rows]
            stats[metric] = dict(mean=sum(samples)/len(samples), maximum=max(samples),
                                 sample_rows=len(rows), positive_rows=sum(x > 0 for x in samples),
                                 distinct_times=len({row[0] for row in rows}))
            for timestamp, value in rows:
                values[(timestamp-start)//10_000_000][metric].append(value)
        for metric, kind in (('card_hbm_read_mib_s', ids['read']), ('card_hbm_write_mib_s', ids['write'])):
            rows = list(conn.execute(
                'SELECT timestampNs,SUM(bandwidth),COUNT(*) FROM HBM '
                'WHERE deviceId=4 AND type=? GROUP BY timestampNs ORDER BY timestampNs', (kind,)))
            if not rows or any(row[2] != 4 for row in rows):
                raise ValueError('expected four HBM stack rows per timestamp')
            samples = [row[1]/1024**2 for row in rows]
            stats[metric] = dict(mean=sum(samples)/len(samples), maximum=max(samples),
                                 sample_times=len(rows),
                                 maximum_interval_ms=max((b[0]-s[0])/1e6
                                                         for s, b in zip(rows, rows[1:])))
            for (timestamp, _, _), value in zip(rows, samples):
                values[(timestamp-start)//10_000_000][metric].append(value)
        for bucket, fields in sorted(values.items()):
            row = dict(phase=capture['phase'], offset_seconds=bucket/100,
                       approximate_monotonic_ns=start+bucket*10_000_000-anchor_real+anchor_mono)
            for metric in ('aic_usage_pct', 'aiv_usage_pct',
                           'card_hbm_read_mib_s', 'card_hbm_write_mib_s'):
                metric_values = fields.get(metric, [])
                row[metric] = sum(metric_values)/len(metric_values) if metric_values else None
            series.append(row)
        stats.update(database=str(db), database_sha256=hashlib.sha256(db.read_bytes()).hexdigest(),
                     scope='20-second partial phase; device-wide HBM; no per-engine bandwidth split',
                     clock_mapping='host launch realtime/monotonic pair; no NTP adjustment measured')
        results.append(stats)
        conn.close()
    with (a.output / 'checkpoint-phase-timeseries-10ms.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(series[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(series)
    (a.output / 'checkpoint-phase-summary.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
