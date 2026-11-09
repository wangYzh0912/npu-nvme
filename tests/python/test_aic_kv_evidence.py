import sqlite3

import pytest

from tools.aic_kv.hbm import read
from tools.aic_kv.provenance import Ranges
from tools.aic_kv.archive_training_dependency import cann83_hbm_clock


def test_hbm_uses_interval_counts_and_keeps_already_absolute_clock(tmp_path):
    path=tmp_path/'sqlite'
    path.mkdir()
    c=sqlite3.connect(path/'hbm.db')
    c.execute('create table HBMOriginalData(device_id, replayid, timestamp, counts, event_type, hbmid)')
    for ts,count in ((1_000_000_000,625),(1_020_000_000,625)):
        for channel in range(4):
            for event in ('read','write'):
                c.execute('insert into HBMOriginalData values(4,0,?,?,?,?)',(ts,count,event,channel))
    c.commit();c.close()
    samples=read(tmp_path)
    assert len(samples)==1
    assert samples[0]['start_ns']==1_000_000_000
    assert samples[0]['end_ns']==1_020_000_000
    assert samples[0]['read_bytes_per_s']==4_000_000
    assert samples[0]['write_bytes_per_s']==4_000_000
    from tools.aic_kv.hbm import summarize
    wall_window=[dict(id='step',begin_ns=121_000_000_000,end_ns=121_020_000_000)]
    assert summarize(samples,wall_window)[0]['coverage']==0
    corrected=read(tmp_path,clock_offset_ns=120_000_000_000)
    assert summarize(corrected,wall_window)[0]['coverage']==1


def test_cpu_caller_lookup_keeps_nested_ranges_and_pid_namespaces():
    events=[dict(pid=1,tid=2,ph='X',cat='cpu_op',name='outer',ts=0,dur=10),
            dict(pid=1,tid=2,ph='X',cat='cpu_op',name='inner',ts=2,dur=4),
            dict(pid=3,tid=2,ph='X',cat='cpu_op',name='other-process',ts=0,dur=10)]
    index=Ranges(events)
    assert {r['name'] for r in index.containing(dict(pid=1,tid=2,ts=3))}=={'outer','inner'}
    assert {r['name'] for r in index.containing(dict(pid=1,tid=2,ts=6))}=={'outer'}
    assert index.containing(dict(pid=1,tid=2,ts=10))==[]


def test_training_hbm_requires_device_to_host_delta_before_wall_offset(tmp_path):
    import json
    (tmp_path/'host_start.log.4').write_text('[Device4]\nclock_monotonic_raw: 30000000000\ncntvct_diff: 100\n')
    (tmp_path/'dev_start.log.4').write_text('clock_monotonic_raw: 10000000000\ncntvct: 1\n')
    (tmp_path/'info.json.4').write_text(json.dumps({'CPU':[{'Frequency':'100'}]}))
    clock=cann83_hbm_clock(tmp_path,100000000000)
    assert clock['device_to_host_delta_ns']==20000001000
    assert clock['offset_ns']==120000001000
