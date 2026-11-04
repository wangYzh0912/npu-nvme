import sqlite3

import pytest

from tools.aic_kv.hbm import read
from tools.aic_kv.provenance import Ranges


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


def test_cpu_caller_lookup_keeps_nested_ranges_and_pid_namespaces():
    events=[dict(pid=1,tid=2,ph='X',cat='cpu_op',name='outer',ts=0,dur=10),
            dict(pid=1,tid=2,ph='X',cat='cpu_op',name='inner',ts=2,dur=4),
            dict(pid=3,tid=2,ph='X',cat='cpu_op',name='other-process',ts=0,dur=10)]
    index=Ranges(events)
    assert {r['name'] for r in index.containing(dict(pid=1,tid=2,ts=3))}=={'outer','inner'}
    assert {r['name'] for r in index.containing(dict(pid=1,tid=2,ts=6))}=={'outer'}
    assert index.containing(dict(pid=1,tid=2,ts=10))==[]
