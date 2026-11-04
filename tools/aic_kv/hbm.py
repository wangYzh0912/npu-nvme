"""CANN 9.1 HBMOriginalData interval counts; four channels, read and write."""
import csv
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import sqlite3


def read(device):
    device=Path(device)
    # The pinned HbmModel uses counts * (256/8) bytes and does NOT delta counts.
    # decode_hbm_v1 has already mapped timestamp to absolute sampling time.
    path=device/'sqlite/hbm.db'
    c=sqlite3.connect('file:'+str(path)+'?immutable=1',uri=True)
    rows=c.execute('select device_id,timestamp,counts,event_type,hbmid from HBMOriginalData where replayid=0 order by timestamp,event_type,hbmid').fetchall()
    c.close()
    groups=defaultdict(dict)
    for dev,ts,counts,event,channel in rows:
        key=(int(dev),int(Decimal(str(ts))))
        entry=(int(channel),event)
        if entry in groups[key]:raise ValueError('duplicate HBM channel/event sample')
        if event not in ('read','write') or not 0<=channel<4 or counts<0:
            raise ValueError('unexpected HBM hardware counter geometry')
        groups[key][entry]=int(counts)
    previous={};output=[]
    expected={(channel,event) for channel in range(4) for event in ('read','write')}
    for (dev,ts),counts in sorted(groups.items()):
        last=previous.get(dev);previous[dev]=ts
        if last is None:continue
        dt=ts-last
        valid=set(counts)==expected and 15_000_000<=dt<=25_000_000
        output.append(dict(device_id=dev,start_ns=last,end_ns=ts,valid=valid,
            read_bytes_per_s=sum(n*32 for (ch,event),n in counts.items() if event=='read')*1e9/dt if valid else None,
            write_bytes_per_s=sum(n*32 for (ch,event),n in counts.items() if event=='write')*1e9/dt if valid else None))
    return output


def summarize(samples, windows):
    output=[]
    for w in windows:
        width=w['end_ns']-w['begin_ns'];covered=0;read_bytes=0;write_bytes=0
        for sample in samples:
            if not sample['valid']:continue
            dt=max(0,min(sample['end_ns'],w['end_ns'])-max(sample['start_ns'],w['begin_ns']))
            covered+=dt
            read_bytes+=sample['read_bytes_per_s']*dt/1e9
            write_bytes+=sample['write_bytes_per_s']*dt/1e9
        output.append(dict(id=w.get('id',w.get('kind')),begin_ns=w['begin_ns'],end_ns=w['end_ns'],
            coverage=covered/width,read_bytes=read_bytes,write_bytes=write_bytes,
            read_bytes_per_s=read_bytes/(covered/1e9) if covered else None,
            write_bytes_per_s=write_bytes/(covered/1e9) if covered else None))
    return output


def export(device, output, windows):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    samples=read(device)
    if not samples:raise ValueError('no native HBM samples')
    with (output/'hbm-timeseries.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    result=dict(source_db=str(Path(device)/'sqlite/hbm.db'),counts_are='per_interval',
        bytes_per_count=32,clock='already_absolute_by_CANN_HBM_parser',
        native_interval_ns=20_000_000,expected_channels=4,discarded_first_sample=True,
        valid_fraction=sum(x['valid'] for x in samples)/len(samples),
        interval_boundary_method='time-weighted overlap of native interval means',
        windows=summarize(samples,windows))
    (output/'hbm-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
