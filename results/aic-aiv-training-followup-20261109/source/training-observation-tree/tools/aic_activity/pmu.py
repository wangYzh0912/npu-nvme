"""Sample-based task cycles / per-engine core capacity. No ratio-CSV inference.

Formula agrees with CANN viewer/get_msvp_summary.py calculate_utilization:
cycles / (core clock * sampling interval). Discard each core's first sample.
"""
import csv
import gzip
import json
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

def read(device):
    device=Path(device)
    info_path=next(p for p in device.glob('info.json.*') if not p.name.endswith('.done'))
    sync_path=next(p for p in device.glob('start_info.*') if not p.name.endswith('.done'))
    info=json.loads(info_path.read_text())['DeviceInfo'][0]
    sync=json.loads(sync_path.read_text())
    # These are the exact per-job clock anchors used by the CANN exporter.
    offset=int(sync['collectionTimeBegin'])*1000-int(sync['clockMonotonicRaw'])
    config=json.loads((device/'sample.json').read_text())
    output={};quality={}
    for engine,filename,nkey,fkey in [('aic','aicore.db','ai_core_num','aic_frequency'),('aiv','ai_vector_core.db','aiv_num','aiv_frequency')]:
        interval=int(config['aicore_sampling_interval' if engine=='aic' else 'aiv_sampling_interval'])*1_000_000
        if config['ai_core_profiling_mode' if engine=='aic' else 'aiv_profiling_mode']!='sample-based':raise ValueError('PMU input is not sample-based')
        frequency=Decimal(str(info[fkey]))/1000 # MHz -> cycles / ns
        c=sqlite3.connect('file:'+str(device/'sqlite'/filename)+'?immutable=1',uri=True)
        cores=defaultdict(list)
        for ts,core,cycles in c.execute('select timestamp,coreid,task_cyc from AICoreOriginalData where replayid=0 order by coreid,timestamp'):
            cores[core].append((int(Decimal(str(ts))),int(cycles)))
        c.close()
        if len(cores)!=int(info[nkey]):raise ValueError(f'{engine}: expected {info[nkey]} cores, observed {len(cores)}')
        samples=[];invalid=defaultdict(int)
        for core,values in cores.items():
            for (previous,_),(ts,cycles) in zip(values,values[1:]):
                dt=ts-previous
                reason=('invalid_interval' if not interval*.75<=dt<=interval*1.25 else
                        'cycles_out_of_range' if cycles<0 or Decimal(cycles)>frequency*dt*Decimal('1.01') else '')
                if reason:invalid[reason]+=1
                samples.append(dict(core=core,start=previous+offset,end=ts+offset,task_cycles=cycles,cycles_per_ns=str(frequency),
                                    invalid_reason=reason,busy_fraction=float(Decimal(cycles)/(frequency*dt)) if dt>0 else None,valid=not reason))
        output[engine]=samples
        quality[engine]=dict(core_count=len(cores),frequency_mhz=str(info[fkey]),native_interval_ns=interval,invalid_intervals=dict(invalid),sample_intervals=len(samples))
    return output,dict(engines=quality,clock_offset_ns=offset,clock_source=str(sync_path))

def bins(samples,begin,end,bin_ns=10_000_000):
    if bin_ns<10_000_000:raise ValueError('PMU display bins must be >=10ms')
    result=[];n=(end-begin)//bin_ns
    axes={}
    for engine,values in samples.items():
        core_count=len({s['core'] for s in values});cap=core_count*bin_ns
        covered=[0.0]*n;busy=[0.0]*n
        for s in values:
            if not s['valid'] or s['end']<=begin or s['start']>=begin+n*bin_ns:continue
            first=max(0,(s['start']-begin)//bin_ns);last=min(n-1,(s['end']-1-begin)//bin_ns)
            for i in range(first,last+1):
                dt=max(0,min(s['end'],begin+(i+1)*bin_ns)-max(s['start'],begin+i*bin_ns))
                covered[i]+=dt;busy[i]+=dt*s['busy_fraction']
        axes[engine]=[(busy[i]/cap*100,covered[i]/cap) for i in range(n)]
    for i in range(n):
        valid=all(axes[e][i][1]>=.95 for e in axes)
        result.append(dict(window_start_ns=begin+i*bin_ns,window_end_ns=begin+(i+1)*bin_ns,
                           aic_activity_pct=axes['aic'][i][0] if valid else None,aiv_activity_pct=axes['aiv'][i][0] if valid else None,
                           aic_coverage=axes['aic'][i][1],aiv_coverage=axes['aiv'][i][1],metric_kind='sampled_core_busy',valid=valid))
    return result

def export(device,out,windows=None):
    samples,quality=read(device);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if windows is None:
        begin=max(min(x['start'] for x in values) for values in samples.values())
        end=min(max(x['end'] for x in values) for values in samples.values())
        windows=[dict(id='collection',begin_ns=begin,end_ns=end)]
    ts=[]
    for w in windows:
        for r in bins(samples,w['begin_ns'],w['end_ns']):r['window_id']=w['id'];ts.append(r)
    quality['valid_bin_fraction']=sum(r['valid'] for r in ts)/len(ts) if ts else 0
    quality['formula']='sum(task_cycles / engine_frequency) / (engine_core_count * window_duration)'
    (out/'pmu_quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    with (out/'core_timeseries.csv').open('w',newline='') as f:
        if ts:w=csv.DictWriter(f,fieldnames=list(ts[0]));w.writeheader();w.writerows(ts)
    with gzip.open(out/'per_core_samples.jsonl.gz','wt') as f:
        for e,values in samples.items():
            for r in values:f.write(json.dumps(dict(r,engine=e),separators=(',',':'))+'\n')
    return ts,quality
