"""Archive completed graph/PMU observations without weight/compiler caches."""
import argparse
import configparser
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from .coordination import ROOT
from .reduce import export_pmu


def cann83_hbm_clock(device,wall_offset):
    """Match CANN 8.3 HBM viewer: raw DB + get_delta_time + wall offset."""
    device=Path(device)
    host=next(p for p in device.glob('host_start.log.*') if not p.name.endswith('.done'))
    dev=next(p for p in device.glob('dev_start.log.*') if not p.name.endswith('.done'))
    info=next(p for p in device.glob('info.json.*') if not p.name.endswith('.done'))
    parser=configparser.ConfigParser();parser.read(host)
    row=dict(parser.items(parser.sections()[0]))
    freq=float(json.loads(info.read_text())['CPU'][0]['Frequency'])*1e6
    if freq<=0:raise ValueError('invalid host counter frequency')
    host_mon=int(row['clock_monotonic_raw'])+round(int(row['cntvct_diff'])*1e9/freq)
    fields=dict(line.split(':',1) for line in dev.read_text().splitlines() if ':' in line)
    delta=host_mon-int(fields['clock_monotonic_raw'])
    return dict(offset_ns=wall_offset+delta,device_to_host_delta_ns=delta,
                wall_offset_ns=wall_offset,source='CANN 8.3 HBM viewer get_delta_time()',
                anchors=[str(host),str(dev),str(info)])


def archive(run,output, *, include_hbm=False):
    output.mkdir(parents=True,exist_ok=True)
    quality=json.loads((run/'sample-evidence.json').read_text())
    if json.loads((run/'result.json').read_text())['validation_status']!='pass':
        raise ValueError('training did not pass')
    for name in ('sample-evidence.json','result.json','run-config.json','source-identity.json',
                 'preflight.json','training-command.json'):
        shutil.copy2(run/name,output/name)
    for rank,row in enumerate(quality['ranks']):
        target=output/f'rank{rank}';target.mkdir(exist_ok=True)
        final_step=json.loads((run/'result.json').read_text())['final_step']
        for name in ('training.json','resolved-config.json','state-initial.json',f'state-step-{final_step}.json',
                     'control-initial.json',f'control-step-{final_step}.json'):
            p=run/f'rank_{rank}'/name
            if p.exists():shutil.copy2(p,target/name)
        device=Path(row['collector_device_path'])
        bins,pmu_quality=export_pmu(device,target/'pmu',row['windows'])
        manifests=[]
        for name in ('aicore.db','ai_vector_core.db'):
            source=device/'sqlite'/name
            c=sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True)
            values=c.execute('select timestamp,coreid,task_cyc,replayid from AICoreOriginalData where replayid=0 order by coreid,timestamp')
            dest=target/(name+'.replay0.csv.gz');count=0
            with gzip.open(dest,'wt',newline='') as f:
                writer=csv.writer(f);writer.writerow(['timestamp','coreid','task_cyc','replayid'])
                for value in values:writer.writerow(value);count+=1
            c.close()
            manifests.append(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=count,archive=dest.name))
        for pattern in ('info.json.*','start_info.*','sample.json'):
            for p in device.glob(pattern):
                if not p.name.endswith('.done'):shutil.copy2(p,target/p.name)
        (target/'counter-input-manifest.json').write_text(json.dumps(manifests,indent=2)+'\n')
        if include_hbm:
            from .hbm import export as hbm_export
            source=device/'sqlite/hbm.db'
            with source.open('rb') as f,gzip.open(target/'hbm.db.gz','wb') as g:
                shutil.copyfileobj(f,g)
            clock=cann83_hbm_clock(device,row['pmu_quality']['clock_offset_ns'])
            hbm_export(device,target/'hbm',row['windows'],clock_offset_ns=clock['offset_ns'])
            (target/'hbm-clock.json').write_text(json.dumps(clock,indent=2)+chr(10))
            for anchor in clock['anchors']:
                shutil.copy2(anchor,target/Path(anchor).name)
    print('ARCHIVED',run.name,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();archive(a.run,a.output)

if __name__=='__main__':main()
