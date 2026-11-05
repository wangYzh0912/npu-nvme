"""Archive completed graph/PMU observations without weight/compiler caches."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from .coordination import ROOT
from .reduce import export_pmu


def archive(run,output):
    output.mkdir(parents=True,exist_ok=True)
    quality=json.loads((run/'sample-evidence.json').read_text())
    if json.loads((run/'result.json').read_text())['validation_status']!='pass':
        raise ValueError('training did not pass')
    for name in ('sample-evidence.json','result.json','run-config.json','source-identity.json',
                 'preflight.json','training-command.json'):
        shutil.copy2(run/name,output/name)
    for rank,row in enumerate(quality['ranks']):
        target=output/f'rank{rank}';target.mkdir(exist_ok=True)
        for name in ('training.json','resolved-config.json','state-initial.json','state-step-96.json','control-initial.json','control-step-96.json'):
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
    print('ARCHIVED',run.name,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();archive(a.run,a.output)

if __name__=='__main__':main()
