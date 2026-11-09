"""Diagnostic real-model O1 optimizer task fragment, separate from PMU runs."""
import argparse
import csv
from decimal import Decimal
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

from .training_jit_campaign import BASE,PYTHON,RAW,RESULTS,archive_failure
from .reduce import compress


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();name='o1-seq128-optimizer-task-diagnostic'
    raw=RAW/name;out=RESULTS/'model-fragment';out.mkdir(exist_ok=True)
    if raw.exists():raise FileExistsError('refusing to overwrite real-model diagnostic')
    if Path('/models/npu_nvme_exp/user7-stack/hardware-campaign.lease.json').exists():
        raise RuntimeError('training lease still active')
    smi=subprocess.check_output(['npu-smi','info'],text=True)
    if not all(f'No running processes found in NPU {d}' in smi for d in (4,5,6,7)):
        raise RuntimeError('experiment cards occupied')
    config=json.loads(BASE.read_text());config.update(seq_length=128,micro_batch=1,jit_level='O1',stop_step=40,lr_horizon=128,
        activity_profile=dict(start_step=18,stop_step=22,aic_metrics='none',profiler_level=0,activities='cpu_npu'))
    path=out/'run-config.json';path.write_text(json.dumps(config,indent=2)+'\n')
    command=[PYTHON,'train.py','fit','--config',str(path),'--output',str(raw)]
    (out/'command.json').write_text(json.dumps(dict(command=command,cwd=str(args.source),
        scope='diagnostic task identity and stage only; excluded from normal-path performance matrix'),indent=2)+'\n')
    env=dict(os.environ,PYTHONPATH=str(args.source/'python')+':'+str(args.source),PYTHONDONTWRITEBYTECODE='1')
    with (out/'supervisor.log').open('x') as log:
        result=subprocess.run(command,cwd=args.source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    if result.returncode:
        archive_failure(raw,out/'failure')
        raise RuntimeError('real-model diagnostic failed: '+str(result.returncode))
    for name in ('result.json','source-identity.json','preflight.json','training-command.json'):
        shutil.copy2(raw/name,out/name)
    ranks=[]
    for rank in range(4):
        target=out/f'rank{rank}';target.mkdir(exist_ok=True)
        for name in ('training.json','resolved-config.json','state-initial.json','state-step-40.json'):
            shutil.copy2(raw/f'rank_{rank}'/name,target/name)
        payload=json.loads((target/'training.json').read_text())['payload']
        losses=payload['losses'];control=[v for v in losses if 11<=v['step']<=16]
        profiled=[v for v in losses if 19<=v['step']<=21]
        timing=lambda values:statistics.median((v['end_ns']-v['begin_ns'])/1e6 for v in values)
        kernels=list((raw/f'rank_{rank}/activity-profiler').glob('**/kernel_details.csv'))
        native=list((raw/f'rank_{rank}/activity-profiler').glob('**/ascend_task.db'))
        if len(kernels)!=1 or len(native)!=1:raise ValueError('ambiguous model diagnostic profiler inputs')
        compress(kernels[0],target/'kernel_details.csv.gz')
        compress(native[0],target/'ascend_task.db.gz')
        device=native[0].parent.parent
        for pattern in ('start_info.*','info.json.*','host_start.log.*','dev_start.log.*','sample.json'):
            for anchor in device.glob(pattern):
                if not anchor.name.endswith('.done'):shutil.copy2(anchor,target/anchor.name)
        timelines=list((raw/f'rank_{rank}/activity-profiler').glob('**/task_time*.csv'))
        if len(timelines)!=1:raise ValueError('ambiguous native model task timeline')
        compress(timelines[0],target/'task_time.csv.gz')
        rows=list(csv.DictReader(kernels[0].open()))
        tasks=[]
        for row in rows:
            start=int(Decimal(row['Start Time(us)'].strip())*1000)
            duration=int(Decimal(row['Duration(us)'].strip())*1000)
            tasks.append(dict(name=row['Name'],start_ns=start,duration_ns=duration,
                              engine=row['Accelerator Core'],stream=row['Stream ID'],task_id=row['Task ID']))
        with gzip.open(target/'tasks.jsonl.gz','wt') as stream:
            for task in tasks:stream.write(json.dumps(task,separators=(',',':'))+'\n')
        ranks.append(dict(rank=rank,physical_device=rank+4,kernel_count=len(tasks),
            native_db_sha256=hashlib.sha256(native[0].read_bytes()).hexdigest(),
            control_ms=timing(control),diagnostic_ms=timing(profiled),
            overhead_fraction=timing(profiled)/timing(control)-1,
            windows=[dict(step=v['step'],begin_ns=v['wall_begin_ns'],end_ns=v['wall_end_ns']) for v in profiled],
            interpretation='diagnostic per-task run; no normal critical-path duration or co-run readiness proof'))
    (out/'diagnostic-evidence.json').write_text(json.dumps(dict(status='pass',
        normal_performance_accepted=False,ranks=ranks),indent=2)+'\n')
    print('archived real-model diagnostic',flush=True)


if __name__=='__main__':main()
