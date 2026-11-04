"""Reduce completed real runs to compact per-card, per-request evidence."""
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import sqlite3

from .hbm import export as hbm_export
from .provenance import ns,digest


def compress(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    with Path(source).open('rb') as f,gzip.open(destination,'wb') as g:
        shutil.copyfileobj(f,g)


def archive_counters(device,target):
    """Keep native PMU inputs actually used by the metric, losslessly."""
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    manifests=[]
    for name in ('aicore.db','ai_vector_core.db'):
        source=Path(device)/'sqlite'/name
        conn=sqlite3.connect('file:'+str(source)+'?immutable=1',uri=True)
        rows=conn.execute('select timestamp,coreid,task_cyc,replayid from AICoreOriginalData where replayid=0 order by coreid,timestamp')
        count=0
        destination=target/(name+'.replay0.csv.gz')
        with gzip.open(destination,'wt',newline='') as f:
            writer=csv.writer(f);writer.writerow(['timestamp','coreid','task_cyc','replayid'])
            for row in rows:writer.writerow(row);count+=1
        conn.close()
        with gzip.open(destination,'rt') as f:
            if sum(1 for _ in csv.DictReader(f))!=count:raise ValueError('counter archive row count changed')
        manifests.append(dict(source_db=str(source),source_sha256=digest(source),table='AICoreOriginalData',
            selected_columns=['timestamp','coreid','task_cyc','replayid'],rows=count,
            archive=destination.name,archive_sha256=digest(destination)))
        # Superseded archive is ours; native full DB remains in raw tmpfs.
        old=target/(name+'.gz')
        if old.exists():old.unlink()
    compress(Path(device)/'sqlite/hbm.db',target/'hbm.db.gz')
    (target/'manifest.json').write_text(json.dumps(manifests,indent=2)+'\n')


def add_pmu_work(summary,bins):
    valid=[r for r in bins if r['valid']]
    for engine in ('aic','aiv'):
        summary[engine+'_capacity_busy_ms_per_generate']=sum(
            r[engine+'_activity_pct']/100*(r['window_end_ns']-r['window_start_ns'])/1e6
            for r in valid)/summary['system_request_count']
    summary['pmu_observed_scope_fraction']=sum(r['window_end_ns']-r['window_start_ns'] for r in valid)/1e9/summary['system_elapsed_s']


def export_pmu(device,output,windows):
    # Reuse the validated reader and bin integrator. Native CSV columns already
    # retain every per-core counter; duplicating a large JSONL adds no evidence.
    sys.path.insert(0,'/models/npu_nvme_exp/user7-stack/aic-aiv-v2/source/tools')
    from aic_activity import pmu
    samples,quality=pmu.read(device)
    rows=[]
    for window in windows:
        for row in pmu.bins(samples,window['begin_ns'],window['end_ns']):
            row['window_id']=window['id'];rows.append(row)
    quality['valid_bin_fraction']=sum(r['valid'] for r in rows)/len(rows) if rows else 0
    quality['formula']='sum(task_cycles / engine_frequency) / (engine_core_count * window_duration)'
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'pmu_quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    with (output/'core_timeseries.csv').open('w',newline='') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows,quality


def process(case,output):
    case,output=Path(case),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    result=json.loads((case/'result.json').read_text())
    windows=json.loads((case/'windows.json').read_text())
    task_windows=[w for w in windows if w['kind']=='task']
    system_windows=[w for w in windows if w['kind']=='system']
    layouts=json.loads((case/'kv-layout.json').read_text())
    for filename in ('result.json','kv-layout.json','windows.json'):
        shutil.copy2(case/filename,output/filename)
    for filename in ('source-identity.json','completed.json','run-config.json'):
        source=case.parent/filename
        if source.exists():shutil.copy2(source,output.parent/filename)
    for filename in ('collector-command.json','collector.log'):
        source=case/filename
        if source.exists():shutil.copy2(source,output/filename)
    all_rows=[]
    for trace in sorted((case/'trace').glob('rank*_*/ASCEND_PROFILER_OUTPUT')):
        kernel=trace/'kernel_details.csv'
        with kernel.open() as f:tasks=list(csv.DictReader(f))
        device=int(tasks[0]['Device_id']);rank=device-4
        target=output/f'rank{rank}';target.mkdir(exist_ok=True)
        compress(kernel,target/'kernel_details.csv.gz')
        snapshot=next(x for x in layouts if x['rank']==rank)
        layers=snapshot['layers']
        if len(layers)!=36 or any(l['key']['shape'][0]!=snapshot['num_blocks'] for l in layers):
            raise ValueError('actual cache geometry does not match measured model/allocator')
        request_rows=[]
        for w in task_windows:
            selected=[t for t in tasks if ns(t['Start Time(us)'])<w['end_ns'] and ns(t['Start Time(us)'])+ns(t['Duration(us)'])>w['begin_ns']]
            elapsed=lambda t:max(0,min(w['end_ns'],ns(t['Start Time(us)'])+ns(t['Duration(us)']))-max(w['begin_ns'],ns(t['Start Time(us)'])))
            slices=[t for t in selected if t['Name']=='aclnnContiguous_SliceAiCore_Slice']
            shapes={t['Output Shapes'].strip('"') for t in slices}
            logical_bytes=sum(math.prod(int(x) for x in t['Output Shapes'].strip('"').split(','))*2 for t in slices)
            aiv=sum(elapsed(t) for t in selected if t['Accelerator Core']=='AI_VECTOR_CORE')
            slice_ns=sum(elapsed(t) for t in slices)
            request_rows.append(dict(index=w['index'],slice_count=len(slices),slice_time_ms=slice_ns/1e6,
                aiv_task_time_ms=aiv/1e6,slice_fraction_of_aiv_task_time=slice_ns/aiv if aiv else None,
                slice_logical_output_bytes=logical_bytes,slice_output_shapes=sorted(shapes),
                request_time_ms=w['duration_ns']/1e6))
        summary=dict(run=case.parent.name,case=case.name,device=device,rank=rank,
            measurement_unit='one_generate_call',batch=result['batch'],
            prompt_tokens=result['prompt'],generated_tokens_per_request=result['generated'],
            actual_blocks=snapshot['num_blocks'],requests=request_rows,
            slice_count_per_request=statistics.mean(r['slice_count'] for r in request_rows),
            slice_ms_per_request=statistics.mean(r['slice_time_ms'] for r in request_rows),
            slice_logical_output_bytes_per_request=statistics.mean(r['slice_logical_output_bytes'] for r in request_rows),
            slice_fraction_of_aiv_task_time=statistics.mean(r['slice_fraction_of_aiv_task_time'] for r in request_rows if r['slice_fraction_of_aiv_task_time'] is not None),
            control_median_ms=result['control_median_ms'],control_cv=result['control_cv'],
            task_quality=result['task_quality'],task_overhead_fraction=result['task_overhead_fraction'],
            output_sha256=result['output_sha256'],kernel_sha256=digest(kernel))
        if system_windows:
            devs=list((case/'system').glob(f'PROF_*/device_{device}'))
            if len(devs)!=1:raise ValueError('expected one simultaneous HBM/PMU capture per device')
            dev=devs[0]
            scope=[dict(id='steady',begin_ns=system_windows[0]['begin_ns'],end_ns=system_windows[-1]['end_ns'])]
            hbm=hbm_export(dev,target,scope)
            bins,quality=export_pmu(dev,target/'pmu',scope)
            valid=[r for r in bins if r['valid']]
            summary['pmu_valid_fraction']=quality['valid_bin_fraction']
            summary['aic_mean_pct']=statistics.mean(r['aic_activity_pct'] for r in valid) if valid else None
            summary['aiv_mean_pct']=statistics.mean(r['aiv_activity_pct'] for r in valid) if valid else None
            summary['hbm']=hbm['windows'][0]
            summary['hbm_valid_fraction']=hbm['valid_fraction']
            summary['system_request_count']=len(system_windows)
            summary['system_elapsed_s']=(scope[0]['end_ns']-scope[0]['begin_ns'])/1e9
            add_pmu_work(summary,bins)
            system_median=statistics.median(w['duration_ns'] for w in system_windows)/1e6
            summary['system_median_ms']=system_median
            summary['system_overhead_fraction']=system_median/result['control_median_ms']-1
            summary['hbm_read_bytes_per_request']=hbm['windows'][0]['read_bytes']/len(system_windows)
            summary['hbm_write_bytes_per_request']=hbm['windows'][0]['write_bytes']/len(system_windows)
            archive_counters(dev,target/'counter-inputs')
            for pattern in ('info.json.*','start_info.*','sample.json'):
                for source in dev.glob(pattern):
                    if not source.name.endswith('.done'):
                        compress(source,target/'counter-inputs'/(source.name+'.gz'))
        (target/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        all_rows.append(summary)
    if len(all_rows)!=4:raise ValueError(f'expected four rank task exports for {case}')
    (output/'summary.json').write_text(json.dumps(all_rows,indent=2)+'\n')
    return all_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    all_rows=[]
    for run in sorted(a.raw.glob('capacity-*'))+sorted(a.raw.glob('work-*')):
        # Active container-created directories are intentionally private. A host
        # export log exists only after the run has closed and made data readable.
        export_log=a.raw/(run.name+'-export.log')
        if not run.is_dir() or not export_log.is_file():continue
        if not (run/'completed.json').exists():continue
        cases=sorted(run.glob('b*-l*-o*'))
        if export_log.read_text().count('All profiling data parsed') < 4*len(cases):continue
        for case in cases:
            target=a.output/run.name/case.name
            if (target/'summary.json').exists():rows=json.loads((target/'summary.json').read_text())
            else:
                if len(list(case.glob('trace/rank*_*/ASCEND_PROFILER_OUTPUT/kernel_details.csv')))!=4:continue
                rows=process(case,target)
            all_rows.extend(rows)
            print('REDUCED',run.name,case.name,flush=True)
    (a.output/'all-runs.json').write_text(json.dumps(all_rows,indent=2)+'\n')


if __name__=='__main__':main()
