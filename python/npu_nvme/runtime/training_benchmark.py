"""Serial acceptance campaign for periodic and repeated fresh-process restart."""
from pathlib import Path
import statistics

from .training_catalog import read_checked
from .training_validation import compare


def campaign(config,output,*,profile='all',reuse=None):
    from npu_nvme.cli.qwen_training import run_fit,write
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    result=dict(validation_status='running',profile=profile,runs=[],comparisons=[],timings={})
    reusable={}
    if reuse:
        from npu_nvme.cli.qwen_training import preflight,ROOT
        from .training_reuse import verified_runs
        preflight(config)
        reusable,rejected=verified_runs(reuse,config,ROOT)
        result['reuse']=dict(source=str(Path(reuse).resolve()),accepted=sorted(reusable),rejected=rejected)
    def record():write(output/'result.json',result)
    record()
    def execute(name,method,*,stop,horizon,interval,root=None,resume=False,generation='latest'):
        if name in reusable:
            path=reusable[name]
            result['runs'].append(dict(name=name,method=method,path=str(path),reused=True));record()
            return path
        settings=dict(config,method=method,stop_step=stop,lr_horizon=horizon,checkpoint_interval=interval)
        settings['identity']=dict(config['identity'],lr_horizon=horizon)
        settings.pop('checkpoint_root',None)
        if root:settings['checkpoint_root']=str(root)
        path=output/name
        if run_fit(settings,path,resume=resume,restore=generation):
            raise RuntimeError('campaign run failed: '+name)
        from .payload_dedup import deduplicate_completed
        deduplicate_completed(path,output/'.payload-cache')
        result['runs'].append(dict(name=name,method=method,path=str(path)));record()
        return path
    def check(oracle,path):
        row=compare(oracle,path);row.update(oracle=str(oracle),run=str(path))
        result['comparisons'].append(row);record()
    try:
        if profile in ('all','multicycle'):
            oracle=execute('trajectory-none','none',stop=24,horizon=24,interval=4)
            for method in ('mindspore_native_save','ours','bytecheckpoint_host'):
                source=execute(method+'-source8',method,stop=8,horizon=24,interval=4)
                check(oracle,source);root=source/'checkpoints'
                explicit=execute(method+'-explicit4',method,stop=8,horizon=24,interval=24,
                    root=root,resume=True,generation='1')
                check(oracle,explicit)
                middle=execute(method+'-latest8-to16',method,stop=16,horizon=24,interval=4,root=root,resume=True)
                check(oracle,middle)
                final=execute(method+'-latest16-to24',method,stop=24,horizon=24,interval=4,root=root,resume=True)
                check(oracle,final)
        if profile in ('all','repeated'):
            oracles=[execute(f'repeat-none-{rep}','none',stop=11,horizon=11,interval=8) for rep in range(3)]
            for oracle in oracles[1:]:check(oracles[0],oracle)
            for method in ('mindspore_native_save','ours','bytecheckpoint_host'):
                samples=[]
                for rep in range(3):
                    source=execute(f'repeat-{method}-source-{rep}',method,stop=11,horizon=11,interval=8)
                    check(oracles[0],source)
                # The most recent source remains retained during all four restores.
                for restore in range(4):
                    path=execute(f'repeat-{method}-restore-{restore}',method,
                        stop=11,horizon=11,interval=8,root=source/'checkpoints',resume=True)
                    check(oracles[0],path)
                    rows=[read_checked(path/f'rank_{r}/training.json') for r in range(4)]
                    seconds=(max(row['restore_global_ready_ns'] for row in rows)-min(row['begin_ns'] for row in rows))/1e9
                    if restore:samples.append(seconds)
                result['timings'][method]=dict(samples_seconds=samples,median_seconds=statistics.median(samples),
                    source_repetitions=3,warmup_restores=1,timed_restores=3,
                    boundary='worker_entry_to_restore_global_ready',mandatory_integrity=True)
                record()
        result['validation_status']='pass';record();return 0
    except BaseException as error:
        result.update(validation_status='fail',error=repr(error));record();raise
