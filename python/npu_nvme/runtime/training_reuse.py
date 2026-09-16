"""Verify completed multicycle runs before resuming a repaired campaign."""
import ast
import json
from pathlib import Path
import subprocess

from .training_catalog import digest_file,read_checked


def verified_runs(previous,config,repo):
    from tools.run_campaign import source_snapshot
    previous=Path(previous).resolve();repo=Path(repo)
    reusable={};rejected=[];verified_sources={}
    def validate(name,method,start,stop,interval):
        run=previous/name
        result=json.loads((run/'result.json').read_text())
        if result!=dict(execution_status='completed',validation_status='pass',method=method,final_step=stop):
            raise ValueError('run is not complete')
        settings=json.loads((run/'run-config.json').read_text())
        expected=dict(identity=config['identity'],method=method,start_step=start,stop_step=stop,
            lr_horizon=24,checkpoint_interval=interval,seq_length=config['seq_length'],retention=config['retention'])
        if any(settings.get(key)!=value for key,value in expected.items()):
            raise ValueError('run settings or environment identity changed')
        identity=json.loads((run/'source-identity.json').read_text())
        if len(identity['sources'])!=1:raise ValueError('ambiguous source checkout')
        old=Path(next(iter(identity['sources'])))
        key=(str(old),method)
        if key not in verified_sources:
            snapshot=source_snapshot(dict(stages=[dict(cwd=str(old))]))
            if snapshot!=identity['sources']:raise ValueError('previous source checkout changed')
            names=subprocess.check_output(['git','ls-files','-z','python','src','include','tools','scripts',
                'config/qwen_runtime_schema.json','config/d2_qwen_region.json',
                'experiments/baselines/repro/workers'],cwd=old).decode().split('\0')
            for name in filter(None,names):
                if name in ('python/npu_nvme/runtime/training_benchmark.py',
                            'python/npu_nvme/runtime/training_reuse.py'):continue
                if name=='python/npu_nvme/cli/qwen_training.py':
                    def supervisor(path):
                        tree=ast.parse(path.read_text())
                        return [ast.dump(node) for node in tree.body if not
                            (isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='main')]
                    if supervisor(old/name)!=supervisor(repo/name):
                        raise ValueError('training supervisor changed')
                    continue
                if method!='bytecheckpoint_host' and (name=='python/qwen_host_checkpoint.py' or
                        name.startswith('experiments/')):continue
                if not (repo/name).is_file() or (old/name).read_bytes()!=(repo/name).read_bytes():
                    raise ValueError('training implementation changed: '+name)
            for name,checksum in identity['extensions'].items():
                if digest_file(repo/name)!=checksum:raise ValueError('allocation extension changed')
            verified_sources[key]=identity
        if verified_sources[key]!=identity:raise ValueError('mixed source revisions within method')
        for rank in range(4):
            row=read_checked(run/f'rank_{rank}/training.json')
            if row['status']!='pass' or row['final_step']!=stop:
                raise ValueError('rank completion differs')
        if method=='ours':
            owner=json.loads((run/'owner/result.json').read_text())
            if owner['status']!='pass' or not owner['closed']:raise ValueError('owner close unverified')
        return run
    try:reusable['trajectory-none']=validate('trajectory-none','none',0,24,4)
    except (OSError,ValueError,KeyError) as error:
        rejected.append(dict(method='none',reason=str(error)))
        return reusable,rejected
    for method in ('mindspore_native_save','ours','bytecheckpoint_host'):
        group={}
        try:
            for suffix,start,stop,interval in [('source8',0,8,4),('explicit4',4,8,24),
                    ('latest8-to16',8,16,4),('latest16-to24',16,24,4)]:
                name=method+'-'+suffix;group[name]=validate(name,method,start,stop,interval)
        except (OSError,ValueError,KeyError) as error:
            rejected.append(dict(method=method,reason=str(error)))
        else:reusable.update(group)
    return reusable,rejected
