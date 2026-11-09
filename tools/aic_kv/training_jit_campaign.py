"""Run isolated MindSpore JIT and workload-size observations."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

RESULTS=Path('/models/npu_nvme_exp/user7-stack/aic-aiv-training-followup-20261109')
RAW=Path('/dev/shm/user7-aic-aiv-training-followup-20261109')
BASE=Path('/models/npu_nvme_exp/user7-stack/aic-aiv-coordination-20261105/training/dependency-run-config.json')
PYTHON='/models/npu_nvme_exp/user7-stack/conda/bin/python'


def conditions(stage,mode):
    if stage=='o0-hbm':
        return [('O0',128,1,0)]
    if stage=='o0-hbm-confirm':
        return [('O0',128,1,1)]
    if stage in ('o1-smoke','o2-smoke'):
        selected='O1' if stage=='o1-smoke' else 'O2'
        return [(selected,128,1,0)]
    if stage in ('o1-baseline','o2-baseline'):
        selected='O1' if stage=='o1-baseline' else 'O2'
        return [(selected,length,1,rep) for length in (128,512) for rep in range(3)]
    if stage=='prevalence':
        if mode not in ('O1','O2'):
            raise ValueError('prevalence mode must be an accepted official JIT level')
        return [(mode,length,batch,rep) for length in (128,512,2048)
                for batch in (1,2) for rep in range(3)]
    raise ValueError(stage)



def retire_generated(raw,durable,graphs=None):
    """Remove only completed-run compiler products after retaining final IR."""
    raw=Path(raw);durable=Path(durable)
    if json.loads((raw/'result.json').read_text())['validation_status']!='pass':
        raise RuntimeError('cannot retire artifacts before training completion')
    if not (durable/'rank0/pmu/pmu_quality.json').is_file():
        raise RuntimeError('PMU archive must precede retirement')
    targets=[raw/'training/qwen3_ms_converted_weight',raw/'compiler-cache']
    if graphs is not None:
        evidence=json.loads((Path(graphs)/'graph-evidence.json').read_text())
        if len(evidence['ranks'])!=4:raise ValueError('final IR rank coverage is incomplete')
        for row in evidence['ranks']:
            original=Path(row['source'])
            compressed=Path(graphs)/f"rank{row['rank']}"/(original.name+'.gz')
            h=hashlib.sha256()
            with gzip.open(compressed,'rb') as stream:
                for data in iter(lambda:stream.read(4<<20),b''):h.update(data)
            if h.hexdigest()!=row['source_sha256']:
                raise ValueError('compressed final IR does not match original')
        targets.extend(raw/f'rank_{rank}/dependency-graphs' for rank in range(4))
    record=[]
    for target in targets:
        if not target.exists():continue
        record.append(dict(path=str(target.relative_to(raw)),
            bytes=sum(f.stat().st_size for f in target.rglob('*') if f.is_file()),
            reason='regenerable compilation/converted weights; final stream-assigned IR retained'))
        shutil.rmtree(target)
    (durable/'cache-retirement.json').write_text(json.dumps(record,indent=2)+chr(10))


def archive_failure(raw,output):
    output.mkdir(parents=True,exist_ok=True)
    paths=list(raw.glob('*.json'))+list(raw.glob('*.log'))
    paths+=list(raw.glob('msrun-log/*.log'))
    paths+=list(raw.glob('rank_*/training.json'))+list(raw.glob('rank_*/resolved-config.json'))
    for path in paths:
        target=output/path.relative_to(raw);target.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix=='.log':
            with path.open('rb') as src,gzip.open(str(target)+'.gz','wb') as dst:
                shutil.copyfileobj(src,dst)
        else:shutil.copy2(path,target)
    for suffix in ('.supervisor.log','.training.log','.collector.log'):
        path=raw.with_suffix(suffix)
        if path.is_file():
            with path.open('rb') as src,gzip.open(output/(path.name+'.gz'),'wb') as dst:
                shutil.copyfileobj(src,dst)


def run(source,stage,mode, *, retry_lock=False,retry_interrupted=False):
    source=Path(source).resolve()
    if not (source/'python/npu_nvme/workloads/qwen.py').is_file():
        raise ValueError('training source is not a Qwen experiment tree')
    if 'AIC_AIV_SAVE_DEPENDENCY_GRAPHS' not in (source/'python/npu_nvme/workloads/qwen.py').read_text():
        raise ValueError('training source lacks graph observation hook')
    if Path('/models/npu_nvme_exp/user7-stack/hardware-campaign.lease.json').exists():
        raise RuntimeError('cannot snapshot observation code while a training lease is active')
    shutil.copytree(Path(__file__).parents[1]/'aic_activity',source/'tools/aic_activity',
                    dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    RESULTS.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    for jit,length,batch,rep in conditions(stage,mode):
        if shutil.disk_usage('/models').free<2<<30 or shutil.disk_usage('/dev/shm').free<64<<30:
            raise RuntimeError('insufficient experiment storage headroom')
        name=f'{jit.lower()}-seq{length}-mb{batch}-r{rep}'
        completed=[]
        for saved in (RESULTS/'status').glob(name+'*.json'):
            previous=json.loads(saved.read_text())
            if previous.get('status')=='complete':completed.append(previous)
        if len(completed)>1:raise ValueError('multiple complete runs for the same independent replicate')
        if completed:
            previous=completed[0]
            if (previous.get('jit_level')!=jit or previous.get('sequence_length')!=length or
                    previous.get('micro_batch')!=batch or previous.get('replicate')!=rep or
                    not (Path(previous['raw'])/'sample-evidence.json').is_file()):
                raise ValueError('completed run identity or evidence differs')
            print(json.dumps(dict(name=previous['name'],status='reused_completed_run')),flush=True)
            continue
        original=RESULTS/'status'/f'{name}.json'
        if original.exists() and retry_interrupted and json.loads(original.read_text()).get('status')=='interrupted_before_preflight':
            previous=json.loads(original.read_text())
            failed=RAW/name
            if (previous.get('supervisor_exit_code')!=143 or
                (failed/'preflight.json').exists() or (failed/'run-config.json').exists() or
                list(failed.glob('rank_*/training.json'))):
                raise ValueError('interrupted retry requires retained preflight-free SIGTERM evidence')
            name+='-supervisor-retry1'
        elif original.exists() and retry_lock:
            failure=json.loads((RAW/name/'result.json').read_text())
            if (failure.get('error')!="BlockingIOError(11, 'Resource temporarily unavailable')" or
                    list((RAW/name).glob('rank_*/training.json'))):
                raise ValueError('retry-lock only permits pre-model hardware lock rejection')
            name+='-lock-retry1'
        raw=RAW/name
        config_path=RESULTS/'configs'/f'{name}.json'
        status_path=RESULTS/'status'/f'{name}.json'
        if status_path.exists():
            previous=json.loads(status_path.read_text())
            if (previous.get('status')=='complete' and previous.get('jit_level')==jit and
                    previous.get('sequence_length')==length and previous.get('micro_batch')==batch and
                    previous.get('replicate')==rep and (raw/'sample-evidence.json').is_file()):
                print(json.dumps(dict(name=name,status='reused_completed_run')),flush=True)
                continue
            raise FileExistsError(f'existing run is incomplete or has different identity: {name}')
        if raw.exists():
            raise FileExistsError(f'refusing to overwrite experiment output: {name}')
        config=json.loads(BASE.read_text())
        config.update(seq_length=length,micro_batch=batch,jit_level=jit,stop_step=96,lr_horizon=128)
        if jit=='O2':
            # Keep enough optimizer iterations for a 35-second collector even
            # if graph sinking greatly reduces the per-step host overhead.
            # LR is constant without warmup; compare the common first96 steps.
            config.update(stop_step=1024,lr_horizon=1024,checkpoint_interval=100000)
        config_path.parent.mkdir(parents=True,exist_ok=True)
        config_path.write_text(json.dumps(config,indent=2)+'\n')
        command=[PYTHON,'-m','tools.aic_activity.train_sample','--source',str(source),
                 '--config',str(config_path),'--output',str(raw)]
        record=dict(name=name,stage=stage,jit_level=jit,sequence_length=length,
                    micro_batch=batch,replicate=rep,command=command,source=str(source),
                    raw=str(raw),status='running')
        status_path.parent.mkdir(parents=True,exist_ok=True)
        status_path.write_text(json.dumps(record,indent=2)+'\n')
        graph_capture=rep==0 and stage in ('o1-smoke','o1-baseline','o2-smoke','o2-baseline')
        env=dict(os.environ,PYTHONPATH='.',PYTHONDONTWRITEBYTECODE='1')
        if graph_capture:
            env['AIC_AIV_SAVE_DEPENDENCY_GRAPHS']='1'
        log=raw.with_suffix('.supervisor.log')
        try:
            with log.open('x') as stream:
                subprocess.run(command,cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT,
                               check=True,timeout=14400)
            evidence=json.loads((raw/'sample-evidence.json').read_text())
            from .archive_training_dependency import archive
            durable=RESULTS/'training'/'runs'/name
            if durable.exists():raise FileExistsError(f'refusing to overwrite archived run: {name}')
            archive(raw,durable,include_hbm=True)
            if graph_capture:
                graph_output=RESULTS/'training'/'graphs'/name
                graph_command=[sys.executable,'-m','tools.aic_kv.graph_evidence',
                               '--run',str(raw),'--output',str(graph_output)]
                subprocess.run(graph_command,cwd=Path(__file__).parents[2],
                               env=dict(os.environ,PYTHONPATH='.'),check=True,timeout=1800)
            retire_generated(raw,durable,graph_output if graph_capture else None)
            record.update(status='complete',measurement_quality=evidence['measurement_quality'])
        except BaseException as error:
            record.update(status='failed',error=repr(error))
            status_path.write_text(json.dumps(record,indent=2)+'\n')
            archive_failure(raw,RESULTS/'training/failures'/name)
            raise
        status_path.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--stage',choices=('o0-hbm','o0-hbm-confirm','o1-smoke','o1-baseline','o2-smoke','o2-baseline','prevalence'),required=True)
    parser.add_argument('--mode',choices=('O1','O2'),default='O1')
    parser.add_argument('--retry-lock',action='store_true')
    parser.add_argument('--retry-interrupted',action='store_true')
    args=parser.parse_args()
    run(args.source,args.stage,args.mode,retry_lock=args.retry_lock,retry_interrupted=args.retry_interrupted)


if __name__=='__main__':
    main()
