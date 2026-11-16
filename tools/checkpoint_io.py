"""Run the registered checkpoint-I/O M0/M1 experiment queue."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
sys.path.insert(0,str(ROOT))

from npu_nvme.cli import qwen_training
from npu_nvme.runtime import training_catalog as catalog
from npu_nvme.runtime.training_validation import compare
from npu_nvme.cli.contracts import source_identity
from npu_nvme.experiments.checkpoint_io_analysis import analyse,probe_overhead
from npu_nvme.experiments.checkpoint_io_events import read_events,timeline

DEFAULT_RAW=Path('/dev/shm/user7-aic-aiv-io-20261110')
DEFAULT_RESULTS=Path('/models/npu_nvme_exp/user7-stack/aic-aiv-io-20261110')
ENVIRONMENT='/models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/environments.json'


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def admission(config):
    if Path(config['hardware_lock']).with_suffix('.lease.json').exists():
        raise RuntimeError('hardware lease is active')
    smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=30)
    occupied=[device for device in config['npu_ids']
              if f'No running processes found in NPU {device}' not in smi]
    if occupied:raise RuntimeError(f'physical devices occupied: {occupied}')
    if shutil.disk_usage('/dev/shm').free < 64<<30:
        raise OSError('tmpfs headroom below 64 GiB')
    if shutil.disk_usage(DEFAULT_RESULTS.parent).free < 2<<30:
        raise OSError('durable evidence headroom below 2 GiB')
    return dict(npu_info=smi,output_tmpfs=str(DEFAULT_RAW),durable_output=str(DEFAULT_RESULTS),
                tmpfs_free=shutil.disk_usage('/dev/shm').free,
                durable_free=shutil.disk_usage(DEFAULT_RESULTS.parent).free)


def base_config(config_path):
    value=json.loads(Path(config_path).read_text())
    value.update(environment_manifest=ENVIRONMENT,npu_ids=[4,5,6,7],seq_length=128,
                 checkpoint_interval=4,retention=3,lr_horizon=128,stop_step=64)
    return qwen_training.resolve(value,ROOT)


def serializable_config(config):
    return {key:value for key,value in config.items() if key!='identity'}


def command(config_path,output,*,resume=False,generation='latest',stop=64):
    argv=[sys.executable,str(ROOT/'scripts/run_user_environment.py'),
          '--manifest',ENVIRONMENT,'--profile','candidate','--','python','train.py','fit',
          '--config',str(config_path),'--output',str(output),'--stop-step',str(stop)]
    if resume:argv.extend(['--resume','--generation',str(generation)])
    log=Path(output).with_suffix('.supervisor.log')
    with log.open('x') as stream:
        result=subprocess.run(argv,cwd=ROOT,env=dict(os.environ,
            PYTHONPATH='python:.',ASCEND_RT_VISIBLE_DEVICES='4,5,6,7',
            PYTHONUNBUFFERED='1'),stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f"run failed ({result.returncode}); inspect {log}")
    row=json.loads((Path(output)/'result.json').read_text())
    if row.get('validation_status')!='pass':raise RuntimeError(f'run did not pass: {row}')


def archive_run(output,archive,*,exclude=()):
    output=Path(output);archive=Path(archive);archive.mkdir(parents=True,exist_ok=False)
    included=lambda path:not any(path.is_relative_to(output/prefix) for prefix in exclude)
    paths=[source for source in output.rglob('*.json') if source.is_file() and included(source)]
    for source in paths:
        target=archive/source.relative_to(output);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    for source in list(output.rglob('*.jsonl'))+list(output.rglob('*.log')):
        if not included(source):continue
        target=archive/source.relative_to(output).with_suffix(source.suffix+'.gz')
        target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as stream,gzip.open(target,'wb') as zipped:
            shutil.copyfileobj(stream,zipped)


def archive_m0(output,durable):
    output=Path(output);durable=Path(durable)
    acceptance=output/'full-acceptance'
    result_path=acceptance/'result.json'
    if not result_path.is_file():
        raise FileNotFoundError(f'M0 acceptance result is missing: {result_path}')
    current=json.loads(result_path.read_text())
    if current.get('status')=='running':
        raise RuntimeError('cannot archive M0 while hardware acceptance is running')
    result_file=output/'m0-result.json'
    if not result_file.is_file():
        admission_file=output/'admission.json'
        if not admission_file.is_file():
            raise FileNotFoundError(f'M0 admission record is missing: {admission_file}')
        save(result_file,dict(status='pass' if current.get('status')=='pass' else 'failed',
            acceptance=current,admission=json.loads(admission_file.read_text())))
    if durable.exists():
        existing=durable/'m0-result.json'
        if not existing.is_file():raise FileExistsError('unrecognized existing M0 archive')
        previous=json.loads(existing.read_text())
        if ([row['path'] for row in previous['acceptance']['runs']] !=
                [row['path'] for row in current['runs']]):
            raise ValueError('existing M0 archive belongs to another hardware run')
    else:durable.mkdir(parents=True)
    for name in ('admission.json','config.json','m0-result.json'):
        source=output/name
        if source.is_file():shutil.copy2(source,durable/name)
    paths=[source for source in acceptance.rglob('*.json') if source.is_file()]
    for source in paths:
        target=durable/source.relative_to(acceptance);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    logs=list(output.glob('m0.log'))+list(acceptance.rglob('*.log'))+list(acceptance.rglob('*.jsonl'))
    for source in logs:
        relative=Path('supervisor')/source.name if source.parent==output else source.relative_to(acceptance)
        target=durable/relative.with_suffix(relative.suffix+'.gz')
        target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as stream,gzip.open(target,'wb') as zipped:
            shutil.copyfileobj(stream,zipped)


def run_diagnostic(config_path,raw_root,results_root):
    """Short separate captures; no P95 or performance acceptance from these runs."""
    raw_root=Path(raw_root);results_root=Path(results_root)
    raw_root.mkdir(parents=True,exist_ok=False);results_root.mkdir(parents=True,exist_ok=False)
    config=base_config(config_path);save(results_root/'admission.json',admission(config))
    config=serializable_config(config);outputs={}
    frozen=source_identity(ROOT)
    record=dict(status='running',runs=[],comparisons=[],source_identity=frozen)
    save(results_root/'result.json',record)
    try:
        for method in ('none','checkpoint_io_sink','ours'):
            if source_identity(ROOT)!=frozen:raise RuntimeError('diagnostic sources changed between runs')
            output=raw_root/method;settings=dict(config,method=method,stop_step=32,
                checkpoint_root=str(output/'checkpoints'),
                io_experiment=dict(events=True,skip_periodic_state=True))
            path=results_root/f'{method}-config.json';save(path,settings)
            command(path,output,stop=32);outputs[method]=output
            archive_run(output,results_root/method)
            if method!='none':
                correlated=timeline(read_events(output))
                save(results_root/method/'timeline.json',correlated)
                if correlated['status']!='correlated':raise ValueError('incomplete diagnostic event correlation')
            record['runs'].append(dict(method=method,path=str(output)))
            save(results_root/'result.json',record)
        for method in ('checkpoint_io_sink','ours'):
            record['comparisons'].append(dict(method=method,**compare(outputs['none'],outputs[method])))
        sink=chunk_signatures(outputs['checkpoint_io_sink'],'sink_consumed_source_safe')
        full=chunk_signatures(outputs['ours'],'chunk_send_begin')
        if sink!=full:raise ValueError('T1/T2 capture geometry, order, bytes or SHA256 differ')
        retained=[(generation,value) for generation,path,value in catalog.committed(outputs['ours']/'checkpoints')
                  if value['step']==28]
        if len(retained)!=1:raise ValueError('diagnostic requires one retained step-28 checkpoint')
        restore_output=raw_root/'ours-restore28-to32'
        restore_settings=dict(config,method='ours',stop_step=32,checkpoint_interval=64,
            checkpoint_root=str(outputs['ours']/'checkpoints'),
            io_experiment=dict(events=True,skip_periodic_state=True))
        restore_config=results_root/'ours-restore-config.json';save(restore_config,restore_settings)
        command(restore_config,restore_output,resume=True,generation=retained[0][0],stop=32)
        archive_run(restore_output,results_root/'ours-restore28-to32')
        record['comparisons'].append(dict(method='ours_fresh_restore_step28',
            **compare(outputs['none'],restore_output)))
        record.update(status='correctness_pass',capture_equivalence=dict(status='pass',
            rank_checkpoint_streams=len(sink),chunks=sum(len(rows) for rows in sink.values())))
        save(results_root/'result.json',record)
    except BaseException as error:
        record.update(status='failed',error=repr(error));save(results_root/'result.json',record);raise
    return record


def chunk_signatures(output,event):
    result={}
    for rank in range(4):
        with (Path(output)/f'io-events/rank-{rank}.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['event']!=event:continue
                result.setdefault((rank,row['step']),[]).append((row['tensor'],row['offset'],
                    row['bytes'],row['sha256']))
    if sorted(result)!=[(rank,step) for rank in range(4) for step in range(4,33,4)]:
        raise ValueError('incomplete diagnostic capture streams')
    return result


def profile_row(name,path,group,replicate):
    ranks=[]
    for rank in range(4):
        row=catalog.read_checked(Path(path)/f'rank_{rank}/training.json')
        losses=row['losses']
        selected=[v for v in losses if 28<=v['step']<=64]
        saves=[v for v in row['checkpoints'] if 28<=v['step']<=64]
        if len(selected)!=37:raise ValueError(f'{name}: expected steps 28..64 on rank {rank}')
        ranks.append(dict(rank=rank,physical_npu=rank+4,loss_steps=selected,
                          checkpoints=saves,save_count=len(saves),
                          final_state_sha256={k:v['sha256'] for k,v in
                            catalog.read_checked(Path(path)/f'rank_{rank}/state-step-64.json').items()}))
    return dict(name=name,group=group,replicate=replicate,path=str(path),ranks=ranks)


def run_m0(config_path,output):
    durable=DEFAULT_RESULTS/'m0'
    if durable.exists():raise FileExistsError('M0 evidence already exists: '+str(durable))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config=base_config(config_path);admit=admission(config)
    save(output/'admission.json',admit)
    acceptance=ROOT/'tools/full_checkpoint_acceptance.py'
    raw=json.loads(Path(config_path).read_text())
    raw['io_experiment']=dict(events=True)
    save(output/'config.json',raw)
    argv=[sys.executable,str(acceptance),'--config',str(output/'config.json'),'--npu-ids','4,5,6,7',
          '--output',str(output/'full-acceptance')]
    env=dict(os.environ,PYTHONPATH='python:.',ASCEND_RT_VISIBLE_DEVICES='4,5,6,7')
    with (output/'m0.log').open('x') as log:
        result=subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    record=dict(status='pass' if result.returncode==0 else 'failed',
                acceptance=json.loads((output/'full-acceptance/result.json').read_text()),
                admission=admit)
    save(output/'m0-result.json',record)
    archive_m0(output,durable)
    if result.returncode:raise RuntimeError(f'M0 FULL acceptance failed; inspect {output}/m0.log')
    return record


def run_m1(config_path,raw_root=DEFAULT_RAW,results_root=DEFAULT_RESULTS/'m1'):
    raw_root=Path(raw_root);results_root=Path(results_root)
    if results_root.exists():raise FileExistsError('M1 evidence already exists: '+str(results_root))
    config=base_config(config_path);admit=admission(config)
    config=serializable_config(config)
    m0=json.loads((results_root.parent/'m0/m0-result.json').read_text())
    diagnostic=json.loads((results_root.parent/'diagnostic/result.json').read_text())
    if m0['status']!='pass' or diagnostic['status']!='correctness_pass':
        raise ValueError('M1 requires passing M0 and T1/T2 diagnostic capture equivalence')
    frozen=source_identity(ROOT)
    if diagnostic['source_identity']!=frozen:raise ValueError('diagnostic source identity differs from M1')
    raw_root.mkdir(parents=True,exist_ok=True);results_root.mkdir(parents=True,exist_ok=False)
    save(results_root/'admission.json',admit)
    record=dict(status='running',workload=dict(model='Qwen3-8B',topology='TP4/DP1/PP1',
        physical_npus=[4,5,6,7],sequence_length=128,micro_batch=1,seed=42,
        warmup_steps=24,formal_steps=[28,64],learning_rate_horizon=128,
        checkpoint_interval=4,independent_processes=3),source_identity=frozen,runs=[],comparisons=[])
    save(results_root/'result.json',record)
    orders=(('none','checkpoint_io_sink','ours'),
            ('checkpoint_io_sink','ours','none'),
            ('ours','none','checkpoint_io_sink'))
    try:
        for rep,order in enumerate(orders):
            rep_root=raw_root/f'replicate-{rep}'
            rep_root.mkdir(exist_ok=False)
            run_paths={}
            for method in order:
                if source_identity(ROOT)!=frozen:raise RuntimeError('M1 sources changed between runs')
                name=f'{method}-r{rep}'
                output=rep_root/name;config_file=results_root/'configs'/f'{name}.json'
                run_config=dict(config,method=method,stop_step=64,lr_horizon=128,
                                checkpoint_root=str(output/'checkpoints'),
                                io_experiment=dict(events=False,skip_periodic_state=True))
                config_file.parent.mkdir(parents=True,exist_ok=True)
                config_file.write_text(json.dumps(run_config,indent=2)+'\n')
                command(config_file,output,stop=64)
                run_paths[method]=output
                record['runs'].append(profile_row(name,output,method,rep));save(results_root/'result.json',record)
                archive_run(output,results_root/'runs'/name)
            oracle=run_paths['none']
            for method in ('checkpoint_io_sink','ours'):
                comparison=compare(oracle,run_paths[method])
                record['comparisons'].append(dict(replicate=rep,method=method,**comparison))
                save(results_root/'result.json',record)
            committed=catalog.committed(run_paths['ours']/'checkpoints')
            candidates=[(generation,path,value) for generation,path,value in committed if value['step']==60]
            if len(candidates)!=1:raise ValueError('expected exactly one retained committed step-60 generation')
            generation,path,_=candidates[0]
            restore_out=rep_root/f'ours-restore60-to64-r{rep}'
            restore_config=dict(config,method='ours',stop_step=64,lr_horizon=128,
                                io_experiment=dict(events=False,skip_periodic_state=True),
                                checkpoint_root=str(run_paths['ours']/'checkpoints'))
            restore_config_path=results_root/'configs'/f'ours-restore-r{rep}.json'
            restore_config_path.write_text(json.dumps(restore_config,indent=2)+'\n')
            command(restore_config_path,restore_out,resume=True,generation=generation,stop=64)
            restore_archive=results_root/'runs'/f'ours-restore60-to64-r{rep}'
            archive_run(restore_out,restore_archive)
            verification=compare(oracle,restore_out)
            record['comparisons'].append(dict(replicate=rep,method='ours_fresh_restore_step60',
                generation=generation,**verification));save(results_root/'result.json',record)
        save(results_root/'analysis.json',analyse(record))
        save(results_root/'probe-overhead.json',probe_overhead(record,results_root.parent/'diagnostic'))
        record['status']='measurements_completed';save(results_root/'result.json',record)
    except BaseException as error:
        record.update(status='failed',error=repr(error));save(results_root/'result.json',record)
        raise
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('preflight','m0','diagnostic','m1','analyse','archive-m0','queue'))
    parser.add_argument('--config',type=Path,default=ROOT/'config/qwen_training.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--raw-root',type=Path,default=DEFAULT_RAW)
    parser.add_argument('--results-root',type=Path,default=DEFAULT_RESULTS/'m1')
    args=parser.parse_args()
    if args.stage=='analyse':
        save(args.results_root/'analysis.json',analyse(json.loads((args.results_root/'result.json').read_text())))
        return 0
    if args.stage=='archive-m0':
        archive_m0(args.output,args.results_root/'m0')
        return 0
    config=base_config(args.config)
    if args.stage=='preflight':
        result=admission(config);result['training_preflight']=qwen_training.preflight(config)
        print(json.dumps(result,indent=2));return 0
    if args.stage=='m0':
        if args.output is None:parser.error('--output required for M0')
        run_m0(args.config,args.output);return 0
    if args.stage=='diagnostic':
        run_diagnostic(args.config,args.raw_root,args.results_root);return 0
    if args.stage=='queue':
        run_m0(args.config,args.raw_root/'m0-retry')
        run_diagnostic(args.config,args.raw_root/'diagnostic',DEFAULT_RESULTS/'diagnostic')
        run_m1(args.config,args.raw_root/'m1',DEFAULT_RESULTS/'m1')
        return 0
    run_m1(args.config,args.raw_root,args.results_root);return 0


if __name__=='__main__':raise SystemExit(main())
