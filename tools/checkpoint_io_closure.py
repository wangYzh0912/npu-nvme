#!/usr/bin/env python3
"""Preregistered E1 -> E4 -> E2/E3 M1 closure queue. Never format raw media."""
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
sys.path[:0]=[str(ROOT/'python'),str(ROOT)]
from tools.checkpoint_io import (save,base_config,serializable_config,command,archive_run,
    chunk_signatures,profile_row)
from tools.full_checkpoint_acceptance import reuse_converted_weights
from npu_nvme.cli.contracts import source_identity
from npu_nvme.cli.qwen_training import preflight
from npu_nvme.runtime import training_catalog as catalog
from npu_nvme.runtime.training_validation import compare
from npu_nvme.experiments.checkpoint_io_analysis import METHODS
from npu_nvme.experiments.checkpoint_io_events import read_events
from npu_nvme.experiments.checkpoint_io_closure import (complete_timeline,lifecycle_gate,
    overhead_gate,device_summary,dependency_bound)

DEFAULT_RAW=Path('/dev/shm/user7-aic-aiv-io-closure-20261116')
DEFAULT_RESULTS=Path('/home/user7/npu-nvme-evidence/aic-aiv-io-20261116')
ORDERS=(METHODS,('checkpoint_io_sink','ours','none'),('ours','none','checkpoint_io_sink'))


def schedule():
    rows=[]
    for rep,order in enumerate(ORDERS):
        for method in order:
            modes=('A','B') if (rep+METHODS.index(method))%2==0 else ('B','A')
            for mode in modes:
                rows.append(dict(name=f'{method}-{mode}-r{rep}',method=method,mode=mode,replicate=rep))
    return rows


def admit(config,raw,durable):
    if Path(config['hardware_lock']).with_suffix('.lease.json').exists():
        raise RuntimeError('hardware lease is active; do not override it')
    info=subprocess.check_output(['npu-smi','info'],text=True,timeout=30)
    if any(f'No running processes found in NPU {device}' not in info for device in config['npu_ids']):
        raise RuntimeError('physical NPU 4..7 occupied')
    if config['npu_ids']!=[4,5,6,7]:raise ValueError('closure requires physical NPU 4..7')
    region=json.loads(Path(config['region']).read_text())
    if (region['pci'],region['offset_bytes'],region['length_bytes'],region['chunk_bytes'])!=(
            '0000:83:00.0',256<<30,1024<<30,4<<20):
        raise ValueError('registered raw83 geometry differs')
    def free(path):
        path=Path(path)
        while not path.exists():path=path.parent
        return shutil.disk_usage(path).free
    tmp_free,durable_free=free(raw),free(durable)
    if tmp_free<96<<30:raise OSError('raw tmpfs headroom below 96 GiB')
    if durable_free<8<<30:raise OSError('durable evidence headroom below 8 GiB')
    return dict(npu_info=info,tmpfs_free=tmp_free,durable_free=durable_free,
                raw_root=str(raw),durable_root=str(durable),raw_format_authorized=False)


class Campaign:
    def __init__(self,config,raw,durable):
        self.config=serializable_config(config);self.raw=Path(raw);self.out=Path(durable)
        self.identity=source_identity(ROOT);self.cache=None
        admission=admit(config,self.raw,self.out)
        self.out.mkdir(parents=True,exist_ok=False);self.raw.mkdir(parents=True,exist_ok=False)
        save(self.out/'admission.json',admission)
        self.record=dict(status='running',phase='preflight',source_identity=self.identity,
            workload=dict(model='Qwen3-8B',tp=4,physical_npus=[4,5,6,7],sequence=128,
                micro_batch=1,seed=42,lr_horizon=128,interval=4,chunk_bytes=4<<20,depth=4,retention=3),
            runs=[],comparisons=[],e2_order=schedule(),
            gates=dict(e1='not_run',e2='not_run',e3='not_run',e4='not_run'))
        self.write()
        try:
            audit=preflight(config);save(self.out/'preflight.json',audit)
            library=Path(audit['profile']['library'])
            self.binary=dict(path=str(library),sha256=catalog.digest_file(library))
            self.binaries={str(path):catalog.digest_file(path) for path in
                [library,*sorted((ROOT/'python/npu_nvme/framework').glob('_address_native*.so'))]}
            m0=Path('/models/npu_nvme_exp/user7-stack/aic-aiv-io-20261110/m0/m0-result.json')
            if json.loads(m0.read_text())['status']!='pass':raise ValueError('previous FULL M0 acceptance not passed')
            self.record.update(native_binary=self.binary,binary_sha256=self.binaries,
                               prior_m0=dict(path=str(m0),sha256=catalog.digest_file(m0)))
            self.write()
        except BaseException as error:
            self.record.update(status='failed',error=repr(error));self.write();raise

    def write(self):save(self.out/'result.json',self.record)

    def frozen(self):
        if source_identity(ROOT)!=self.identity:raise RuntimeError('frozen source identity changed')
        if any(catalog.digest_file(path)!=digest for path,digest in self.binaries.items()):
            raise RuntimeError('frozen native binary or allocation extension changed')

    def run(self,name,method,*,mode='B',stop=32,resume=None,profile=False,save_final=True):
        self.frozen()
        admit(self.config,self.raw,self.out)
        output=self.raw/name;settings=dict(self.config,method=method,stop_step=stop,
            checkpoint_root=str(output/'checkpoints') if resume is None else str(resume[0]),
            checkpoint_interval=4 if save_final else 128,
            io_experiment=dict(closure=True,events=mode=='B',skip_periodic_state=True,device_profile=profile))
        config_path=self.out/'configs'/f'{name}.json';save(config_path,settings)
        self.record['active_run']=dict(name=name,method=method,mode=mode,raw=str(output),
            config=str(config_path),begin_ns=time.monotonic_ns())
        self.write()
        command(config_path,output,stop=stop,resume=resume is not None,
                generation=resume[1] if resume else 'latest')
        self.frozen()
        lifecycle=lifecycle_gate(output,detailed=mode=='B')
        save(output/'closure-lifecycle.json',lifecycle)
        if mode=='B' and method!='none' and resume is None:
            reports=[catalog.read_checked(output/f'rank_{rank}/training.json') for rank in range(4)]
            correlated=complete_timeline(read_events(output),reports,method,range(4,stop+1,4))
            save(output/'closure-timeline.json',correlated)
            if correlated['status']!='pass':raise ValueError('full boundary or required event coverage failed: '+name)
        if self.cache is None:self.cache=output
        else:
            reused=reuse_converted_weights(self.cache,output)
            if not reused:raise ValueError('verified converted-weight deduplication failed: '+name)
            save(output/'converted-weight-reuse.json',dict(canonical=str(self.cache),verified_links=reused))
        archive_run(output,self.out/'runs'/name,
                    exclude=tuple(f'rank_{rank}/device-profile' for rank in range(4)) if profile else ())
        with output.with_suffix('.supervisor.log').open('rb') as stream, gzip.open(self.out/'runs'/name/'supervisor.log.gz','wb') as zipped:
            shutil.copyfileobj(stream,zipped)
        self.record['runs'].append(dict(name=name,method=method,mode=mode,path=str(output),
            source_identity=self.identity,end_to_end=lifecycle['end_to_end']))
        self.record.pop('active_run',None);self.write()
        return output

    def comparison(self,oracle,run,**identity):
        self.record['comparisons'].append(dict(**identity,**compare(oracle,run)));self.write()

    def initial_comparison(self,oracle,run):
        for rank in range(4):
            for kind in ('state','control'):
                name=f'rank_{rank}/{kind}-initial.json'
                if catalog.read_checked(oracle/name)!=catalog.read_checked(run/name):
                    raise ValueError('fresh-process initial state/control differs: '+str(run))

    def restore(self,source,name,step,*,mode='B',stop=32):
        retained=[(generation,path) for generation,path,value in catalog.committed(source/'checkpoints')
                  if value['step']==step]
        if len(retained)!=1:raise ValueError('expected one retained checkpoint at step '+str(step))
        return self.run(name,'ours',mode=mode,stop=stop,resume=(source/'checkpoints',retained[0][0]),save_final=False)

    def e1(self):
        self.record['phase']='e1_full_boundary';self.write();paths={}
        for method in METHODS:paths[method]=self.run('e1-'+method,method)
        for method in METHODS[1:]:
            self.initial_comparison(paths['none'],paths[method])
            self.comparison(paths['none'],paths[method],stage='e1',method=method)
        sink=chunk_signatures(paths['checkpoint_io_sink'],'sink_consumed_source_safe')
        full=chunk_signatures(paths['ours'],'chunk_send_begin')
        if sink!=full:raise ValueError('all-rank T1/T2 chunk order/geometry/bytes/SHA differ')
        save(self.out/'e1-chunk-equivalence.json',dict(status='pass',streams=len(sink),
            chunks=sum(map(len,sink.values())),logical_bytes=sum(v[2] for rows in sink.values() for v in rows)))
        restored=self.restore(paths['ours'],'e1-restore28-to32',28)
        self.comparison(paths['none'],restored,stage='e1',method='fresh_restore28')
        self.record['gates']['e1']='pass';self.write()
        return paths['none']

    def e4(self,oracle):
        self.record['phase']='e4_device_captures';self.write()
        inventory=[]
        for method in METHODS:
            output=self.run('e4-'+method,method,profile=True)
            self.initial_comparison(oracle,output)
            self.comparison(oracle,output,stage='e4',method=method)
            reports=[catalog.read_checked(output/f'rank_{rank}/training.json') for rank in range(4)]
            bound=dependency_bound(reports)
            for rank in range(4):
                report=catalog.read_checked(output/f'rank_{rank}/training.json')
                if report.get('device_profile',{}).get('status')!='captured':
                    raise ValueError('device profiler did not complete')
                directory=output/f'rank_{rank}/device-profile'
                files=[p for p in directory.rglob('*') if p.is_file()]
                if not files:raise ValueError('device profiler produced no evidence')
                for source in files:
                    target=self.out/'runs'/output.name/source.relative_to(output)
                    target=target.with_suffix(target.suffix+'.gz');target.parent.mkdir(parents=True,exist_ok=True)
                    with source.open('rb') as stream,gzip.open(target,'wb') as zipped:shutil.copyfileobj(stream,zipped)
                summaries=[p for p in files if p.suffix=='.csv']
                trace=[p for p in files if p.suffix=='.json' and 'trace' in p.name.lower()]
                if not summaries or not trace:raise ValueError('device trace/CSV export missing')
                inventory.append(dict(method=method,rank=rank,profile=report['device_profile'],
                    exports=device_summary(directory),dependency=bound,
                    files=[dict(path=str(p.relative_to(output)),archive_path=str(p.relative_to(output))+'.gz',
                                bytes=p.stat().st_size,sha256=catalog.digest_file(p)) for p in files]))
                save(self.out/'e4-device-inventory.json',dict(status='running',captures=inventory))
        save(self.out/'e4-device-inventory.json',dict(status='dependency_limit_verified',captures=inventory,
            interpretation='Independent profiler runs, excluded from E2 statistics. Verified synchronous dependency limit; no device-service, engine-idleness or PMU-overlap claim.'))
        self.record['gates']['e4']='dependency_limit_verified';self.write()

    def e2(self):
        self.record['phase']='e2_e3_observer_matrix';self.write();metrics=[];paths={};restores={}
        for spec in schedule():
            name,method,mode,rep=(spec[k] for k in ('name','method','mode','replicate'))
            output=self.run('e2-'+name,method,mode=mode,stop=64)
            paths[method,mode,rep]=output
            self.initial_comparison(self.raw/'e1-none',output)
            metrics.append(dict(profile_row(name,output,method,rep),mode=mode))
            save(self.out/'e2-processes.json',dict(runs=metrics))
            if method=='ours':
                restores[mode,rep]=self.restore(output,'e2-restore60-'+mode+f'-r{rep}',60,mode=mode,stop=64)
            if len([key for key in paths if key[2]==rep])==6:
                for checked_mode in ('A','B'):
                    oracle=paths['none',checked_mode,rep]
                    for checked_method in METHODS[1:]:
                        self.comparison(oracle,paths[checked_method,checked_mode,rep],
                            stage='e2',method=checked_method,mode=checked_mode,replicate=rep)
                    self.comparison(oracle,restores[checked_mode,rep],
                        stage='e2',method='fresh_restore60',mode=checked_mode,replicate=rep)
                self.comparison(paths['none','A',rep],paths['none','B',rep],stage='e2',method='t0_A_B',replicate=rep)
        gate=overhead_gate(metrics);save(self.out/'e2-overhead-gate.json',gate)
        self.record['gates']['e2']=gate['status'];self.record['gates']['e3']='pass';self.write()
        if gate['status']!='pass':raise ValueError('A/B observation perturbation exceeds ten percent')

    def queue(self):
        try:
            oracle=self.e1();self.e4(oracle);self.e2()
            self.record.update(status='closure_gates_passed',phase='completed',
                limitation='E4 establishes the synchronous dependency limit; engine/PMU bottleneck attribution remains a separate analysis. No M2 implementation authorized.')
            self.write()
        except BaseException as error:
            self.record.update(status='failed',error=repr(error));self.write()
            active=self.record.get('active_run')
            if active and Path(active['raw']).exists():
                try:
                    archive_run(active['raw'],self.out/'failed'/active['name'])
                    log=Path(active['raw']).with_suffix('.supervisor.log')
                    if log.exists():
                        with log.open('rb') as stream,gzip.open(self.out/'failed'/active['name']/'supervisor.log.gz','wb') as zipped:
                            shutil.copyfileobj(stream,zipped)
                except Exception as archive_error:
                    self.record['failure_archive_error']=repr(archive_error);self.write()
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('preflight','queue'))
    parser.add_argument('--config',type=Path,default=ROOT/'config/qwen_training.json')
    parser.add_argument('--raw-root',type=Path,default=DEFAULT_RAW)
    parser.add_argument('--results-root',type=Path,default=DEFAULT_RESULTS)
    args=parser.parse_args();config=base_config(args.config)
    if args.stage=='preflight':
        print(json.dumps(dict(admission=admit(config,args.raw_root,args.results_root),
            training=preflight(config),e2_order=schedule()),indent=2));return 0
    Campaign(config,args.raw_root,args.results_root).queue();return 0


if __name__=='__main__':raise SystemExit(main())
