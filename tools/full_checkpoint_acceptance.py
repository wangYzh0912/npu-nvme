#!/usr/bin/env python3
"""Targeted ABI3 TP4 FULL acceptance, not a performance campaign.

Uses only the registered raw83 [256,1280) GiB region, without formatting.
Retained runs keep the existing supervisor's lease and are never killed here.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
sys.path.insert(0,str(ROOT))

from npu_nvme.cli import qwen_training as training
from npu_nvme.cli.contracts import digest, source_identity
from npu_nvme.runtime import training_catalog as catalog
from npu_nvme.runtime.qwen_config import resolve
from npu_nvme.runtime.training_validation import compare


def reuse_converted_weights(canonical, candidate):
    canonical=Path(canonical)/'training/qwen3_ms_converted_weight'
    candidate=Path(candidate)/'training/qwen3_ms_converted_weight'
    if not canonical.is_dir() or not candidate.is_dir():return 0
    files=lambda root: {p.relative_to(root):p for p in root.rglob('*')
                        if p.is_file() and (p.suffix=='.safetensors' or p.name=='param_name_map.json')}
    source, target=files(canonical),files(candidate)
    if source.keys()!=target.keys():return 0
    reused=0
    for relative,original in source.items():
        duplicate=target[relative]
        if digest(original)!=digest(duplicate):return 0
    for relative,original in source.items():
        duplicate=target[relative]
        if original.samefile(duplicate):continue
        temporary=duplicate.with_name(duplicate.name+'.verified-link')
        os.link(original,temporary)
        os.replace(temporary,duplicate)
        reused+=1
    return reused


def admit_conversion_space(model, output):
    cache_bytes=sum(shard.stat().st_size for shard in Path(model).glob('*.safetensors'))
    # The oracle cache stays present while the next run converts its own copy.
    required=2*cache_bytes+(8<<30)
    if shutil.disk_usage(output).free<required:
        raise OSError(f'insufficient output space for two converted model caches and 8 GiB headroom: need {required} bytes')


@contextmanager
def publication_fault():
    """Replace only this run's worker entry; the production CLI is unchanged."""
    original=subprocess.Popen
    def launch(argv,*args,**kwargs):
        argv=list(argv)
        if 'npu_nvme.workloads.qwen' in argv:
            index=argv.index('npu_nvme.workloads.qwen')
            if argv[index-1]!='-m':raise ValueError('unexpected training worker command')
            argv[index-1:index+1]=[str(ROOT/'tests/hardware/qwen_publication_fault.py')]
        return original(argv,*args,**kwargs)
    subprocess.Popen=launch
    try:yield
    finally:subprocess.Popen=original


def acceptance(raw,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',runs=[],comparisons=[],scope='targeted FULL correctness; no performance claim')
    weight_cache=None
    def record():training.write(output/'result.json',result)
    record()
    try:
        resolved=resolve(raw,ROOT)
        region=json.loads(Path(resolved['region']).read_text())
        physical_devices=raw.get('npu_ids',[0,1,2,3])
        if (region['pci'],region['offset_bytes'],region['length_bytes'])!=(
                '0000:83:00.0',256<<30,1<<40) or physical_devices not in ([0,1,2,3],[4,5,6,7]):
            raise ValueError('acceptance requires registered raw83 region and supported TP4 device mapping')
        smi=subprocess.check_output(['npu-smi','info'],text=True,timeout=30)
        training.write(output/'device-admission.json',dict(npu_info=smi))
        if not all(f'No running processes found in NPU {r}' in smi for r in physical_devices):
            result.update(status='blocked',reason=f'NPU {physical_devices} occupied; no hardware job started');record();return 3
        if os.geteuid()!=0:raise PermissionError('raw SPDK acceptance requires root')
        admit_conversion_space(resolved['model'],output)
        frozen=source_identity(ROOT)
        training.write(output/'execution-sources.json',frozen)
        def execute(name,*,method='ours',start=0,stop=8,root=None,resume=False,generation='latest',fault=False,interval=4):
            if source_identity(ROOT)!=frozen:raise RuntimeError('acceptance sources changed between runs')
            settings=dict(raw,method=method,stop_step=stop,lr_horizon=24,checkpoint_interval=interval)
            settings.pop('checkpoint_root',None)
            if root is not None:settings['checkpoint_root']=str(root)
            path=output/name
            if fault:
                with publication_fault():rc=training.run_fit(resolve(settings,ROOT),path)
            else:rc=training.run_fit(resolve(settings,ROOT),path,resume=resume,restore=generation)
            row=json.loads((path/'result.json').read_text())
            result['runs'].append(dict(name=name,path=str(path),returncode=rc,result=row));record()
            if fault:
                checkpoint=path/'checkpoints/generation-000000000001'
                proof=catalog.read_checked(checkpoint/'publication-fault.json')
                owner=json.loads((path/'owner/result.json').read_text())
                if (rc!=1 or not proof['injected'] or (checkpoint/'checkpoint.json').exists() or
                        owner['status']!='pass' or not owner['closed']):
                    raise RuntimeError('publication fault or safe owner close was not established')
            elif rc:raise RuntimeError('acceptance run failed: '+name)
            settings=json.loads((path/'run-config.json').read_text())
            if settings['start_step']!=start:raise ValueError('restored training step differs from acceptance case')
            nonlocal weight_cache
            if weight_cache is None:weight_cache=path
            else:reuse_converted_weights(weight_cache,path)
            return path
        def check(oracle,path):
            result['comparisons'].append(dict(run=str(path),**compare(oracle,path)));record()
        oracle=execute('trajectory-none',method='none',stop=24)
        source=execute('ours-source8');check(oracle,source)
        root=source/'checkpoints'
        explicit=execute('ours-explicit4',start=4,root=root,resume=True,generation='1',interval=24);check(oracle,explicit)
        middle=execute('ours-latest8-to16',start=8,stop=16,root=root,resume=True);check(oracle,middle)
        final=execute('ours-latest16-to24',start=16,stop=24,root=root,resume=True);check(oracle,final)
        fault=execute('ours-publication-interruption',fault=True)
        recovered=execute('ours-publication-recovery',start=4,root=fault/'checkpoints',resume=True)
        reconciliation=json.loads((recovered/'reconciliation.json').read_text())
        selection=json.loads((recovered/'restore-selection.json').read_text())
        if reconciliation['recovered']!=[1] or selection['generation']!=1:
            raise ValueError('fresh process did not reconcile and select the interrupted publication')
        publication=catalog.read_checked(fault/'checkpoints/generation-000000000001/checkpoint.json')
        if not publication.get('recovered'):raise ValueError('recovery provenance missing')
        check(oracle,recovered)
        result.update(status='pass');record();return 0
    except BaseException as error:
        result.update(status='fail',error=repr(error));record();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/qwen_training.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--npu-ids',default='0,1,2,3',help='physical TP4 mapping: 0,1,2,3 or 4,5,6,7')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    try:config['npu_ids']=[int(device) for device in args.npu_ids.split(',')]
    except ValueError:parser.error('--npu-ids must be a comma-separated TP4 map')
    return acceptance(config,args.output)


if __name__=='__main__':sys.exit(main())
