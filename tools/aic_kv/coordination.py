"""Run isolated coordination probes, keeping older archives immutable."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/models/npu_nvme_exp/user7-stack/aic-aiv-coordination-20261105')
RAW=Path('/dev/shm/user7-aic-aiv-20261031/scope-20261104/coordination-20261105')
LOCK=Path('/models/npu_nvme_exp/user7-stack/hardware-campaign.lock')
CONTAINER='user7-aic-scope-official'

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def run(name,arguments):
    target=RAW/name
    if target.exists():raise RuntimeError('output exists: '+str(target))
    source=RAW/'sources'/name
    source.mkdir(parents=True)
    for package in ('aic_kv','aic_activity'):
        shutil.copytree(Path(__file__).parent.parent/package,source/package,ignore=shutil.ignore_patterns('__pycache__'))
    info=json.loads(subprocess.check_output(['docker','inspect',CONTAINER]))[0]
    env=dict(s.split('=',1) for s in info['Config']['Env'])
    visible='/output/coordination-20261105'
    command=['docker','exec','-e','PYTHONPATH='+visible+'/sources/'+name+':'+env['PYTHONPATH'],
             '-e','VLLM_CACHE_ROOT='+visible+'/compiler-cache/'+name,
             '-e','TORCHINDUCTOR_CACHE_DIR='+visible+'/inductor-cache/'+name,
             '-e','PYTHONUNBUFFERED=1',CONTAINER,'python','-u','-m','aic_kv.run',
             '--output',visible+'/'+name,*arguments]
    save(ROOT/'commands'/f'{name}.json',command)
    with (RAW/(name+'.log')).open('x') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    subprocess.run(['docker','exec',CONTAINER,'chmod','-R','a+rX',visible],check=True)
    if result.returncode == 0 and list(target.glob('*/trace')):
        export=command[:command.index('aic_kv.run')]+['aic_kv.analyse','--trace',visible+'/'+name]
        with (ROOT/(name+'-export.log')).open('x') as log:
            subprocess.run(export,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
    subprocess.run(['docker','exec',CONTAINER,'chmod','-R','a+rX',visible],check=True)
    shutil.copy2(RAW/(name+'.log'),ROOT/(name+'.log'))
    for p in target.rglob('*.json'):
        if not {'trace','system'} & set(p.relative_to(target).parts):
            dest=ROOT/'evidence'/name/p.relative_to(target);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    text=(RAW/(name+'.log')).read_text()
    cause='driver graph-update rejection' if 'current driver version does not support to update this op' in text else 'runtime failure; see log'
    if 'output changed within one deterministic condition' in text:
        cause='strict repeated output correctness gate failed'
    status=dict(name=name,returncode=result.returncode,cause=None if result.returncode==0 else cause)
    save(ROOT/'status'/f'{name}.json',status)
    print(json.dumps(status),flush=True)
    return result.returncode==0

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['inference','inference-cold','piecewise-cold','piecewise-pa-cache','diagnostics','correctness-full'],required=True);a=p.parse_args()
    ROOT.mkdir(exist_ok=True);RAW.mkdir(exist_ok=True)
    if shutil.disk_usage('/models').free < 2 << 30:
        raise RuntimeError('/models headroom below 2 GiB')
    with LOCK.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if LOCK.with_suffix('.lease.json').exists():raise RuntimeError('hardware lease occupied')
        smi=subprocess.check_output(['npu-smi','info'],text=True)
        if not all(f'No running processes found in NPU {i}' in smi for i in (4,5,6,7)):raise RuntimeError('cards occupied')
        subprocess.run(['docker','start',CONTAINER],check=True)
        try:
            if a.stage=='piecewise-pa-cache':
                for rep in range(3):
                    if not run(f'cold-piecewise-pa-cache-r{rep}',[
                        '--graph-mode','PIECEWISE','--official-pa-cache','--blocks','1654',
                        '--cases','1:128:32,8:128:32','--trace-cases','b1-l128-o32' if rep==0 else 'none',
                        '--trace-requests','1','--metadata-probe','--control-requests','6']):raise RuntimeError('model condition failed; see retained status/log')
                return
            if a.stage=='correctness-full':
                accepted=[]
                for label,extra in [('pa',[]),('fia',['--force-fia-with-pa-cache'])]:
                    accepted.append(run('correctness-full-'+label,[
                        '--graph-mode','FULL_DECODE_ONLY','--pa-shapes','1,8','--blocks','1654',
                        '--max-model-len','2304','--cases','8:128:32,8:2048:32',
                        '--prompt-fixture','diverse','--request-metrics',
                        '--trace-cases','none','--control-requests','6',*extra]))
                if not all(accepted):raise RuntimeError('strict output gate failed; both paths retained')
                return
            if a.stage=='piecewise-cold':
                for rep in range(3):
                    if not run(f'cold-piecewise-dense-r{rep}',[
                        '--graph-mode','PIECEWISE','--layout','dense','--blocks','1654',
                        '--cases','1:128:32,8:128:32','--trace-cases','none','--control-requests','6']):raise RuntimeError('model condition failed; see retained status/log')
                return
            if a.stage=='diagnostics':
                for name,extra in [('pa',[]),('fia',['--force-fia-with-pa-cache'])]:
                    if not run('diagnostic-full-'+name,[
                        '--graph-mode','FULL_DECODE_ONLY','--pa-shapes','1,8','--blocks','1654',
                        '--cases','1:128:32,8:128:32','--trace-requests','1',
                        '--control-requests','3','--metadata-probe','--system-sample',*extra]):raise RuntimeError('model condition failed; see retained status/log')
                return
            prefix='cold-' if a.stage=='inference-cold' else ''
            passed=run(prefix+'full-pa-cache-fia-probe',[
                '--graph-mode','FULL_DECODE_ONLY','--pa-shapes','1,8','--force-fia-with-pa-cache',
                '--blocks','1654','--cases','1:128:32','--metadata-probe','--trace-requests','1',
                '--control-requests','3'])
            if not passed:raise RuntimeError('inference intervention probe failed; see retained status/log')
            variants=[('pa',[]),('fia',['--force-fia-with-pa-cache'])]
            for rep in range(3):
                for label,extra in variants[rep%2:]+variants[:rep%2]:
                    if not run(prefix+f'full-pa-cache-{label}-r{rep}',[
                        '--graph-mode','FULL_DECODE_ONLY','--pa-shapes','1,8',
                        '--blocks','1654','--cases','1:128:32,8:128:32',
                        '--trace-cases','none','--control-requests','6',*extra]):raise RuntimeError('model condition failed; see retained status/log')
        finally:
            subprocess.run(['docker','stop','--time','10',CONTAINER],check=True)

if __name__=='__main__':main()
