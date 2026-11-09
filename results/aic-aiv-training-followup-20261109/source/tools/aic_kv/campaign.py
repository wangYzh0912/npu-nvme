"""Hold the shared hardware lock across owned experiment subprocesses."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/models/npu_nvme_exp/user7-stack/aic-aiv-kv-20261103')
RAW = Path('/dev/shm/user7-aic-aiv-20261031/inference/kv-20261103')
CONTAINER = 'user7-aic-aiv-030-priv'
LOCK = Path('/models/npu_nvme_exp/user7-stack/hardware-campaign.lock')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['metadata', 'capacity', 'work', 'calibration', 'micro'], required=True)
    p.add_argument('--blocks', type=int)
    p.add_argument('--tag', default='')
    p.add_argument('--metadata-run', default='metadata')
    args = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if LOCK.with_suffix('.lease.json').exists():
            raise RuntimeError('unreconciled hardware lease')
        smi = subprocess.check_output(['npu-smi', 'info'], text=True)
        if not all(f'No running processes found in NPU {i}' in smi for i in (4,5,6,7)):
            raise RuntimeError('requested NPU devices are occupied')
        source = ROOT/'source'/(args.stage+('-'+args.tag if args.tag else ''))
        shutil.copytree(Path(__file__).parent, source/'aic_kv', ignore=shutil.ignore_patterns('__pycache__'))
        sources = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in source.rglob('*.py')}
        (source/'SHA256SUMS.json').write_text(json.dumps(sources,indent=2)+'\n')
        info = json.loads(subprocess.check_output(['docker','inspect',CONTAINER],text=True))[0]
        if not info['State']['Running']:
            subprocess.run(['docker','start',CONTAINER],check=True)
        (source/'container.json').write_text(json.dumps(dict(image=info['Image'],
            mounts=info['Mounts'], env=info['Config']['Env']),indent=2)+'\n')
        old_python = next(x.split('=',1)[1] for x in info['Config']['Env'] if x.startswith('PYTHONPATH='))
        pythonpath = str(source)+':/experiment:'+old_python
        def execute(name, command):
            logpath = RAW/(name+'.log')
            with logpath.open('x') as log:
                cmd = ['docker','exec','-e','PYTHONPATH='+pythonpath,'-e','PYTHONUNBUFFERED=1',
                       CONTAINER,'python','-u','-m',*command]
                print('START',name,flush=True)
                try:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
                except BaseException:
                    subprocess.run(['docker','exec',CONTAINER,'chmod','-R','a+rX',
                                    '/output/kv-20261103'],check=True)
                    # docker exec exit does not guarantee engine/worker teardown.
                    subprocess.run(['docker','stop','--time','10',CONTAINER],check=True)
                    raise
                else:
                    subprocess.run(['docker','exec',CONTAINER,'chmod','-R','a+rX',
                                    '/output/kv-20261103'],check=True)
            print('DONE',name,flush=True)
        if args.stage == 'metadata':
            name='metadata'+('-'+args.tag if args.tag else '')
            execute(name, ['aic_kv.run','--output','/output/kv-20261103/'+name,'--metadata-only'])
        elif args.stage == 'calibration':
            execute('calibration',['aic_kv.calibrate','--output','/output/kv-20261103/calibration'])
        elif args.stage == 'capacity':
            if args.blocks is None:
                snapshots=json.loads((RAW/args.metadata_run/'b1-l128-o32/kv-layout.json').read_text())
                args.blocks=min(x['num_blocks'] for x in snapshots)
            b=args.blocks
            for rep,capacities in enumerate(((b,b//2,b//4),(b//2,b//4,b),(b//4,b,b//2))):
                for capacity in capacities:
                    name=f'capacity-r{rep}-n{capacity}'
                    execute(name,['aic_kv.run','--output','/output/kv-20261103/'+name,
                                  '--blocks',str(capacity),'--system-sample'])
                    trace=RAW/name/'b1-l128-o32/trace'
                    # Analyse after model workers exit (vLLM workers are daemons).
                    execute(name+'-export',['aic_kv.analyse','--trace',str(trace).replace(str(RAW),'/output/kv-20261103')])
        elif args.stage == 'work':
            if args.blocks is None:
                args.blocks=min(x['num_blocks'] for x in json.loads((RAW/args.metadata_run/'b1-l128-o32/kv-layout.json').read_text()))
            cases=','.join(f'1:{l}:{g}' for l in (128,512) for g in (1,2,8,32))+',8:128:32'
            for rep in range(3):
                name=f'work-r{rep}'
                execute(name,['aic_kv.run','--output','/output/kv-20261103/'+name,
                              '--blocks',str(args.blocks),'--cases',cases])
                execute(name+'-export',['aic_kv.analyse','--trace','/output/kv-20261103/'+name])
        else:
            name='micro'+('-'+args.tag if args.tag else '')
            execute(name,['aic_kv.microbench','--output','/output/kv-20261103/'+name,
                            '--metadata','/output/kv-20261103/'+args.metadata_run+'/b1-l128-o32/kv-layout.json',
                            '--native-probe','/output/kv-20261103/'+args.metadata_run+'/b1-l128-o32/kv-probe-rank0.json'])


if __name__ == '__main__':
    main()
