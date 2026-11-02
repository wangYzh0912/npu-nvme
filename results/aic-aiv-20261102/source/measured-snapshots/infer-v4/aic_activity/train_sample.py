"""Run validated training and attach an independent sample-only collector."""
import argparse,json,os,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    python='/models/npu_nvme_exp/user7-stack/conda/bin/python'
    env=dict(os.environ,PYTHONPATH=str(a.source/'python')+':'+str(a.source),PYTHONDONTWRITEBYTECODE='1')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    trainlog=a.output.with_suffix('.training.log').open('w')
    app=subprocess.Popen([python,'train.py','fit','--config',str(a.config),'--output',str(a.output)],cwd=a.source,env=env,stdout=trainlog,stderr=subprocess.STDOUT)
    collector=None
    try:
        deadline=time.monotonic()+1800
        while time.monotonic()<deadline:
            if app.poll() is not None:raise RuntimeError(f'training exited before collector attach: {app.returncode}')
            rank=a.output/'rank_0/training.json'
            if rank.exists():
                d=json.loads(rank.read_text())['payload']
                if d['losses'] and d['losses'][-1]['step']>=16:break
            time.sleep(.2)
        else:raise TimeoutError('no steady training progress within 30 minutes')
        collectlog=a.output.with_suffix('.collector.log').open('w')
        manifest='/models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/environments.json'
        collector=subprocess.Popen([python,'scripts/run_user_environment.py','--manifest',manifest,'--profile','candidate','--','msprof',
                                    '--output='+str(a.output/'pmu'),'--sys-devices=4,5,6,7','--sys-period=35','--ai-core=on',
                                    '--aic-mode=sample-based','--aic-freq=100','--aic-metrics=PipeUtilization','--task-time=off','--type=text'],
                                   cwd=a.source,env=env,stdout=collectlog,stderr=subprocess.STDOUT)
        code=app.wait(timeout=1800);sample_code=collector.wait(timeout=600)
        result=json.loads((a.output/'result.json').read_text())
        if code or sample_code or result.get('validation_status')!='pass':raise RuntimeError(f'training={code}, sample={sample_code}, evidence={result}')
        print(json.dumps(dict(status='pass',training=result,collector_exit=sample_code)),flush=True)
    finally:
        # Leave ownership with the supervisor; do not release live NPU buffers.
        trainlog.close()

if __name__=='__main__':main()
