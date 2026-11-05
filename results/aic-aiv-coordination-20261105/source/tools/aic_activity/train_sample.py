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
        baseline_steps=None;attach_step=None
        import statistics
        while time.monotonic()<deadline:
            if app.poll() is not None:raise RuntimeError(f'training exited before collector attach: {app.returncode}')
            rank=a.output/'rank_0/training.json'
            if rank.exists():
                d=json.loads(rank.read_text())['payload']
                if d['losses'] and d['losses'][-1]['step']>=16:
                    rank_data=[json.loads((a.output/f'rank_{rank}/training.json').read_text())['payload']['losses'] for rank in range(4)]
                    common=min(r[-1]['step'] for r in rank_data)
                    if common>=16:
                        baseline_steps=list(range(common-5,common+1))
                        times=[[(r['end_ns']-r['begin_ns'])/1e6 for r in rows if r['step'] in baseline_steps] for rows in rank_data]
                        if all(statistics.pstdev(values)/statistics.mean(values)<=.05 for values in times):
                            attach_step=common;break
                        if common>=40:raise RuntimeError('training warmup did not stabilize within 40 steps')
            time.sleep(.2)
        else:raise TimeoutError('no steady training progress within 30 minutes')
        collectlog=a.output.with_suffix('.collector.log').open('w')
        manifest='/models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/environments.json'
        collector=subprocess.Popen([python,'scripts/run_user_environment.py','--manifest',manifest,'--profile','candidate','--','msprof',
                                    '--output='+str(a.output/'pmu'),'--sys-devices=4,5,6,7','--sys-period=35','--ai-core=on',
                                    '--aic-mode=sample-based','--aic-freq=100','--aic-metrics=PipeUtilization','--task-time=off','--type=text'],
                                   cwd=a.source,env=env,stdout=collectlog,stderr=subprocess.STDOUT)
        sample_code=collector.wait(timeout=600)
        code=app.wait(timeout=1800)
        result=json.loads((a.output/'result.json').read_text())
        if code or sample_code or result.get('validation_status')!='pass':raise RuntimeError(f'training={code}, sample={sample_code}, evidence={result}')
        from .pmu import read as read_pmu
        ranks=[]
        for rank in range(4):
            devices=list((a.output/'pmu').glob(f'PROF_*/device_{rank+4}'))
            if len(devices)!=1:raise ValueError(f'expected one PMU device directory: {devices}')
            samples,pmu_quality=read_pmu(devices[0])
            sample_begin=max(min(r['start'] for r in values) for values in samples.values())
            sample_end=min(max(r['end'] for r in values) for values in samples.values())
            losses=json.loads((a.output/f'rank_{rank}/training.json').read_text())['payload']['losses']
            before=[(r['end_ns']-r['begin_ns'])/1e6 for r in losses if r['step'] in baseline_steps]
            selected=[r for r in losses if r['step']>attach_step and r['wall_begin_ns']>=sample_begin+1_000_000_000 and r['wall_end_ns']<sample_end]
            measured=[(r['end_ns']-r['begin_ns'])/1e6 for r in selected]
            if len(measured)<6:raise ValueError('too few complete steady sampled steps')
            baseline=statistics.median(before);overhead=statistics.median(measured)/baseline-1
            ranks.append(dict(rank=rank,control_median_ms=baseline,measured_median_ms=statistics.median(measured),
                              collector_device_path=str(devices[0]),pmu_quality=pmu_quality,
                              warmup_cv=statistics.pstdev(before)/statistics.mean(before),profile_overhead_fraction=overhead,
                              windows=[dict(id=f"step-{r['step']}",begin_ns=r['wall_begin_ns'],end_ns=r['wall_end_ns']) for r in selected]))
        quality='valid' if all(r['profile_overhead_fraction']<=.1 and r['warmup_cv']<=.05 for r in ranks) else 'diagnostic_perturbation_or_stability'
        evidence=dict(status='pass',measurement_quality=quality,training=result,collector_exit=sample_code,ranks=ranks)
        evidence.update(attach_after_step=attach_step,baseline_steps=baseline_steps)
        (a.output/'sample-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print(json.dumps(evidence),flush=True)
    finally:
        # Leave ownership with the supervisor; do not release live NPU buffers.
        trainlog.close()

if __name__=='__main__':main()
