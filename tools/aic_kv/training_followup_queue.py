"""One serial device queue with explicit experimental quality gates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .training_jit_campaign import RAW,RESULTS
from .training_jit_report import summarize

SOURCE=Path('/tmp/user7-aic-aiv-training-followup-source')


def invoke(stage,*extra):
    return subprocess.run([sys.executable,'-m','tools.aic_kv.training_jit_campaign',
        '--source',str(SOURCE),'--stage',stage,*extra],check=False).returncode


def baseline_gate(mode):
    rows=[]
    for run in sorted((RESULTS/'training/runs').glob(mode.lower()+'*')):
        if (run/'sample-evidence.json').is_file():
            rows.extend(r for r in summarize(run) if r['sequence_length'] in (128,512) and r['micro_batch']==1)
    accepted=len(rows)==24 and all(r['measurement_accepted'] for r in rows)
    details=dict(mode=mode,expected_device_rows=24,actual_device_rows=len(rows),
                 measurement_accepted=accepted,production_numerical_acceptance=False)
    (RESULTS/f'{mode.lower()}-baseline-gate.json').write_text(json.dumps(details,indent=2)+chr(10))
    return accepted


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int);args=parser.parse_args()
    # Existing owned fragment campaign completes before the model queue.
    if args.wait_pid:
        while Path('/proc',str(args.wait_pid)).exists():time.sleep(2)
    if list((RESULTS/'fragments/runs').glob('*/failure.json')):
        raise RuntimeError('fragment queue failed; inspect retained failure before model continuation')
    if len(list((RESULTS/'fragments/runs').glob('*/result.json')))!=27:
        raise RuntimeError('fragment queue is incomplete')
    if invoke('o1-baseline','--retry-lock','--retry-interrupted'):raise RuntimeError('O1 baseline stopped')
    o1=baseline_gate('O1')
    if not o1:raise RuntimeError('O1 measurement quality gate failed; matrix not started')
    o2_smoke=invoke('o2-smoke')
    mode='O1'
    o2_status=dict(smoke_exit_code=o2_smoke,failure_records='training/failures',
                   production_optimization_accepted=False)
    if not o2_smoke:
        smoke=RESULTS/'training/runs/o2-seq128-mb1-r0'
        quality=all(r['measurement_accepted'] for r in summarize(smoke))
        o2_status['smoke_measurement_accepted']=quality
        if quality:
            code=invoke('o2-baseline')
            o2_status.update(baseline_exit_code=code,quality_gate=baseline_gate('O2') if not code else False)
    (RESULTS/'o2-observation-status.json').write_text(json.dumps(o2_status,indent=2)+chr(10))
    (RESULTS/'matrix-mode.json').write_text(json.dumps(dict(mode=mode,
        reason='registered O1 official fusion path; O2 separately assessed, not mixed into one recurrence matrix'),indent=2)+chr(10))
    if invoke('prevalence','--mode',mode,'--retry-lock','--retry-interrupted'):
        raise RuntimeError('prevalence matrix stopped; inspect failed run')
    subprocess.run([sys.executable,'-m','tools.aic_kv.training_fragment_campaign',
        '--source',str(SOURCE),'--resident-gradients'],check=True)
    if invoke('o0-hbm'):raise RuntimeError('O0 HBM control stopped')
    subprocess.run([sys.executable,'-m','tools.aic_kv.training_fragment_report'],check=True)
    subprocess.run([sys.executable,'-m','tools.aic_kv.training_jit_report'],check=True)
    subprocess.run([sys.executable,'-m','tools.aic_kv.training_jit_render'],check=True)
    (RESULTS/'queue-completed.json').write_text(json.dumps(dict(status='measurements_completed',production_optimization_accepted=False))+chr(10))


if __name__=='__main__':main()
