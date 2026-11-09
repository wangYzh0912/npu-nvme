"""Freeze completed or explicitly incomplete training investigation evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from tools.aic_activity.archive import manifest
from .training_jit_campaign import RESULTS,RAW
from .training_jit_report import summarize,recurrence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=RESULTS)
    parser.add_argument('--allow-incomplete',action='store_true');args=parser.parse_args();root=args.root
    lease=Path('/models/npu_nvme_exp/user7-stack/hardware-campaign.lease.json')
    if lease.exists():raise RuntimeError('active model lease must finish before finalization')
    smi=subprocess.check_output(['npu-smi','info'],text=True)
    if not all(f'No running processes found in NPU {d}' in smi for d in (4,5,6,7)):
        raise RuntimeError('experiment cards occupied')
    rows=[]
    for run in sorted((root/'training/runs').glob('*')):
        if (run/'sample-evidence.json').is_file():rows.extend(summarize(run))
    matrix=recurrence(rows,'O1')
    resident=list((root/'resident-fragments/runs').glob('*/result.json'))
    resident_ok=len(resident)==27 and all(json.loads(p.read_text())['oracle']['passed'] for p in resident)
    resident_timing_ok=sum(statistics.pstdev(d['timings_ms_per_update'])/statistics.mean(d['timings_ms_per_update'])<=.05
                           for d in (json.loads(p.read_text()) for p in resident))
    measured_complete=all(r['matrix_complete'] for r in matrix)
    o0=[r for r in rows if r['jit_level']=='O0' and r['sequence_length']==128 and r['micro_batch']==1]
    o0_controls=[]
    for run in sorted({r['run'] for r in o0}):
        own=[r for r in o0 if r['run']==run]
        o0_controls.append(dict(run=run,all_four_cards_accepted=len(own)==4 and
            all(r['measurement_accepted'] and r.get('hbm_measurement_accepted',False) for r in own)))
    o0_ok=any(r['all_four_cards_accepted'] for r in o0_controls)
    o2_attempts=list((root/'status').glob('o2-seq128-mb1-r0.json'))
    o2_attempted=len(o2_attempts)==1 and json.loads(o2_attempts[0].read_text())['status'] in ('complete','failed')
    model_diagnostic=root/'model-fragment/diagnostic-evidence.json'
    model_diagnostic_ok=model_diagnostic.is_file() and json.loads(model_diagnostic.read_text())['status']=='pass'
    identified=root/'model-fragment/optimizer-tail-evidence.json'
    model_diagnostic_ok=model_diagnostic_ok and identified.is_file() and len(json.loads(identified.read_text())['rows'])==12
    complete=measured_complete and resident_ok and o0_ok and o2_attempted and model_diagnostic_ok
    if not complete and not args.allow_incomplete:raise RuntimeError('required measurements or resident fragment oracle incomplete')
    subprocess.run([sys.executable,'-m','tools.aic_kv.training_counter_replay','--root',str(root)],check=True)
    (root/'acceptance.json').write_text(json.dumps(dict(
        status='investigation_complete' if complete else 'investigation_incomplete',
        planned_measurements_complete=complete,production_optimization_accepted=False,
        recurrence_per_card=matrix,resident_fragment_processes=len(resident),
        resident_fragment_oracles_passed=resident_ok,raw=str(RAW),
        resident_fragment_stable_timing_processes=resident_timing_ok,
        resident_fragment_timing_acceptance='each CV<=5%; unstable repeats remain diagnostic',
        matched_o0_hbm_control_accepted=o0_ok,official_o2_path_attempt_recorded=o2_attempted,
        matched_o0_hbm_control_attempts=o0_controls,
        real_model_optimizer_task_diagnostic_completed=model_diagnostic_ok,
        full_model_cross_jit_numerical_equivalence=False,real_model_ready_work='unknown',
        cooperative_scheduling_gain_measured=False,
        no_bandwidth_saturation_or_reclaimable_flops_claim=True),indent=2)+chr(10))
    (root/'final-npu-info.txt').write_text(smi)
    source=root/'source/tools';source.mkdir(parents=True,exist_ok=True)
    for package in ('aic_kv','aic_activity'):
        shutil.copytree(Path(__file__).parents[1]/package,source/package,
                        ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    files=[]
    for p in sorted(source.rglob('*.py')):
        files.append(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (root/'source-identity.json').write_text(json.dumps(files,indent=2)+chr(10))
    result=manifest(root)
    if result['total_bytes']>2<<30:raise RuntimeError('archive exceeds 2GiB')
    entries=[r for r in result['files'] if r['path']!='DELIVERY_MANIFEST.json']
    (root/'DELIVERY_MANIFEST.json').write_text(json.dumps(dict(kind='compact_review_package',
        raw_archive=str(root),files=entries),indent=2)+chr(10))
    result=manifest(root)
    if result['total_bytes']>2<<30:raise RuntimeError('archive exceeds 2GiB')
    print(json.dumps(dict(complete=complete,files=len(result['files']),bytes=result['total_bytes'])),flush=True)


if __name__=='__main__':main()
