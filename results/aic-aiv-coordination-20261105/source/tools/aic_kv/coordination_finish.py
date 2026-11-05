"""Finalize evidence after devices are released; experimental gates stay explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .coordination import ROOT, RAW
from .coordination_report import collect, compare
from tools.aic_activity.archive import manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();root=a.root
    smi=subprocess.check_output(['npu-smi','info'],text=True)
    if not all(f'No running processes found in NPU {d}' in smi for d in (4,5,6,7)):
        raise RuntimeError('cannot finalize while experiment cards are occupied')
    (root/'final-npu-info.txt').write_text(smi)
    rows=collect(root);comparisons=compare(rows)
    training=[]
    for length in (128,512):
        target=root/f'training/seq{length}'
        q=json.loads((target/'sample-evidence.json').read_text())
        oracle=json.loads((root/f'training/seq{length}-loss-oracle.json').read_text())
        graphs=json.loads((root/f'training/seq{length}-graphs/graph-evidence.json').read_text())
        training.append(dict(sequence_length=length,execution_passed=True,
            sample_quality=q['measurement_quality'],
            exact_loss_trajectory_matches=all(r['exact_loss_trajectory_matches'] for r in oracle['ranks']),
            static_graphs=len(graphs['ranks']),runtime_readiness='unknown'))
    correctness=[json.loads(path.read_text()) for path in sorted((root/'status').glob('correctness-full-*.json'))]
    if len(correctness)!=2:raise ValueError('both strict correctness conditions must be accounted for')
    acceptance=dict(status='investigation_complete_correctness_and_runtime_readiness_unresolved',
        investigation_complete=True,production_optimization_accepted=False,
        inference_backend_comparisons=comparisons,training=training,
        correctness_conditions=correctness,
        model_corun=dict(status='not_entered_by_gate',reason='no real-model ready AIC candidate proven'),
        multi_request_optimization=dict(status='not_entered_by_gate',reason='strict repeated output gate failed on native FULL/PA and experimental FULL/FIA'),
        raw_inference=str(RAW),raw_training='/dev/shm/user7-aic-aiv-coordination-training-r1',
        unprofiled_process_conditions=len(rows),initial_shared_aot_runs='diagnostic_not_backend_intervention_acceptance',
        independent_repetitions=3,system_samples_per_backend=1,
        no_reclaimable_flops_or_global_scheduler_omission_claim=True)
    (root/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    source=root/'source';source.mkdir(exist_ok=True)
    for package in ('aic_kv','aic_activity'):
        shutil.copytree(Path(__file__).parent.parent/package,source/'tools'/package,
            ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    for name in ('aic_aiv_followup.py','aic_aiv_analyse.py'):
        shutil.copy2(Path(__file__).parent.parent/name,source/'tools'/name)
    snapshot=[dict(path=str(p.relative_to(source)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(source.rglob('*.py'))]
    (root/'source-identity.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    result=manifest(root)
    if result['total_bytes']>2<<30:raise RuntimeError('durable evidence exceeds 2 GiB budget')
    delivery=dict(kind='compact_review_package',raw_archive=str(root),
        original_manifest='SHA256SUMS.json',files=result['files'])
    (root/'DELIVERY_MANIFEST.json').write_text(json.dumps(delivery,indent=2)+'\n')
    result=manifest(root)
    if result['total_bytes']>2<<30:raise RuntimeError('durable evidence exceeds 2 GiB budget')
    print(json.dumps(dict(files=len(result['files']),bytes=result['total_bytes'],status=acceptance['status'])),flush=True)

if __name__=='__main__':main()
