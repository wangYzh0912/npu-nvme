#!/usr/bin/env python3
"""Publish allowlisted graph experiment source and compact measurements."""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent / 'npu-nvme-results'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign', type=Path, default=Path('/models/npu_nvme_exp/user7-stack/graph-topk-20260917-001'))
    args = parser.parse_args()
    relatives = ['EXECUTION_ENVIRONMENT_AND_COMMANDS.md', 'config/qwen_release_status.json', 'config/graph_topk.json', 'tools/graph_topk_run.py', 'tools/publish_graph_topk.py',
                 'docs/plans/GRAPH_TOPK_EXECUTION.md', 'tools/graph_topk_report.py', 'tools/graph_topk_analyse.py']
    for pattern in ('python/npu_nvme/experiments/graph_*.py', 'tests/python/test_graph_topk*.py'):
        relatives += [str(p.relative_to(ROOT)) for p in ROOT.glob(pattern)]
    for relative in relatives:
        src = ROOT / relative
        if src.exists():
            dst = WORK / relative; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    dest = WORK / 'results/graph-topk-20260917'
    dest.mkdir(parents=True, exist_ok=True)
    for src in (ROOT / 'results/graph-topk-20260917').glob('*'):
        if src.is_file(): shutil.copy2(src, dest / src.name)
    runs = []
    for run in sorted(args.campaign.glob('*')):
        if not run.is_dir(): continue
        row = dict(run=run.name)
        for name in ('run-config.json', 'source-identity.json', 'result.json', 'compile-interruption.json', 'lease-reconciliation.json', 'device-overlap-summary.json', 'dependency-gate.json'):
            src = run / name
            if src.exists():
                dst = dest / 'runs' / run.name / name; dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                if name == 'result.json': row.update(json.loads(src.read_text()))
        row['ranks'] = []
        for rank in range(4):
            src = run / f'rank_{rank}/result.json'
            if not src.exists(): src = run / f'rank_{rank}/progress.json'
            if src.exists():
                payload = json.loads(src.read_text())
                target = dest / 'runs' / run.name / f'rank-{rank}.json'; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(payload, indent=2) + '\n')
                row['ranks'].append(dict(rank=rank, status=payload['status']))
        runs.append(row)
    (dest / 'runs.json').write_text(json.dumps(dict(updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), runs=runs), indent=2) + '\n')
    subprocess.run(['git', 'add', '-f', '--', *relatives, 'results/graph-topk-20260917',
                    'results/incremental-phase1-20260916/status.json', 'results/incremental-phase1-20260916/README.md'], cwd=WORK, check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=WORK).returncode:
        subprocess.run(['git', 'commit', '-m', 'Record graph Top-K experiment implementation and progress'], cwd=WORK, check=True)
    subprocess.run(['git', 'push', 'origin', 'incremental-phase1-results'], cwd=WORK, check=True, timeout=120)

if __name__ == '__main__':
    if '--loop' in sys.argv:
        sys.argv.remove('--loop')
        while True:
            try:
                subprocess.run([sys.executable, str(ROOT / 'tools/graph_topk_report.py')], cwd=ROOT, check=True)
                subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], cwd=ROOT, check=True)
            except Exception as error:
                print(repr(error), flush=True)
            time.sleep(120)
    else:
        main()
