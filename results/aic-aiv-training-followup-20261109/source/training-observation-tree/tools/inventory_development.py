#!/usr/bin/env python3
"""Capture the implementation baseline and source inventories without device I/O."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def command(argv):
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=30)
    return dict(argv=argv, exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)

def write(root, name, data):
    (root / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--source-worktree', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    paths = command(['git', 'ls-files'])['stdout'].splitlines()
    source_paths = [p for p in paths if p.endswith(('.py', '.c', '.h')) and not p.startswith('third_party/')]
    callers = []
    modes = []
    for name in source_paths:
        path = ROOT / name
        content = path.read_text()
        for line, text in enumerate(content.splitlines(), 1):
            symbols = sorted(set(re.findall(r'\bnpu_nvme_[a-zA-Z0-9_]+\b', text)))
            if symbols:
                callers.append(dict(path=name, line=line, symbols=symbols, source=text.strip(),
                                    file_sha256=digest(path)))
            if re.search(r'\b(serial|queue|async|frozen_async|live_async)\b', text) and 'io_mode' in text:
                modes.append(dict(path=name, line=line, source=text.strip()))
    write(args.out, 'caller_inventory.json', {'scope':'symbol references, including declarations; semantic ownership is reviewed per migration', 'entries':callers})
    write(args.out, 'legacy_mode_map.json', {
        'entry':'DirectCheckpoint.save', 'source':'python/direct_checkpoint.py',
        'modes':{
            'serial':{'capture':'frozen','transport':'legacy_sync','wait':'background worker; benchmark caller explicitly waits'},
            'queue':{'capture':'frozen','transport':'legacy_sync','wait':'caller-controlled handle'},
            'async':{'capture':'frozen','transport':'async write only where supported; existing host/CRC callers require migration','wait':'caller-controlled handle'},
            'frozen_async':{'capture':'frozen','transport':'same write route as async','wait':'save waits for persistence'},
            'live_async':{'capture':'live generation-owned pinned Host staging','transport':'Host request after ACL event','wait':'update fence plus later persistence wait'}},
        'admission':{'block':'two blocking semaphores without deadline in BASE','try':'nonblocking busy'},'source_references':modes})
    plan = args.plan.read_text()
    corpus=[]
    for line in plan.splitlines():
        if re.match(r'\| RC-\d\d \|', line):
            cells=[x.strip() for x in line.strip('|').split('|')]
            corpus.append(dict(id=cells[0],input=cells[1],historical_claim=cells[2],target=cells[3],gate=cells[4],
                               evidence_level='H',state='planned',runtime_reproduced=False))
    write(args.out,'regression_inventory.json',corpus)
    candidates=['config/user_environments.json','python/user_environment.py','scripts/run_user_environment.py',
        'experiments/benchmarks/environment_upgrade_inventory.py','experiments/training/train_qwen3_full_restart.py',
        'experiments/training/check_qwen_training_run.py','scripts/run_qwen3_four_rank.sh','scripts/run_qwen_worker.sh',
        'tests/python/test_environment_upgrade.py','tests/python/test_evidence_reporting.py','tests/python/test_qwen_training_entry.py',
        'src/npu_nvme.c','python/c_bindings.py','tests/hardware/g1_full_restart.py','tests/hardware/g2_metadata.py']
    write(args.out,'migration_inventory.json',[{'path':n,'source_sha256':digest(args.source_worktree/n),
        'base_sha256':digest(ROOT/n) if (ROOT/n).is_file() else None,'state':'review_before_transfer'} for n in candidates])
    write(args.out,'base_manifest.json',{'commit':command(['git','rev-parse','HEAD'])['stdout'].strip(),
        'plan_sha256':digest(args.plan),'plan_path':str(args.plan),'source_worktree':str(args.source_worktree),
        'source_status':subprocess.check_output(['git','status','--short'],cwd=args.source_worktree,text=True),
        'source_blobs':{n:command(['git','rev-parse','HEAD:'+n])['stdout'].strip() for n in ['python/direct_checkpoint.py','src/npu_nvme.c']},
        'build_dependencies':{p:{'exists':Path(p).exists(),'sha256':digest(p) if Path(p).is_file() else None} for p in [
            '/home/user7/npu-nvme/build/dpdk_fix/librte_mempool_ring_fixed.a',
            '/home/user7/npu-nvme/third_party/spdk/build/lib/libspdk_nvme.a']},
        'hardware_write':{'state':'blocked','pci':'0000:83:00.0','authorized_region':None,
                          'reason':'Dedicated offset range not yet established; old 64/128 GiB offsets are not current ownership proof'},
        'reviewer':'self-reviewed','can_prove':['source inventory'],'cannot_prove':['runtime correctness','DMA safety']})
    macros=[]
    for n in source_paths:
        if n.startswith('tests/c/'):
            macros += [{'path':n,'line':i,'source':l.strip()} for i,l in enumerate((ROOT/n).read_text().splitlines(),1) if re.search('HAS_NPU|TBD|npu_nvme_init',l)]
    write(args.out,'c_test_inventory.json',macros)
    print(json.dumps({'call_references':len(callers),'regression_seeds':len(corpus),'out':str(args.out)}))

if __name__ == '__main__':
    main()
