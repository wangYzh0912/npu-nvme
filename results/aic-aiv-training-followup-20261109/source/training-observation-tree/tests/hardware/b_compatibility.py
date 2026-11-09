#!/usr/bin/env python3
"""B bidirectional H01 compatibility; each writer/reader is a fresh process.

Only the explicitly authorized 83 device is supported by this evidence runner.
The existing V2 geometry is never reformatted. Nonzero phase exit stops the run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ('python', 'src', 'include')
            for p in sorted((root/directory).rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.c', '.h')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--directions', nargs='+', default=['entry-to-exit', 'exit-to-entry'],
                        choices=['entry-to-exit', 'exit-to-entry', 'entry-to-entry', 'exit-to-exit'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43])
    parser.add_argument('--deterministic', choices=['ON', 'OFF'], default='ON')
    parser.add_argument('--phase-timeout', type=float, default=300)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error('hardware runner requires root and explicit CANN environment')
    region = json.loads((ROOT/'config/raw_test_region.json').read_text())
    if region['pci_addr'] != '0000:83:00.0' or region['write_authorized'] is not True:
        parser.error('83 whole-device authorization missing')
    if Path('/sys/bus/pci/devices/0000:83:00.0/driver').resolve().name != 'uio_pci_generic':
        parser.error('83 is not bound to uio_pci_generic')
    if Path('/sys/bus/pci/devices/0000:84:00.0/driver').resolve().name != 'nvme':
        parser.error('protected 84 device driver changed')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    entry = args.entry_root.resolve()
    sources = {str(root): snapshot(root) for root in (entry, ROOT)}
    write(out/'sources.json', sources)
    results = []
    implementations = {'entry': entry, 'exit': ROOT}
    for direction in args.directions:
        writer, reader = (implementations[name] for name in direction.split('-to-'))
        for seed in args.seeds:
            run = out/f'{direction}-s{seed}'
            run.mkdir()
            for phase, implementation in (('save', writer), ('restore', reader)):
                env = dict(os.environ, NPU_NVME_TEST_CHECKPOINT_ROOT=str(implementation),
                           PYTHONPATH=str(implementation/'python'), NPU_NVME_DPDK_ARGS='--iova-mode=pa')
                env['LD_LIBRARY_PATH'] = str(implementation/'build_out/lib') + ':' + os.environ.get('LD_LIBRARY_PATH', '')
                argv = [sys.executable, str(ROOT/'tests/hardware/c1_training_state_restart.py'),
                    '--phase', phase, '--run-dir', str(run), '--pci', '0000:83:00.0',
                    '--npu', '7', '--model', 'gpt2', '--seed', str(seed), '--seq-len', '129',
                    '--save-step', '2', '--continue-steps', '3', '--dropout-rate', '0',
                    '--slot-size-gb', '10', '--shm-id', str(os.getpid()+len(results)+20000),
                    '--io-timeout', '120', '--deterministic', args.deterministic]
                record = dict(direction=direction, seed=seed, phase=phase, argv=argv,
                              implementation=str(implementation))
                write(run/f'{phase}_command.json', record)
                started = time.monotonic()
                print(f'{direction} seed={seed} {phase}', flush=True)
                with (run/f'{phase}.log').open('w') as log:
                    try:
                        process = subprocess.run(argv, cwd=run, env=env, stdout=log,
                                                 stderr=subprocess.STDOUT, timeout=args.phase_timeout)
                        returncode = process.returncode
                    except subprocess.TimeoutExpired:
                        returncode = 124
                record.update(returncode=returncode, seconds=time.monotonic()-started)
                results.append(record)
                write(out/'phases.json', results)
                if returncode:
                    write(out/'result.json', dict(status='fail', phases=results))
                    return 1
            result = json.loads((run/'result.json').read_text())
            if result['status'] != 'pass' or not result['loaded_state_byte_exact'] or result['continuation_steps'] < 3:
                raise AssertionError('H01 compatibility proof missing')
    changed = [str(root) for root in (entry, ROOT) if snapshot(root) != sources[str(root)]]
    write(out/'result.json', dict(status='pass' if not changed else 'invalid', phases=results,
        source_changed=changed, cases=len(args.directions)*len(args.seeds),
        deterministic=args.deterministic, directions=args.directions, seeds=args.seeds,
        boundary='B legacy FULL compatibility; not D1 strict restore'))
    return int(bool(changed))


if __name__ == '__main__':
    sys.exit(main())
