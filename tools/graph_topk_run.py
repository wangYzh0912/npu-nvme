#!/usr/bin/env python3
"""Launch one graph Top-K run under the shared hardware lease, with no owner."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from user_environment import read_profile, isolated_environment
from npu_nvme.experiments.graph_workload import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'config/graph_topk.json')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--level', type=int, choices=range(9), default=0)
    parser.add_argument('--layout', choices=['serial', 'parallel'], default='serial')
    parser.add_argument('--scan-fraction', type=float, default=1.0)
    parser.add_argument('--ratio', type=float, default=.1)
    parser.add_argument('--role', choices=['main', 'auxiliary'], default='main')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--dump-graphs', action='store_true')
    args = parser.parse_args()
    options = json.loads(args.config.read_text())
    options.update(output=str(args.output.resolve()), level=args.level, layout=args.layout,
                   scan_fraction=args.scan_fraction, ratio=args.ratio, role=args.role,
                   profile=args.profile, dump_graphs=args.dump_graphs)
    options['strategy'] = str((ROOT / options['strategy']).resolve())
    if args.steps:
        options['formal_steps'] = args.steps
    output = Path(options['output']); output.mkdir(parents=True, exist_ok=False)
    lock_path = Path(options['hardware_lock']); lease = lock_path.with_suffix('.lease.json')
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if lease.exists():
            raise RuntimeError('unreconciled hardware lease')
        smi = subprocess.check_output(['npu-smi', 'info'], text=True)
        if not all(f'No running processes found in NPU {r}' in smi for r in range(4)):
            raise RuntimeError('TP4 cards occupied')
        profile = read_profile(options['environment_manifest'], 'candidate', ROOT)
        env = isolated_environment(profile, ROOT)
        env.update(ASCEND_RT_VISIBLE_DEVICES='0,1,2,3', RUN_MODE='finetune', PYTHONUNBUFFERED='1',
                   HCCL_CONNECT_TIMEOUT='300', MS_COMPILER_CACHE_PATH=str(output / 'compiler-cache'),
                   LOCAL_DEFAULT_PATH=str(output / 'framework-output'))
        sources = [ROOT / 'config/graph_topk.json', Path(__file__), *sorted((ROOT / 'python/npu_nvme/experiments').glob('graph_*.py'))]
        identity = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        write(output / 'source-identity.json', identity)
        write(output / 'run-config.json', options)
        write(output / 'environment.json', profile)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
        command = [profile['python'], '-m', 'mindspore.parallel.cluster.run', '--worker_num=4',
                   '--local_worker_num=4', '--master_addr=127.0.0.1', '--master_port=' + str(port),
                   '--join=True', '--cluster_time_out=300', '--log_dir=' + str(output / 'msrun-log'),
                   'python', '-m', 'npu_nvme.experiments.graph_workload', '--config', str(output / 'run-config.json')]
        write(output / 'command.json', command)
        child = None
        try:
            with (output / 'training.log').open('w') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, pass_fds=(lock.fileno(),))
                write(lease, dict(status='running', run=str(output), pid=os.getpid(),
                                 children=[dict(pid=child.pid, process_group=child.pid)], scope='graph_only_no_io'))
                returncode = child.wait(timeout=14400)
            reports = [json.loads((output / f'rank_{r}/result.json').read_text()) if (output / f'rank_{r}/result.json').exists() else dict(status='missing') for r in range(4)]
            changed = [str(p) for p in sources if identity[str(p.relative_to(ROOT))] != hashlib.sha256(p.read_bytes()).hexdigest()]
            if changed:
                raise RuntimeError('source changed during run: ' + repr(changed))
            passed = returncode == 0 and all(r['status'] == 'pass' for r in reports)
            result = dict(status='pass' if passed else 'failed', returncode=returncode,
                          rank_status=[r['status'] for r in reports], mode='diagnostic' if args.profile else 'timing')
            if passed:
                result.update(completion_seconds=max(r['all_done_ns'] for r in reports) / 1e9 - min(r['begin_ns'] for r in reports) / 1e9,
                              rank_seconds=[(r['all_done_ns'] - r['begin_ns']) / 1e9 for r in reports],
                              training_seconds=(max(r['training_end_ns'] for r in reports) - min(r['begin_ns'] for r in reports)) / 1e9,
                              peak_memory_bytes=[r['memory_peak_bytes'] for r in reports])
            write(output / 'result.json', result)
            return int(not passed)
        except BaseException as error:
            write(output / 'result.json', dict(status='failed', error=repr(error)))
            raise
        finally:
            # Never erase a lease while live device workers remain.
            smi = subprocess.check_output(['npu-smi', 'info'], text=True)
            idle = all(f'No running processes found in NPU {r}' in smi for r in range(4))
            if child is not None and child.poll() is not None and idle:
                lease.unlink(missing_ok=True)
                cache = output / 'framework/qwen3_ms_converted_weight'
                if cache.is_dir():
                    shutil.rmtree(cache)  # Regenerable HF conversion cache, never initial FULL or evidence.
            elif child is not None:
                write(lease, dict(status='retained', run=str(output), pid=os.getpid(), child=child.pid))
                while True:
                    time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
