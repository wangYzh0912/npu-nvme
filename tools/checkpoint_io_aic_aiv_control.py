#!/usr/bin/env python3
"""Small single-card AIC/AIV bandwidth controls for the M1 report."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=4)
    parser.add_argument('--lock', type=Path,
                        default=Path('/models/npu_nvme_exp/user7-stack/hardware-campaign.lock'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=120)
    args = parser.parse_args()
    if args.device != 4 or args.repeats != 3 or args.iterations != 120:
        raise ValueError('control contract is frozen to NPU4, 3 repeats, 120 iterations')
    lease = args.lock.with_suffix('.lease.json')
    if lease.exists():
        raise RuntimeError(f'hardware lease is active: {lease}')
    with args.lock.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('hardware campaign lock is active') from error
        info = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=30)
        if 'No running processes found in NPU 4' not in info:
            raise RuntimeError('NPU4 is occupied')
        args.output.mkdir(parents=True, exist_ok=False)
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(args.device)
        import numpy as np
        import mindspore as ms
        from mindspore import Tensor, ops

        if ms.__version__ != '2.7.1':
            raise RuntimeError(f'unexpected MindSpore version: {ms.__version__}')
        ms.set_context(device_target='Ascend', device_id=0, mode=ms.PYNATIVE_MODE)
        ms.set_seed(42)
        rng = np.random.default_rng(42)
        left = Tensor(rng.standard_normal((4096, 4096)).astype(np.float32), ms.bfloat16)
        right = Tensor(rng.standard_normal((4096, 4096)).astype(np.float32), ms.bfloat16)
        vector_left = Tensor(rng.standard_normal((16 * 1024 * 1024,)).astype(np.float32), ms.bfloat16)
        vector_right = Tensor(rng.standard_normal((16 * 1024 * 1024,)).astype(np.float32), ms.bfloat16)
        matmul = ops.MatMul()
        add = ops.Add()
        aic_stream = ms.runtime.Stream()
        aiv_stream = ms.runtime.Stream()

        def run(mode, count):
            result_aic = result_aiv = None
            if mode == 'aic_only':
                with ms.runtime.StreamCtx(aic_stream):
                    for _ in range(count):
                        result_aic = matmul(left, right)
            elif mode == 'aiv_only':
                with ms.runtime.StreamCtx(aiv_stream):
                    for _ in range(count):
                        result_aiv = add(vector_left, vector_right)
            elif mode == 'serial_pair':
                for _ in range(count):
                    result_aic = matmul(left, right)
                    result_aiv = add(vector_left, vector_right)
            elif mode == 'parallel_pair':
                with ms.runtime.StreamCtx(aic_stream):
                    for _ in range(count):
                        result_aic = matmul(left, right)
                with ms.runtime.StreamCtx(aiv_stream):
                    for _ in range(count):
                        result_aiv = add(vector_left, vector_right)
                aic_stream.synchronize()
                aiv_stream.synchronize()
            else:
                raise ValueError('unknown mode')
            ms.runtime.synchronize()
            if result_aic is not None and not np.isfinite(result_aic.asnumpy()).all():
                raise RuntimeError(f'{mode} produced non-finite matrix output')
            if result_aiv is not None:
                observed = result_aiv.asnumpy().astype(np.float32)
                expected = (vector_left[:1024] + vector_right[:1024]).asnumpy().astype(np.float32)
                if not np.isfinite(observed).all():
                    raise RuntimeError(f'{mode} produced non-finite vector output')
                if not np.array_equal(observed[:1024], expected):
                    raise RuntimeError(f'{mode} vector output differs')

        for mode in ('aic_only', 'aiv_only', 'serial_pair', 'parallel_pair'):
            run(mode, 1)

        msprof = shutil.which('msprof')
        if not msprof:
            raise FileNotFoundError('msprof is not available in the candidate environment')
        command = [msprof, '--output=' + str(args.output / 'system-profile'),
                   '--sys-devices=' + str(args.device), '--sys-period=36',
                   '--ai-core=on', '--aic-mode=sample-based', '--aic-freq=100',
                   '--aic-metrics=PipeUtilization', '--task-time=off',
                   '--sys-hardware-mem=on', '--sys-hardware-mem-freq=50', '--type=text']
        (args.output / 'collector-command.json').write_text(json.dumps(command, indent=2) + '\n')
        records = []
        with (args.output / 'collector.log').open('w') as log:
            collector = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                time.sleep(2)
                if collector.poll() is not None:
                    raise RuntimeError('msprof system collector exited before measurement')
                for mode in ('aic_only', 'aiv_only', 'serial_pair', 'parallel_pair'):
                    for repeat in range(args.repeats):
                        begin_mono = time.monotonic_ns()
                        begin_real = time.time_ns()
                        run(mode, args.iterations)
                        end_mono = time.monotonic_ns()
                        end_real = time.time_ns()
                        records.append(dict(mode=mode, repeat=repeat,
                            monotonic_begin_ns=begin_mono, monotonic_end_ns=end_mono,
                            realtime_begin_ns=begin_real, realtime_end_ns=end_real,
                            aic_iterations=args.iterations if mode != 'aiv_only' else 0,
                            aiv_iterations=args.iterations if mode != 'aic_only' else 0,
                            elapsed_ms=(end_mono - begin_mono) / 1e6))
                code = collector.wait(timeout=120)
                if code:
                    raise RuntimeError(f'msprof system collector exited with {code}')
            finally:
                if collector.poll() is None:
                    collector.terminate()
                    collector.wait(timeout=30)

        result = dict(status='pass', device=args.device, mindspore=ms.__version__,
            shape=dict(matrix=[4096, 4096], vector=[16 * 1024 * 1024], dtype='bfloat16'),
            semantics='AIC-only matrix and AIV-only vector are measured in isolated windows; '
                'serial and two-stream co-run report device-wide HBM only, never per-engine split',
            collector=dict(command=command, aic_sample_hz=100, hbm_sample_hz=50),
            windows=records, npu_info=info)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(status=result['status'], output=str(args.output), windows=len(records)),
                         indent=2))


if __name__ == '__main__':
    main()
