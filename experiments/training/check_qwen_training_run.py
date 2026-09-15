#!/usr/bin/env python3
"""Audit a finished four-rank run without importing the NPU runtime."""
import argparse
import json
from pathlib import Path
import struct
import sys


def inspect_checkpoint(path, expected_step):
    with path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 32 * 1024 * 1024:
            raise ValueError('Invalid safetensors header size')
        header = json.loads(stream.read(length))
        tensors = {k: v for k, v in header.items() if k != '__metadata__'}
        end = max(v['data_offsets'][1] for v in tensors.values())
        if path.stat().st_size != 8 + length + end:
            raise ValueError(f'Truncated checkpoint: {path}')
        moment_keys = [k for k in tensors if k.startswith('adam_m.')]
        if not moment_keys:
            raise ValueError('Missing Adam state')
        for key in moment_keys:
            name = key.removeprefix('adam_m.')
            m, v, weight = tensors[key], tensors['adam_v.' + name], tensors[name]
            if not (m['shape'] == v['shape'] == weight['shape']):
                raise ValueError(f'Moment shape mismatch: {name}')
            if m['dtype'] != 'F32' or v['dtype'] != 'F32':
                raise ValueError(f'Adam state is not FP32: {name}')
        weights = [k for k in tensors if k.endswith('.weight') and not k.startswith(('adam_m.', 'adam_v.'))]
        if set(weights) != {k.removeprefix('adam_m.') for k in moment_keys}:
            raise ValueError('Adam states do not cover every model weight')
        scalar_values = {}
        formats = {'I32': '<i', 'I64': '<q', 'F32': '<f'}
        for key in ('global_step', 'step_num', 'epoch_num', 'loss_scale'):
            entry = tensors[key]
            start, end = entry['data_offsets']
            stream.seek(8 + length + start)
            scalar_values[key] = struct.unpack(formats[entry['dtype']], stream.read(end - start))[0]
        if scalar_values['global_step'] != expected_step or scalar_values['step_num'] != expected_step:
            raise ValueError(f'Wrong checkpoint step: {scalar_values}')
    return {'path': str(path), 'bytes': path.stat().st_size, 'adam_parameter_pairs': len(moment_keys),
            'model_parameter_tensors': len(weights), 'control': scalar_values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'status': 'failed', 'restart': 'not_run', 'oracle_compare': 'not_run', 'ranks': []}
    try:
        oracle_losses = None
        for rank in range(4):
            report = json.loads((args.output / f'rank_{rank}/acceptance.json').read_text())
            if report['status'] != 'training_pass_restart_not_tested':
                raise ValueError(f'Rank {rank} did not finish training')
            losses = report['losses']
            if not losses or any(x['overflow'] for x in losses):
                raise ValueError(f'Rank {rank}: missing or overflowed steps')
            if oracle_losses is not None and losses != oracle_losses:
                raise ValueError('Loss records disagree across tensor-parallel ranks')
            oracle_losses = losses
            directory = args.output / f'training/checkpoint/rank_{rank}'
            meta = json.loads((directory / 'meta.json').read_text())
            checkpoint = inspect_checkpoint(directory / meta['last_ckpt_file'], losses[-1]['step'])
            result['ranks'].append({'rank': rank, 'steps': len(losses), 'first_loss': losses[0]['loss'],
                                    'last_loss': losses[-1]['loss'], 'checkpoint': checkpoint})
        result['status'] = 'training_and_checkpoint_structure_pass'
        result['limits'] = 'Short fixed-text training; checkpoint structure checked, fresh-process restart not tested.'
    except Exception as error:
        result['error'] = repr(error)
    path = args.output / 'acceptance.json'
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['status'] == 'training_and_checkpoint_structure_pass' else 1


if __name__ == '__main__':
    sys.exit(main())
