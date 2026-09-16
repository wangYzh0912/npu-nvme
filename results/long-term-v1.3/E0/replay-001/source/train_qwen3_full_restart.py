#!/usr/bin/env python3
"""Run real Qwen3 training; training success is not restart acceptance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def write_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/models/Qwen3-8B')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--seq-length', type=int, default=128)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup_steps < 0 or args.seq_length < 8 or args.seq_length % 4:
        parser.error('steps must be positive; warmup nonnegative; sequence length >=8 and divisible by 4')
    rank = int(os.environ.get('RANK_ID', '0'))
    out = Path(args.output).resolve()
    rank_dir = out / f'rank_{rank}'
    rank_dir.mkdir(parents=True, exist_ok=True)
    report_path = rank_dir / 'acceptance.json'
    report = {
        'model': args.model, 'rank': rank, 'pid': os.getpid(),
        'python': sys.executable, 'environment_id': os.environ.get('NPU_NVME_ENVIRONMENT_ID'),
        'parallel': {'tensor_parallel': 4, 'data_parallel': 1, 'pipeline_parallel': 1},
        'status': 'running', 'losses': [],
        'phases': {name: 'not_run' for name in (
            'config', 'communication', 'model_construct', 'load', 'train',
            'checkpoint', 'restart', 'oracle_compare')},
    }
    stage = 'config'
    write_report(report_path, report)
    try:
        import numpy as np
        import mindspore as ms
        import mindformers
        from mindformers import Trainer, build_context
        from mindformers.tools.register import MindFormerConfig
        from mindformers.core.context import is_legacy_model
        from mindspore.communication import get_rank, get_group_size
        from transformers import AutoTokenizer

        source = Path(args.model)
        raw_config = (source / 'config.json').read_bytes()
        model_info = json.loads(raw_config)
        if (model_info.get('model_type'), model_info.get('hidden_size'),
                model_info.get('num_hidden_layers')) != ('qwen3', 4096, 36):
            raise ValueError('Expected the full Qwen3-8B architecture')
        index = json.loads((source / 'model.safetensors.index.json').read_text())
        shards = sorted(set(index['weight_map'].values()))
        report['weights'] = [{'name': name, 'bytes': (source / name).stat().st_size} for name in shards]
        report['config_sha256'] = hashlib.sha256(raw_config).hexdigest()
        report['versions'] = {'mindspore': ms.__version__, 'mindformers': mindformers.__version__}
        template = Path(mindformers.__file__).parent.parent / 'configs/qwen3/finetune_qwen3.yaml'
        config = MindFormerConfig(str(template))
        config.output_dir = str(out / 'training')
        config.pretrained_model_dir = str(source)
        config.load_checkpoint = str(source)
        config.use_legacy = False
        config.run_mode = 'finetune'
        config.seed = 42
        config.parallel_config.update(data_parallel=1, model_parallel=4, pipeline_stage=1,
                                      micro_batch_num=1, vocab_emb_dp=False, use_seq_parallel=True)
        config.context.ascend_config.pop('parallel_speed_up_json_path', None)
        config.parallel.strategy_ckpt_config.save_file = str(rank_dir / 'strategy.ckpt')
        config.model.model_config.update(seq_length=args.seq_length, batch_size=1,
                                         params_dtype='float32', compute_dtype='bfloat16',
                                         residual_dtype='float32', input_sliced_sig=True)
        total_steps = args.warmup_steps + args.steps
        config.runner_config.update(epochs=1, sink_mode=True, sink_size=1, stop_step=total_steps)
        config.lr_schedule.total_steps = total_steps
        config.train_dataset.data_loader = MindFormerConfig(type='GeneratorDataset')
        config.callbacks[1]['save_checkpoint_steps'] = total_steps
        report['precision'] = {'compute': 'bfloat16', 'parameters_and_adam': 'float32'}
        report['phases']['config'] = 'pass'
        stage = 'communication'
        build_context(config)
        if is_legacy_model() or get_group_size() != 4 or get_rank() != rank:
            raise RuntimeError('Qwen3 requires non-legacy context and exactly four ranks')
        report['world_size'] = get_group_size()
        report['device_id'] = ms.get_context('device_id')
        report['phases']['communication'] = 'pass'
        write_report(report_path, report)
        print(f'QWEN_COMM_READY rank={rank} device={report["device_id"]} python={sys.executable}', flush=True)
        ms.set_seed(42)
        np.random.seed(42)

        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
        text = ('Checkpoint recovery restores model parameters and optimizer state. '
                'Tensor parallel training divides the model across accelerator devices. ')
        tokens = tokenizer.encode(text * (args.seq_length // 8 + 2), add_special_tokens=False)
        input_ids = np.asarray(tokens[:args.seq_length + 1], dtype=np.int32)
        if input_ids.size != args.seq_length + 1:
            raise RuntimeError('Tokenized fixture is too short')
        report['data_sha256'] = hashlib.sha256(input_ids.tobytes()).hexdigest()
        report['data_kind'] = 'fixed local tokenized text, smoke training only'
        row = (input_ids[:-1], input_ids[1:], np.ones(args.seq_length, np.float32),
               np.arange(args.seq_length, dtype=np.int32),
               np.triu(np.ones((1, args.seq_length, args.seq_length), np.uint8), k=1))
        dataset = ms.dataset.GeneratorDataset([row] * total_steps,
                    column_names=['input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask'],
                    shuffle=False, num_parallel_workers=1).batch(1, drop_remainder=True)

        class Progress(ms.Callback):
            def on_train_step_end(self, run_context):
                cb = run_context.original_args()
                values = cb.net_outputs
                loss = values[0] if isinstance(values, (list, tuple)) else values
                loss = float(np.asarray(loss.asnumpy()).mean())
                if not np.isfinite(loss):
                    raise RuntimeError(f'Non-finite training loss: {loss}')
                overflow = bool(np.asarray(values[1].asnumpy()).any()) if isinstance(values, (list, tuple)) else False
                if overflow:
                    raise RuntimeError('Optimizer skipped update due to overflow')
                report['phases'].update(model_construct='pass', load='pass', train='running')
                report['losses'].append({'step': int(cb.cur_step_num), 'loss': loss, 'overflow': overflow})
                write_report(report_path, report)
                print(f'QWEN_TRAIN_STEP rank={rank} step={cb.cur_step_num} loss={loss:.8f}', flush=True)

        stage = 'model_load_train'
        trainer = Trainer(args=config, train_dataset=dataset, callbacks=[Progress()])
        trainer.finetune(resume_from_checkpoint=str(source), auto_trans_ckpt=True)
        if len(report['losses']) != total_steps:
            raise RuntimeError('Training returned without completing requested steps')
        report['phases']['train'] = 'pass'
        report['status'] = 'training_pass_restart_not_tested'
        report['checkpoint_files'] = [str(p.relative_to(out)) for p in (out / 'training').rglob('*.safetensors')]
    except Exception as error:
        report['status'] = 'failed'
        report['failed_stage'] = stage
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
        if stage in report['phases']:
            report['phases'][stage] = 'fail'
        write_report(report_path, report)
        traceback.print_exc()
        return 1
    write_report(report_path, report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
