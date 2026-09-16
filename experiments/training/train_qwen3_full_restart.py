#!/usr/bin/env python3
"""Run real Qwen3 training; training success is not restart acceptance."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
import sys
import traceback


def write_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    process_begin_ns = time.monotonic_ns()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/models/Qwen3-8B')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--seq-length', type=int, default=128)
    parser.add_argument('--checkpoint-step',type=int)
    parser.add_argument('--source-stop-step',type=int)
    parser.add_argument('--lr-horizon',type=int)
    parser.add_argument('--resume-run',type=Path)
    parser.add_argument('--deterministic',action='store_true')
    parser.add_argument('--checkpoint-method',choices=['none','mindspore_native_save','ours','bytecheckpoint_host'],default='mindspore_native_save')
    parser.add_argument('--d2-connection',type=Path,help='explicit blocking D2 session; disables Native checkpoint writing')
    parser.add_argument('--d2-strategy',type=Path,default=Path(__file__).resolve().parents[2]/'config/qwen_runtime_schema.json')
    args = parser.parse_args()
    if args.d2_connection:args.checkpoint_method='ours'
    custom_checkpoint=args.checkpoint_method in ('ours','bytecheckpoint_host')
    if args.checkpoint_method=='none' and args.resume_run:parser.error('none has no restart')
    if args.steps < 1 or args.warmup_steps < 0 or args.seq_length < 8 or args.seq_length % 4:
        parser.error('steps must be positive; warmup nonnegative; sequence length >=8 and divisible by 4')
    total_steps = args.warmup_steps + args.steps if args.source_stop_step is None else args.source_stop_step
    checkpoint_step = args.warmup_steps + args.steps if args.checkpoint_step is None else args.checkpoint_step
    lr_horizon = total_steps if args.lr_horizon is None else args.lr_horizon
    if not 0 < checkpoint_step <= total_steps <= lr_horizon:
        parser.error('require 0 < checkpoint_step <= source_stop_step <= lr_horizon')
    if args.resume_run and total_steps <= checkpoint_step:
        parser.error('resume requires continuation steps after checkpoint')

    rank = int(os.environ.get('RANK_ID', '0'))
    out = Path(args.output).resolve()
    rank_dir = out / f'rank_{rank}'
    rank_dir.mkdir(parents=True, exist_ok=True)
    report_path = rank_dir / 'acceptance.json'
    report = {
        'model': args.model, 'rank': rank, 'pid': os.getpid(),
        'checkpoint_backend':{'ours':'d2','bytecheckpoint_host':'bytecheckpoint_host','none':'none','mindspore_native_save':'native'}[args.checkpoint_method],
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
        if args.resume_run:
            if custom_checkpoint:
                from qwen_d2_state import validate_restart
                validate_restart(args.resume_run,rank,checkpoint_step,lr_horizon)
            else:
                from qwen_native_state import validate_restart_contract
                validate_restart_contract(args.resume_run, rank, checkpoint_step, lr_horizon)
        import numpy as np
        import mindspore as ms
        import mindformers
        from mindformers import Trainer, build_context
        from mindformers.tools.register import MindFormerConfig
        from mindformers.core.context import is_legacy_model
        from mindspore.communication import get_rank, get_group_size
        from transformers import AutoTokenizer
        from npu_nvme.runtime.resources import framework_memory_snapshot
        from qwen_native_state import (parameter_manifest, capture_control, restore_control,
                                       compare_manifests, ready_barrier)

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
        config.context.deterministic = 'ON' if args.deterministic else 'OFF'
        report['deterministic'] = args.deterministic
        config.parallel.strategy_ckpt_config.save_file = str(rank_dir / 'strategy.ckpt')
        config.model.model_config.update(seq_length=args.seq_length, batch_size=1,
                                         params_dtype='float32', compute_dtype='bfloat16',
                                         residual_dtype='float32', input_sliced_sig=True)
        config.runner_config.update(epochs=1, sink_mode=True, sink_size=1, stop_step=total_steps)
        config.lr_schedule.total_steps = lr_horizon
        config.train_dataset.data_loader = MindFormerConfig(type='GeneratorDataset')
        config.callbacks[1]['save_checkpoint_steps'] = checkpoint_step
        config.callbacks[1]['keep_checkpoint_max'] = 3
        report['schedule'] = dict(checkpoint_step=checkpoint_step,source_stop_step=total_steps,
                                  lr_horizon=lr_horizon,resume_start_step=checkpoint_step if args.resume_run else 0)
        if args.checkpoint_method!='mindspore_native_save' or args.resume_run:
            config.callbacks=[c for c in config.callbacks if c.get('type')!='CheckpointMonitor']
            if args.resume_run and custom_checkpoint:config.runner_config.stop_step=total_steps-checkpoint_step
        else:
            from mindformers.core.callback import CheckpointMonitor
            from mindformers.tools.register import MindFormerRegister,MindFormerModuleType
            from qwen_native_state import fixed_step_checkpoint_monitor
            monitor=fixed_step_checkpoint_monitor(CheckpointMonitor)
            MindFormerRegister.register_cls(monitor,MindFormerModuleType.CALLBACK)
            for callback in config.callbacks:
                if callback.get('type')=='CheckpointMonitor':
                    callback.update(type='FixedStepCheckpointMonitor',async_save=False)
        if args.resume_run and not custom_checkpoint:
            config.load_checkpoint = str(args.resume_run / 'training/checkpoint')
            config.resume_training = f'qwen3_rank_0-{checkpoint_step}_1.safetensors'
            config.auto_trans_ckpt = False
        write_report(rank_dir/'resolved_config.json',dict(config))
        report['precision'] = {'compute': 'bfloat16', 'parameters_and_adam': 'float32'}
        report['phases']['config'] = 'pass'
        stage = 'communication'
        if args.resume_run:
            if custom_checkpoint:
                from qwen_d2_state import validate_identity
                validate_identity(args.resume_run,dict(config),report['config_sha256'],remaining_steps=total_steps-checkpoint_step)
            else:
                from qwen_native_state import validate_training_identity
                validate_training_identity(args.resume_run, dict(config), report['config_sha256'])
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
        random.seed(42)

        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
        text = ('Checkpoint recovery restores model parameters and optimizer state. '
                'Tensor parallel training divides the model across accelerator devices. ')
        tokens = tokenizer.encode(text * (args.seq_length // 8 + 2), add_special_tokens=False)
        input_ids = np.asarray(tokens[:args.seq_length + 1], dtype=np.int32)
        if input_ids.size != args.seq_length + 1:
            raise RuntimeError('Tokenized fixture is too short')
        report['data_sha256'] = hashlib.sha256(input_ids.tobytes()).hexdigest()
        np.save(rank_dir/'input_ids.npy',input_ids,allow_pickle=False)
        report['data_kind'] = 'fixed local tokenized text, smoke training only'
        row = (input_ids[:-1], input_ids[1:], np.ones(args.seq_length, np.float32),
               np.arange(args.seq_length, dtype=np.int32),
               np.triu(np.ones((1, args.seq_length, args.seq_length), np.uint8), k=1))
        dataset = ms.dataset.GeneratorDataset([row] * total_steps,
                    column_names=['input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask'],
                    shuffle=False, num_parallel_workers=1).batch(1, drop_remainder=True)

        report['resource_samples']=[]
        def record_memory(phase):
            sample=framework_memory_snapshot(ms,rank=rank,phase=phase)
            report['resource_samples'].append(sample)
            write_report(rank_dir/'resources.json',report['resource_samples'])

        class Progress(ms.Callback):
            def on_train_begin(self, run_context):
                record_memory('train_begin')
                if not args.resume_run:
                    cb=run_context.original_args()
                    initial,small=parameter_manifest(cb.train_network)
                    write_report(rank_dir/'state-initial.json',initial)
                    write_report(rank_dir/'control-initial.json',capture_control(ms,step=0,data_sha256=report['data_sha256'],lr_horizon=lr_horizon,small=small))
                    return
                started=time.monotonic()
                cb=run_context.original_args()
                source_rank=args.resume_run/f'rank_{rank}'
                expected_control=json.loads((source_rank/'control-checkpoint.json').read_text())
                def verify(control,step):
                    if control!=expected_control or step!=checkpoint_step:
                        raise ValueError('restored D2 controls differ from source')
                    if (control['logical_optimizer_step']!=checkpoint_step or
                        control['data_sha256']!=report['data_sha256'] or control['lr_horizon']!=lr_horizon):
                        raise ValueError('resume step/data/LR contract mismatch')
                    restore_control(ms,cb.train_network,control)
                    ms.hal.synchronize()
                    actual,small=parameter_manifest(cb.train_network)
                    expected=json.loads((source_rank/'state-checkpoint.json').read_text())
                    compare_manifests(expected,actual)
                    readback=capture_control(ms,step=checkpoint_step,data_sha256=report['data_sha256'],lr_horizon=lr_horizon,small=small)
                    readback=json.loads(json.dumps(readback))
                    if readback!=control:raise ValueError('restored control/RNG readback differs')
                    write_report(rank_dir/'restored-control.json',readback)
                    write_report(rank_dir/'restored-state.json',actual)
                if args.checkpoint_method=='bytecheckpoint_host':
                    from qwen_host_checkpoint import checkpoint
                    from mindspore.communication.comm_func import barrier
                    result=checkpoint(ms,cb.train_network,operation='restore',rank=rank,out=out,source=args.resume_run,verify=verify,barrier=barrier)
                    write_report(rank_dir/'d2-restore.json',result)
                elif args.d2_connection:
                    from npu_nvme.framework.d2_client import checkpoint_session
                    from mindspore.communication.comm_func import barrier
                    result=checkpoint_session(ms,cb.train_network,operation='restore',rank=rank,npu_id=rank,
                        connection=args.d2_connection,library=os.environ['NPU_NVME_LIBRARY_PATH'],
                        partitions=json.loads((source_rank/'d2-partitions.json').read_text()),
                        verify_controls=verify,collective_barrier=barrier)
                    write_report(rank_dir/'d2-restore.json',result)
                else:verify(expected_control,checkpoint_step)
                report['phases']['restart']='pass'
                report['restore_seconds']=time.monotonic()-started
                record_memory('restore_verified')
                write_report(report_path,report)
                ready_barrier(out,rank)
                write_report(rank_dir/'restore-timing.json',dict(
                    begin_ns=process_begin_ns,global_ready_ns=time.monotonic_ns(),
                    boundary='restore_begin_to_global_ready',mandatory_integrity=True,
                    includes='framework import, model construction, load, state/control verification, global ready barrier'))

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
                logical_step=int(cb.cur_step_num) + (checkpoint_step if args.resume_run else 0)
                report['losses'].append({'step': logical_step, 'callback_step': int(cb.cur_step_num), 'loss': loss, 'overflow': overflow})
                if logical_step == checkpoint_step or logical_step == total_steps:
                    ms.hal.synchronize()
                    manifest,small=parameter_manifest(cb.train_network)
                    label='checkpoint' if logical_step==checkpoint_step else 'final'
                    write_report(rank_dir/f'state-{label}.json',manifest)
                    control=capture_control(ms,step=logical_step,data_sha256=report['data_sha256'],
                                            lr_horizon=lr_horizon,small=small)
                    write_report(rank_dir/f'control-{label}.json',control)
                    if custom_checkpoint and logical_step==checkpoint_step:
                        from qwen_d2_state import partitions_for
                        from npu_nvme.framework.d2_client import checkpoint_session
                        partitions=partitions_for(manifest,small,args.d2_strategy)
                        write_report(rank_dir/'d2-partitions.json',partitions)
                        if args.checkpoint_method=='bytecheckpoint_host':
                            from qwen_host_checkpoint import checkpoint
                            result=checkpoint(ms,cb.train_network,operation='save',rank=rank,out=out,controls=control)
                        else:
                            result=checkpoint_session(ms,cb.train_network,operation='save',rank=rank,npu_id=rank,
                                connection=args.d2_connection,library=os.environ['NPU_NVME_LIBRARY_PATH'],
                                partitions=partitions,controls=control)
                        write_report(rank_dir/'d2-save.json',result)

                record_memory(f'step_{logical_step}')
                write_report(report_path, report)
                print(f'QWEN_TRAIN_STEP rank={rank} step={logical_step} loss={loss:.8f}', flush=True)

        stage = 'model_load_train'
        trainer = Trainer(args=config, train_dataset=dataset, callbacks=[Progress()])
        if args.resume_run and not custom_checkpoint:
            trainer.finetune(resume_from_checkpoint=str(args.resume_run/'training/checkpoint'),
                             resume_training=f'qwen3_rank_0-{checkpoint_step}_1.safetensors',auto_trans_ckpt=False)
        else:
            trainer.finetune(resume_from_checkpoint=str(source), auto_trans_ckpt=True)
        expected_steps=total_steps-checkpoint_step if args.resume_run else total_steps
        if len(report['losses']) != expected_steps:
            raise RuntimeError('Training returned without completing requested steps')
        record_memory('train_end')
        report['phases']['train'] = 'pass'
        report['status'] = 'restored_and_continued' if args.resume_run else 'training_pass_restart_not_tested'
        report['converted_weight_files'] = [str(p.relative_to(out)) for p in (out/'training/qwen3_ms_converted_weight').rglob('*.safetensors')]
        report['checkpoint_files'] = [str(p.relative_to(out)) for p in (out / 'training/checkpoint').rglob('*.safetensors')]
    except Exception as error:
        write_report(out/f'failed-rank-{rank}.json',dict(rank=rank,error=repr(error)))
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
