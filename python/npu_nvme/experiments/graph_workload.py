"""No-I/O graph Top-K worker. All result files are written outside timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def restore_base(ms, network, directory, identity):
    import numpy as np
    from npu_nvme.experiments.initial_state import registry
    from npu_nvme.runtime.training_catalog import read_checked
    from qwen_native_state import restore_control
    manifest = read_checked(directory / 'manifest.json')
    if manifest['identity'] != identity:
        raise ValueError('initial FULL identity mismatch')
    parameters = registry(network)
    ms.runtime.synchronize()
    for name, entry in manifest['tensors'].items():
        parameter = parameters[name]
        array = np.load(directory / entry['file'], allow_pickle=False)
        if list(array.shape) != list(parameter.shape) or hashlib.sha256(memoryview(array).cast('B')).hexdigest() != entry['sha256']:
            raise ValueError('initial tensor mismatch: ' + name)
        parameter.set_data(ms.Tensor(array, dtype=parameter.dtype))
    restore_control(ms, network, manifest['control'])
    ms.runtime.synchronize()
    return dict(tensors=len(manifest['tensors']), identity=identity, verified_input_sha=True)


def run(options):
    import numpy as np
    import mindspore as ms
    import mindformers
    from mindformers import Trainer, build_context
    from mindformers.tools.register import MindFormerConfig
    from mindspore.communication import get_rank, get_group_size, create_group
    from mindspore.communication.comm_func import barrier
    from transformers import AutoTokenizer
    from npu_nvme.experiments.graph_chain import install_wrapper
    from npu_nvme.experiments.graph_geometry import summarize
    from npu_nvme.experiments.device_score import local_graph

    rank = int(os.environ.get('RANK_ID', '0'))
    directory = Path(options['output']) / f'rank_{rank}'
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(status='initializing', options=options, rank=rank, losses=[], warmup=[],
                  actual_overlap='unverified', parallel_label='requested_dependency_layout_only')
    report['framework_versions'] = dict(mindspore=ms.__version__, mindformers=mindformers.__version__)
    if ms.__version__ != '2.7.1' or not mindformers.__version__.startswith('1.7.'):
        raise RuntimeError('unvalidated framework version')
    holder = {}
    profiler = None
    try:
        source = Path(options['models'][options['role']])
        template = Path(mindformers.__file__).parent.parent / 'configs/qwen3/finetune_qwen3.yaml'
        config = MindFormerConfig(str(template))
        config.output_dir = str(Path(options['output']) / 'framework')
        config.pretrained_model_dir = str(source)
        config.load_checkpoint = str(source)
        config.use_legacy = False
        config.run_mode = 'finetune'
        config.seed = 42
        config.parallel_config.update(data_parallel=1, model_parallel=4, pipeline_stage=1,
                                      micro_batch_num=1, vocab_emb_dp=False, use_seq_parallel=True)
        config.context.ascend_config.pop('parallel_speed_up_json_path', None)
        config.context.deterministic = 'ON'
        config.parallel.strategy_ckpt_config.save_file = str(directory / 'strategy.ckpt')
        config.model.model_config.update(seq_length=options['seq_length'], batch_size=1,
            params_dtype='float32', compute_dtype='bfloat16', residual_dtype='float32', input_sliced_sig=True)
        total = options['compile_steps'] + options['warmup_steps'] + options['formal_steps']
        config.runner_config.update(epochs=1, sink_mode=True, sink_size=1, stop_step=total)
        config.lr_schedule.total_steps = 24
        config.train_dataset.data_loader = MindFormerConfig(type='GeneratorDataset')
        # Remove disk/log/timing callbacks; keep only our in-memory observer.
        config.callbacks = []
        write(directory / 'resolved-config.json', dict(config))
        build_context(config)
        if get_group_size() != 4 or get_rank() != rank:
            raise ValueError('TP4 required')
        if options.get('dump_graphs'):
            ms.set_context(save_graphs=1, save_graphs_path=str(directory / 'graphs'))
        if options['level'] >= 5:
            create_group('graph_topk_aux_tp4', [0, 1, 2, 3])
        ms.set_seed(42); ms.manual_seed(42); np.random.seed(42); random.seed(42)
        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
        sequence = options['seq_length']
        text = ('Checkpoint recovery restores model parameters and optimizer state. '
                'Tensor parallel training divides the model across accelerator devices. ')
        tokens = tokenizer.encode(text * (sequence // 8 + 2), add_special_tokens=False)
        ids = np.asarray(tokens[:sequence + 1], dtype=np.int32)
        report['data_sha256'] = hashlib.sha256(ids.tobytes()).hexdigest()
        row = (ids[:-1], ids[1:], np.ones(sequence, np.float32), np.arange(sequence, dtype=np.int32),
               np.triu(np.ones((1, sequence, sequence), np.uint8), k=1))
        dataset = ms.dataset.GeneratorDataset([row] * total,
            column_names=['input_ids', 'labels', 'loss_mask', 'position_ids', 'attention_mask'],
            shuffle=False, num_parallel_workers=1).batch(1, drop_remainder=True)
        schema = json.loads(Path(options['strategy']).read_text())
        install_wrapper(options, schema, rank, holder)
        initial = Path(options['initial_root']) / ('initial-full-' + options['role']) / f'rank_{rank}'
        identity = options['initial_identity'] + ':' + options['role']

        def drain():
            if 'chain' in holder:
                with local_graph(ms):
                    result = holder['chain'](ms.Tensor(0.0, ms.float32), ms.Tensor(True, ms.bool_))
                return result
            return None

        class Progress(ms.Callback):
            def on_train_begin(self, context):
                network = context.original_args().train_network
                holder['network'] = network
                report['parallel_mode'] = ms.get_auto_parallel_context('parallel_mode')
                if options.get('reference_only'):
                    import math
                    model_schema = holder.get('schema', schema)
                    names = {t['name'] for t in model_schema['tensors'] if t['role']=='model' and math.prod(t['global_shape']) >= options['block_elements']}
                    with local_graph(ms):
                        holder['reference_allocations'] = [ms.ops.mul(p, ms.Tensor(.99, ms.float32))
                            for p in holder['wrapper'].weights if p.name in names]
                    ms.runtime.synchronize()
                    report['reference_allocation_bytes'] = sum(int(p.size)*4 for p in holder['reference_allocations'])
                    report['allocation_control'] = 'resident reference tensors; never scanned during formal training'
                if 'chain' in holder:
                    holder['chain'].reset_reference()
                    drain()  # Compile the final drain graph before any measurement.
                    ms.runtime.synchronize()
                    report['geometry'] = summarize(holder['chain'].rows)
                    report['compute_control'] = dict(iterations=holder['chain'].minimal.iterations, working_elements=holder['chain'].minimal.sample_elements)
                report['status'] = 'compiling'
                write(directory / 'progress.json', report)

            def on_train_step_begin(self, context):
                self.begin_ns = time.monotonic_ns()

            def on_train_step_end(self, context):
                nonlocal profiler
                cb = context.original_args()
                step = int(cb.cur_step_num)
                outputs = cb.net_outputs
                loss = float(np.asarray(outputs[0].asnumpy()).mean())
                overflow = bool(np.asarray(outputs[1].asnumpy()).any())
                if overflow or not np.isfinite(loss):
                    raise ValueError('nonfinite loss or skipped update')
                row = dict(step=step, loss=loss, begin_ns=self.begin_ns, end_ns=time.monotonic_ns())
                boundary = options['compile_steps'] + options['warmup_steps']
                if step <= boundary:
                    report['warmup'].append(row)
                else:
                    report['losses'].append(row)
                if step == options['compile_steps']:
                    report['restore'] = restore_base(ms, holder['network'], initial, identity)
                    if 'chain' in holder:
                        holder['chain'].reset_reference()
                        drain()
                    ms.runtime.synchronize()
                    barrier()
                    report['status'] = 'common_post_restore_warmup'
                    write(directory / 'progress.json', report)
                if step == boundary:
                    ms.runtime.synchronize()
                    barrier()
                    if 'chain' in holder:
                        report['version_begin'] = int(holder['chain'].version.asnumpy())
                    if options.get('profile'):
                        from mindspore.profiler import ProfilerLevel, AicoreMetrics
                        profiler = ms.Profiler(start_profile=False, output_path=str(directory / 'profiler'),
                            profiler_level=ProfilerLevel.Level1, aic_metrics=AicoreMetrics.PipeUtilization,
                            hbm_ddr=True, data_simplification=False)
                        profiler.start()
                    report['status'] = 'measuring'
                    write(directory / 'progress.json', report)
                    ms.runtime.reset_peak_memory_stats()
                    report['memory_start_bytes'] = ms.runtime.memory_allocated()
                    report['begin_ns'] = time.monotonic_ns()
                if step == total:
                    report['training_end_ns'] = time.monotonic_ns()
                    if options['layout'] == 'parallel':
                        drain()  # Diagnostic prologue/epilogue boundary, not accepted formal timing.
                    ms.runtime.synchronize()
                    report['all_done_ns'] = time.monotonic_ns()
                    report['memory_peak_bytes'] = ms.runtime.max_memory_allocated()
                    if profiler:
                        profiler.stop()
                    if 'chain' in holder:
                        report['version_end'] = int(holder['chain'].version.asnumpy())
                        if report['version_end'] - report['version_begin'] != options['formal_steps']:
                            raise ValueError('wrong number of graph auxiliary invocations')
                    report['status'] = 'timing_complete'
                    write(directory / 'progress.json', report)

        trainer = Trainer(args=config, train_dataset=dataset, callbacks=[Progress()])
        trainer.finetune(resume_from_checkpoint=str(source), auto_trans_ckpt=True)
        if profiler:
            profiler.analyse()
        if len(report['losses']) != options['formal_steps']:
            raise ValueError('missing formal steps')
        if 'chain' in holder:
            chain = holder['chain']
            # Small result arrays only; no full-state capture or storage readback.
            scores = chain.scores.asnumpy()
            indices = chain.indices.asnumpy()
            values = chain.values.asnumpy()
            if not np.all(np.isfinite(scores)):
                raise ValueError('nonfinite detection output')
            if options['level'] >= 6:
                if len(set(map(int, indices))) != chain.k or np.any(indices < 0) or np.any(indices >= chain.count):
                    raise ValueError('invalid Top-K index set')
                np.testing.assert_allclose(values, scores[indices], rtol=1e-6, atol=1e-6)
                threshold = float(np.partition(scores, -chain.k)[-chain.k])
                order_violations = int(np.count_nonzero(values[:-1] < values[1:]))
                report['topk_diagnostic'] = dict(sorted_requested=True,
                    order_violations=order_violations, minimum_selected=float(values.min()),
                    last_selected=float(values[-1]), cpu_kth_threshold=threshold,
                    threshold_gap=float(values.min() - threshold))
                if order_violations:
                    raise ValueError('Top-K sorted=True output is not descending: ' + repr(report['topk_diagnostic']))
                if values.min() < threshold - 1e-5:
                    raise ValueError('Top-K selected set is below CPU threshold: ' + repr(report['topk_diagnostic']))
            from npu_nvme.experiments.graph_validation import check_scores
            report['score_oracle'] = check_scores(ms, chain, options['level'])
            report['output_check'] = dict(status='indices_and_values_checked', score_count=len(scores),
                selected_count=len(indices), score_min=float(scores.min()), score_max=float(scores.max()),
                sampled_indices=indices[:16].tolist(), sampled_values=values[:16].tolist(),
                independent_score_oracle=report['score_oracle']['status'])
            times = []
            for _ in range(3):
                begin = time.monotonic_ns(); drain(); ms.runtime.synchronize()
                times.append(time.monotonic_ns() - begin)
            report['standalone_auxiliary_ns'] = times
        report['status'] = 'pass'
        report['formal_contract'] = 'twenty_updated_weight_detections'
        write(directory / 'result.json', report)
        return 0
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        write(directory / 'result.json', report)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    raise SystemExit(run(json.loads(parser.parse_args().config.read_text())))
