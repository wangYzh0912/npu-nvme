"""One independent model process for capacity or work-scaling observation."""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--blocks', type=int)
    p.add_argument('--cases', default='1:128:32')
    p.add_argument('--metadata-only', action='store_true')
    p.add_argument('--system-sample', action='store_true')
    p.add_argument('--model', default='/models/Qwen3-8B')
    p.add_argument('--max-model-len', type=int, default=640)
    p.add_argument('--graph-mode', choices=['DEFAULT','PIECEWISE','FULL_AND_PIECEWISE','FULL_DECODE_ONLY','NONE'], default='PIECEWISE')
    p.add_argument('--pa-shapes', default='')
    p.add_argument('--layout', choices=['native','dense'], default='native')
    p.add_argument('--metadata-probe', action='store_true')
    p.add_argument('--trace-requests',type=int,default=6)
    p.add_argument('--control-requests',type=int,default=6)
    p.add_argument('--probe-graph-shapes',action='store_true')
    p.add_argument('--worker',default='aic_kv.worker.KVWorker')
    p.add_argument('--trace-cases',default='all')
    p.add_argument('--metadata-cases',default='all')
    p.add_argument('--prompt-fixture',choices=['repeated','diverse'],default='repeated')
    p.add_argument('--request-metrics',action='store_true')
    p.add_argument('--sync-scheduling',action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'run-config.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
          model=args.model,tensor_parallel_size=4,dtype='bfloat16',max_model_len=args.max_model_len,
          max_num_seqs=8,max_num_batched_tokens=4096,enable_prefix_caching=False,
          gpu_memory_utilization=.55,seed=42,graph_mode=args.graph_mode))
    os.environ['AIC_ACTIVITY_OUTPUT'] = str(args.output)
    os.environ['AIC_ACTIVITY_LEVEL'] = '1'
    os.environ['AIC_KV_LAYOUT'] = args.layout
    from vllm import LLM, SamplingParams
    kwargs = dict(num_gpu_blocks_override=args.blocks) if args.blocks is not None else {}
    if args.sync_scheduling:kwargs['async_scheduling']=False
    if args.graph_mode!='DEFAULT':kwargs['compilation_config']={'cudagraph_mode':args.graph_mode}
    if args.probe_graph_shapes:
        kwargs.setdefault('compilation_config',{})['cudagraph_capture_sizes']=[1,8]
    if args.pa_shapes:
        kwargs['additional_config']={'pa_shape_list':[int(x) for x in args.pa_shapes.split(',')]}
    model = LLM(model=args.model, tensor_parallel_size=4, dtype='bfloat16',
                max_model_len=args.max_model_len, max_num_seqs=8, max_num_batched_tokens=4096,
                enable_prefix_caching=False, gpu_memory_utilization=.55, seed=42,
                worker_cls=args.worker,
                disable_log_stats=not args.request_metrics,
                **kwargs)
    write(args.output/'source-identity.json', model.collective_rpc('kv_source_identity'))
    for spec in args.cases.split(','):
        batch, length, generated = map(int, spec.split(':'))
        case = args.output/f'b{batch}-l{length}-o{generated}'
        case.mkdir()
        os.environ['AIC_ACTIVITY_OUTPUT'] = str(case)
        model.collective_rpc('activity_configure', args=(str(case),))
        tokens = model.get_tokenizer().encode(
            'Matrix and vector engines execute transformer layers, normalization, and attention. '*200,
            add_special_tokens=False)[:length]
        if len(tokens) != length:
            raise RuntimeError('input fixture length mismatch')
        prompts=[tokens for _ in range(batch)]
        if args.prompt_fixture=='diverse':
            questions=('What is the capital of France?','What is 2 plus 3?',
                'Name the largest planet in the solar system.','Translate hello into Spanish.',
                'What color is a ripe banana?','List the first three prime numbers.',
                'What is the capital of Japan?','Write the word zebra exactly once.')
            prompts=[]
            for index in range(batch):
                tail=model.get_tokenizer().encode('\nQuestion '+str(index)+': '+questions[index%len(questions)]+'\nAnswer:',add_special_tokens=False)
                if len(tail)>=length:raise ValueError('diverse prompt does not fit requested input length')
                prompts.append(tokens[:length-len(tail)]+tail)
            if len({tuple(p) for p in prompts})!=batch:raise RuntimeError('diverse request prompts are not distinct')
        write(case/'input-fixture.json',dict(kind=args.prompt_fixture,
            input_sha256=[hashlib.sha256(json.dumps(p).encode()).hexdigest() for p in prompts]))
        params = SamplingParams(temperature=0, max_tokens=generated, min_tokens=generated,
                                ignore_eos=True, seed=42)
        records = []
        def call(kind, index):
            begin = time.time_ns()
            mono = time.monotonic_ns()
            result = model.generate([dict(prompt_token_ids=prompt) for prompt in prompts], params, use_tqdm=False)
            duration = time.monotonic_ns()-mono
            end = time.time_ns()
            output_tokens = [r.outputs[0].token_ids for r in result]
            if len(result) != batch or any(len(t) != generated for t in output_tokens):
                raise RuntimeError('output token count mismatch')
            row = dict(kind=kind, index=index, begin_ns=begin, end_ns=end,
                       duration_ns=duration, output_sha256=hashlib.sha256(json.dumps(output_tokens).encode()).hexdigest(),
                       output_token_ids=output_tokens)
            if args.request_metrics:
                metrics=[]
                for request in result:
                    stat=request.metrics
                    if stat is None or stat.first_token_ts<=0 or stat.last_token_ts<stat.first_token_ts:
                        raise RuntimeError('requested engine token timestamps missing or invalid')
                    metrics.append(dict(ttft_ms=stat.first_token_latency*1000,
                        decode_ms=(stat.last_token_ts-stat.first_token_ts)*1000,
                        mean_inter_token_ms=(stat.last_token_ts-stat.first_token_ts)*1000/(generated-1) if generated>1 else None,
                        prefill_ms=(stat.first_token_ts-stat.scheduled_ts)*1000))
                row['request_metrics']=metrics
                row['output_tokens_per_s']=batch*generated/(duration/1e9)
            records.append(row)
            write(case/'windows.json', records)
            print('KV_REQUEST', spec, kind, index, round(duration/1e6,3), flush=True)
        stable = False
        for i in range(20):
            call('warmup', i)
            recent = [r['duration_ns'] for r in records[-3:]]
            if i >= 4 and statistics.pstdev(recent)/statistics.mean(recent) <= .05:
                stable = True
                break
        if not stable:
            raise RuntimeError('warmup did not reach CV <= 5%')
        snapshot = model.collective_rpc('kv_snapshot')
        write(case/'kv-layout.json', snapshot)
        if args.blocks is not None and any(r['num_blocks'] != args.blocks for r in snapshot):
            raise RuntimeError('requested KV blocks do not match allocator')
        if args.metadata_only or (args.metadata_probe and (args.metadata_cases=='all' or case.name in args.metadata_cases.split(','))):
            model.collective_rpc('kv_probe_start')
            call('metadata-probe', 0)
            write(case/'probe-result.json', model.collective_rpc('kv_probe_stop', args=(str(case),)))
            if args.metadata_only:continue
        for i in range(args.control_requests):
            call('control-before', i)
        trace_enabled=args.trace_cases=='all' or case.name in args.trace_cases.split(',')
        if trace_enabled:
            model.collective_rpc('activity_start', args=(str(case/'trace'),))
            for i in range(args.trace_requests):
                call('task', i)
            model.collective_rpc('activity_stop')
        if args.system_sample:
            command = ['msprof', '--output='+str(case/'system'), '--sys-devices=4,5,6,7',
                       '--sys-period=36', '--ai-core=on', '--aic-mode=sample-based', '--aic-freq=100',
                       '--aic-metrics=PipeUtilization', '--task-time=off', '--sys-hardware-mem=on',
                       '--sys-hardware-mem-freq=50', '--storage-limit=500MB', '--type=text']
            write(case/'collector-command.json', command)
            with (case/'collector.log').open('w') as log:
                collector = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    time.sleep(2)
                    if collector.poll() is not None:
                        raise RuntimeError('system collector exited before measurement')
                    begin = time.monotonic()
                    i = 0
                    while i < 6 or time.monotonic()-begin < 30:
                        call('system', i)
                        i += 1
                    code = collector.wait(timeout=180)
                    if code:
                        raise RuntimeError(f'system collector exit {code}')
                finally:
                    if collector.poll() is None:
                        collector.terminate()
                        collector.wait(timeout=30)
        for i in range(args.control_requests):
            call('control-after', i)
        model.collective_rpc('activity_flush')
        controls = [r['duration_ns'] for r in records if r['kind'].startswith('control')]
        tasks = [r['duration_ns'] for r in records if r['kind']=='task']
        hashes = {r['output_sha256'] for r in records}
        if len(hashes) != 1:
            raise RuntimeError('output changed within one deterministic condition')
        baseline = statistics.median(controls)
        overhead = statistics.median(tasks)/baseline-1 if tasks else None
        write(case/'result.json', dict(status='pass', batch=batch, prompt=length, generated=generated,
              requested_blocks=args.blocks, actual_blocks=[r['num_blocks'] for r in snapshot],
              control_median_ms=baseline/1e6, task_overhead_fraction=overhead,
              control_cv=statistics.pstdev(controls)/statistics.mean(controls),
              output_sha256=next(iter(hashes)), task_quality='not_collected' if overhead is None else 'valid' if overhead<=.1 else 'diagnostic'))
    write(args.output/'completed.json', dict(status='pass'))


if __name__ == '__main__':
    main()
