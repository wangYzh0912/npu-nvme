#!/usr/bin/env python3
"""Small offline vLLM-Ascend workload with explicit prefill/decode phases."""
import json
import os
import time
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    out = Path(os.environ.get('AIC_AIV_INFER_OUTPUT', '/output'))
    out.mkdir(parents=True, exist_ok=True)
    model_path='/models/Qwen3-8B'
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    task_profile=os.environ.get('AIC_AIV_TASK_PROFILE','1')=='1'
    profiler_config=({'profiler':'torch','torch_profiler_dir':str(out/'torch-profiler'),
                      'torch_profiler_with_stack':False} if task_profile else None)
    model = LLM(model=model_path, tensor_parallel_size=4, dtype='bfloat16',
                max_model_len=640, max_num_seqs=8, enforce_eager=True,
                trust_remote_code=True, gpu_memory_utilization=0.70,
                enable_prefix_caching=False,profiler_config=profiler_config)
    prefill_text = ('Explain how matrix multiplication and vector operations interact in a transformer model. '
                    'Describe attention, normalization, residual connections, and why efficient data movement matters. ')*12
    prefill = tokenizer.decode(tokenizer.encode(prefill_text,add_special_tokens=False)[:512])
    decode = 'Summarize AIC and AIV roles.'
    short_prompts = [f'Give one concise observation about workload {i}.' for i in range(8)]
    cases = (
        ('prefill_dominant', [prefill], SamplingParams(temperature=0.0,max_tokens=1)),
        ('decode_dominant', [decode], SamplingParams(temperature=0.0,max_tokens=32)),
        ('batch8_sampled', short_prompts, SamplingParams(temperature=0.8,top_p=0.95,max_tokens=32,seed=42)),
    )
    records = []
    for stage, batch, params in cases:
        if task_profile:model.start_profile(stage)
        start = time.time_ns()
        mono_start = time.monotonic_ns()
        outputs = model.generate(batch, params)
        mono_end = time.monotonic_ns()
        end = time.time_ns()
        if task_profile:model.stop_profile()
        records.append(dict(stage=stage, batch=len(batch), begin_ns=start, end_ns=end,
                            monotonic_begin_ns=mono_start, monotonic_end_ns=mono_end,
                            prompt_token_counts=[len(x.prompt_token_ids or []) for x in outputs],
                            prompt_tokens=sum(len(x.prompt_token_ids or []) for x in outputs),
                            output_tokens=sum(len(x.outputs[0].token_ids) for x in outputs)))
    (out / 'inference-stages.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records))


if __name__ == '__main__':
    main()
