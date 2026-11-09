"""Independent AIC matrix/AIV copy control; no model readiness is inferred."""
import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    import torch
    import torch_npu
    torch.npu.set_device(0);torch.manual_seed(42)
    left=torch.randn((4096,4096),device='npu',dtype=torch.bfloat16)
    right=torch.randn_like(left);out=torch.empty_like(left)
    backing=torch.randn((2048,2,128,256),device='npu',dtype=torch.bfloat16)
    view=backing[:,0];copied=None
    reference=torch.mm(left,right)
    matrix_stream=torch.npu.Stream();vector_stream=torch.npu.Stream()
    torch.npu.synchronize()
    def matrix():torch.mm(left,right,out=out)
    def copy():
        nonlocal copied
        copied=view.contiguous()
    def serial():matrix();copy()
    def parallel():
        with torch.npu.stream(matrix_stream):matrix()
        with torch.npu.stream(vector_stream):copy()
    modes={'matrix_only':matrix,'copy_only':copy,'serial_pair':serial,'parallel_pair':parallel}
    rows=[]
    for name,fn in modes.items():
        for _ in range(5):fn()
        torch.npu.synchronize();times=[]
        for _ in range(3):
            start=time.perf_counter_ns()
            for _ in range(30):fn()
            torch.npu.synchronize();times.append((time.perf_counter_ns()-start)/30/1e6)
        rows.append(dict(mode=name,mean_batch_ms=times,median_ms=statistics.median(times)))
    if not torch.equal(copied,view) or not torch.allclose(out,reference,rtol=.01,atol=.01):
        raise RuntimeError('co-run output mismatch')
    config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,data_simplification=False)
    with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            experimental_config=config,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.output/'trace'))) as profiler:
        for name,fn in modes.items():
            with torch.profiler.record_function('KV_CORUN::'+name):
                for _ in range(3):fn()
                torch.npu.synchronize()
    (a.output/'results.json').write_text(json.dumps(dict(results=rows,output_matches=True,
        contract='two independent synthetic operations; does not establish ready model work or co-scheduling speedup',
        matrix_shape=[4096,4096,4096],copy_shape=list(view.shape),copy_stride=list(view.stride()),dtype='bfloat16'),indent=2)+'\n')


if __name__=='__main__':main()
