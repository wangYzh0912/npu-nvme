"""Real AdamW update on recorded Qwen shard geometries; no model-ready claim."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('adamw-o0','adamw-o1','fused-o0'),required=True)
    p.add_argument('--shape',choices=('128','4096','4096x3072'),required=True)
    p.add_argument('--trace',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    lock=Path('/models/npu_nvme_exp/user7-stack/hardware-campaign.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lease=Path('/models/npu_nvme_exp/user7-stack/hardware-campaign.lease.json')
    if lease.exists():raise RuntimeError('training hardware lease is active')
    if 'No running processes found in NPU 4' not in subprocess.check_output(['npu-smi','info'],text=True):
        raise RuntimeError('physical NPU 4 is occupied')
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!='4':
        raise RuntimeError('fragment requires physical NPU 4 only')
    import numpy as np
    import mindspore as ms
    from mindformers.core.optim.adamw import AdamW
    from mindformers.core.optim.fused_adamw import FusedAdamW
    ms.set_context(mode=ms.GRAPH_MODE,device_target='Ascend',device_id=0,
                   jit_config={'jit_level':'O1' if args.mode=='adamw-o1' else 'O0'})
    ms.set_seed(42)
    shape=tuple(int(x) for x in args.shape.split('x'))
    rng=np.random.default_rng(42)
    weights=rng.normal(0,.02,shape).astype(np.float32)
    gradient=rng.normal(0,.001,shape).astype(np.float32)
    parameter=ms.Parameter(ms.Tensor(weights),name='recorded_qwen_shard')
    constructor=FusedAdamW if args.mode=='fused-o0' else AdamW
    opt=constructor([parameter],learning_rate=1e-6,betas=(.9,.95),eps=1e-8,weight_decay=0.)
    grads=(ms.Tensor(gradient),)
    # Complete-output oracle fixed before timings; no device work in NumPy.
    expected=weights.astype(np.float64);grad=gradient.astype(np.float64)
    moment=np.zeros(shape,np.float64);variance=np.zeros(shape,np.float64)
    for step in range(1,9):
        opt(grads);ms.runtime.synchronize()
        moment=.9*moment+.1*grad;variance=.95*variance+.05*grad*grad
        expected=expected-1e-6/(1-.9**step)*moment/(np.sqrt(variance/(1-.95**step))+1e-8)
    actual=parameter.asnumpy()
    actual_m=opt.exp_avg[0].asnumpy();actual_v=opt.exp_avg_sq[0].asnumpy()
    moment_ok=bool(np.allclose(actual_m,moment,rtol=1e-6,atol=1e-8))
    variance_ok=bool(np.allclose(actual_v,variance,rtol=1e-6,atol=1e-10))
    passed=bool(np.allclose(actual,expected,rtol=1e-6,atol=1e-7)) and moment_ok and variance_ok
    oracle=dict(passed=passed,rtol=1e-6,atol=1e-7,max_abs_error=float(np.max(np.abs(actual-expected))),
                input_sha256=hashlib.sha256(weights.tobytes()+gradient.tobytes()).hexdigest(),steps=8,
                moment_passed=moment_ok,variance_passed=variance_ok,
                moment_max_abs_error=float(np.max(np.abs(actual_m-moment))),
                variance_max_abs_error=float(np.max(np.abs(actual_v-variance))))
    np.savez_compressed(args.output/'oracle.npz',actual=actual,expected=expected.astype(np.float32))
    timings=[]
    for repeat in range(10):
        start=time.perf_counter_ns()
        for _ in range(40):opt(grads)
        ms.runtime.synchronize()
        timings.append((time.perf_counter_ns()-start)/1e6/40)
    # No reordering across independent model branches is attempted by this probe.
    result=dict(mode=args.mode,shape=shape,physical_device=4,oracle=oracle,
        timings_ms_per_update=timings,numerical_input='fixed finite synthetic values; actual shard geometry',
        selected_source='mindformers/core/optim/'+('fused_adamw.py' if args.mode=='fused-o0' else 'adamw.py'),
        model_ready_work_proven=False,external_io_in_timed_region=False)
    if args.trace:
        from mindspore.profiler import ProfilerLevel,AicoreMetrics,ProfilerActivity
        profiler=ms.Profiler(start_profile=False,output_path=str(args.output/'trace'),
            profiler_level=ProfilerLevel.Level0,aic_metrics=AicoreMetrics.AiCoreNone,
            activities=[ProfilerActivity.CPU,ProfilerActivity.NPU],
            hbm_ddr=False,data_simplification=False)
        profiler.start()
        for _ in range(8):opt(grads)
        ms.runtime.synchronize();profiler.stop();profiler.analyse()
        result['trace']='diagnostic path identification; excluded from unprofiled timings'
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+chr(10))
    print(json.dumps(result),flush=True)
    if not passed:raise RuntimeError('optimizer fragment numerical oracle failed')


if __name__=='__main__':main()
