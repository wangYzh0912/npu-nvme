#!/usr/bin/env python3
"""P6 auxiliary-compute injection on a real GPT-2 XL training loop."""
from __future__ import annotations
import argparse, json, sys, threading, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT),str(ROOT/"python")]
from experiments.benchmarks.r0_real_e2e import build, train_one
from ppt_evidence import EvidenceBundle, environment_snapshot, stats, command

def cpu_chain(parameter, topk, task):
    value=np.asarray(parameter.asnumpy(),dtype=np.float32).reshape(-1)
    delta=value-value.mean()
    if task=="diff": return int(delta.size),int(delta.nbytes)
    blocks=np.add.reduceat(delta*delta,np.arange(0,value.size,65536))
    if task=="norm": return int(blocks.size),int(blocks.nbytes)
    chosen=np.argpartition(blocks,-min(topk,blocks.size))[-min(topk,blocks.size):]
    if task=="topk": return int(chosen.size),int(chosen.nbytes)
    encoded=value.astype(np.float16)
    return int(chosen.size),int(encoded.nbytes)

def npu_chain(ms, parameter, topk, task):
    from mindspore import ops
    value=ops.cast(parameter,ms.float32); delta=value-ops.mean(value)
    if task=="diff": ms.hal.synchronize(); return int(delta.size),int(delta.size)*4
    flat=ops.reshape(delta,(-1,)); take=min(int(flat.shape[0]),65536*max(topk,1))
    score=ops.abs(flat[:take])
    if task=="norm": ms.hal.synchronize(); return int(score.size),int(score.size)*4
    values,indices=ops.top_k(score,min(topk,take),sorted=False)
    if task=="topk": ms.hal.synchronize(); return int(indices.size),int(indices.size)*4
    encoded=ops.cast(values,ms.float16); ms.hal.synchronize()
    return int(indices.shape[0]),int(encoded.size)*2

def main():
    p=argparse.ArgumentParser(); p.add_argument("--modes",nargs="+",default=("none","cpu","npu_serial","npu_offset","npu_parallel")); p.add_argument("--tasks",nargs="+",default=("diff","norm","topk","fp16","full")); p.add_argument("--mode",choices=("none","cpu","npu_serial","npu_offset","npu_parallel"),default=None,help="run one configuration in an isolated process"); p.add_argument("--task",choices=("diff","norm","topk","fp16","full"),default=None,help="run one configuration in an isolated process"); p.add_argument("--seeds",nargs="+",type=int,default=(41,42,43)); p.add_argument("--warmups",type=int,default=10); p.add_argument("--steps",type=int,default=30); p.add_argument("--seq-len",type=int,default=129); p.add_argument("--topk",type=int,default=128); p.add_argument("--npu",type=int,default=7); p.add_argument("--pci",default="0000:83:00.0"); p.add_argument("--output-root",default=None); args=p.parse_args(); args.model="gpt2_xl"
    modes=(args.mode,) if args.mode else args.modes
    tasks=(args.task,) if args.task else args.tasks
    for seed in args.seeds:
      args.seed=seed
      for mode in modes:
       for task in tasks:
        bundle=EvidenceBundle("P6",{"model":"gpt2_xl","seed":seed,"mode":mode,"auxiliary":task,"warmups":args.warmups,"samples":args.steps},root=args.output_root,repo_root=ROOT,environment=environment_snapshot(pci=args.pci,npu=str(args.npu),repo_root=ROOT,npu_info=command(["npu-smi","info"])))
        values=[]; aux_values=[]
        try:
            ms,model,opt,cell=build(args); parameter=next(iter(model.trainable_params()))
            for index in range(args.warmups+args.steps):
                aux_ms=0.0; worker=None; error=[]
                def aux():
                    try:
                        (cpu_chain(parameter,args.topk,task) if mode=="cpu" else npu_chain(ms,parameter,args.topk,task))
                    except BaseException as exc: error.append(exc)
                if mode=="npu_offset":
                    t=time.perf_counter_ns(); aux(); aux_ms=(time.perf_counter_ns()-t)/1e6
                elif mode=="npu_parallel":
                    t=time.perf_counter_ns(); worker=threading.Thread(target=aux); worker.start()
                loss,train_ms=train_one(cell,ms,index+1,args.seq_len)
                if mode in ("cpu","npu_serial"):
                    t=time.perf_counter_ns(); aux(); aux_ms=(time.perf_counter_ns()-t)/1e6
                elif worker:
                    worker.join(); aux_ms=(time.perf_counter_ns()-t)/1e6
                if error: raise error[0]
                if index>=args.warmups:
                    values.append(train_ms+aux_ms); aux_values.append(aux_ms); bundle.add_sample({"status":"pass","step":index-args.warmups+1,"train_ms":train_ms,"aux_ms":aux_ms,"total_ms":train_ms+aux_ms,"loss":loss})
        except BaseException as exc: bundle.add_failure({"error":repr(exc)})
        result=bundle.finalize(metrics={"model":"gpt2_xl","seed":seed,"mode":mode,"latency_mean":stats(values),"foreground_wait":stats(aux_values),"auxiliary":task,"parallel_semantics":"separate host thread; profiler must confirm device overlap" if mode=="npu_parallel" else "serial"},status="pass" if len(values)==args.steps and not bundle.failures else "fail")
        print(json.dumps({"run_id":result["run_id"],"status":result["status"]},sort_keys=True),flush=True)
if __name__=="__main__": main()
