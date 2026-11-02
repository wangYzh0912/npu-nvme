"""Fixed-token steady-state measurement with explicit controls and scopes."""
import argparse,json,time,statistics,os,hashlib,subprocess
from pathlib import Path
from vllm import LLM,SamplingParams

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--length',type=int,choices=[128,512],default=128);p.add_argument('--batch',type=int,choices=[1,8],default=1);p.add_argument('--sample',action='store_true');p.add_argument('--mode',choices=['control','task','pmu'],default='task');p.add_argument('--level',type=int,choices=[0,1],default=1);p.add_argument('--eager',action='store_true');p.add_argument('--piecewise',action='store_true');p.add_argument('--matrix',action='store_true',help='Five cases per independent model process; PMU selects the two short-prompt greedy cases');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);os.environ['AIC_ACTIVITY_OUTPUT']=str(a.output);os.environ['AIC_ACTIVITY_LEVEL']=str(a.level)
    graph_options={'compilation_config':{'cudagraph_mode':'PIECEWISE'}} if a.piecewise else {}
    if a.eager and a.piecewise:raise ValueError('choose one explicit graph mode')
    model=LLM(model='/models/Qwen3-8B',tensor_parallel_size=4,dtype='bfloat16',max_model_len=640,max_num_seqs=8,
              max_num_batched_tokens=4096,enable_prefix_caching=False,enforce_eager=a.eager,
              gpu_memory_utilization=.55,seed=42,worker_cls='aic_activity.worker.ActivityWorker',**graph_options)
    cases=([(128,1,False),(128,8,False)] if a.mode=='pmu' else [(128,1,False),(512,1,False),(128,8,False),(512,8,False),(128,8,True)]) if a.matrix else [(a.length,a.batch,a.sample)]
    results=[]
    for length,batch,sample in cases:
        output=a.output/f"b{batch}-l{length}-{'sample' if sample else 'greedy'}" if a.matrix else a.output
        output.mkdir(parents=True,exist_ok=True)
        os.environ['AIC_ACTIVITY_OUTPUT']=str(output)
        model.collective_rpc('activity_configure',args=(str(output),))
        results.append(measure(model,a,output,length,batch,sample))
    (a.output/'matrix-result.json').write_text(json.dumps(dict(status='pass',cases=results),indent=2)+'\n')

def measure(model,a,output,length,batch,sample):
    tokenizer=model.get_tokenizer();tokens=tokenizer.encode('Matrix and vector engines execute transformer layers, normalization, and attention. '*200,add_special_tokens=False)[:length]
    if len(tokens)!=length:raise ValueError('fixed token fixture too short')
    prompts=[dict(prompt_token_ids=tokens) for _ in range(batch)]
    params=SamplingParams(temperature=.8 if sample else 0,top_p=.95 if sample else 1,top_k=50 if sample else -1,max_tokens=32,min_tokens=32,ignore_eos=True,seed=42)
    records=[]
    def call(kind,index):
        begin=time.time_ns();mono=time.monotonic_ns();out=model.generate(prompts,params,use_tqdm=False);end_mono=time.monotonic_ns();end=time.time_ns()
        if len(out)!=batch or any(len(r.outputs[0].token_ids)!=32 or len(r.prompt_token_ids)!=length for r in out):raise RuntimeError('inference token count differs')
        records.append(dict(id=f'{kind}-{index}',kind=kind,begin_ns=begin,end_ns=end,duration_ns=end_mono-mono,
                            output_sha256=hashlib.sha256(json.dumps([r.outputs[0].token_ids for r in out]).encode()).hexdigest()))
        (output/'windows.json').write_text(json.dumps(records,indent=2)+'\n')
        print('ACTIVITY_REQUEST',records[-1],flush=True)
    stable=False
    for i in range(20):
        call('warmup',i)
        if i>=4:
            last=[x['duration_ns'] for x in records[-3:]]
            if statistics.pstdev(last)/statistics.mean(last)<=.05:stable=True;break
    if not stable:raise RuntimeError('inference warmup did not stabilize within 20 batches')
    for i in range(6):call('control-before',i)
    collector=None
    if a.mode=='task':model.collective_rpc('activity_start',args=(str(output/'trace'),))
    elif a.mode=='pmu':
        collectlog=(output/'collector.log').open('w')
        collector=subprocess.Popen(['msprof','--output='+str(output/'pmu'),'--sys-devices=4,5,6,7','--sys-period=35','--ai-core=on','--aic-mode=sample-based','--aic-freq=100','--aic-metrics=PipeUtilization','--task-time=off','--type=text'],stdout=collectlog,stderr=subprocess.STDOUT)
        time.sleep(2)
        if collector.poll() is not None:raise RuntimeError('sample-only collector exited before measurement')
    begin=time.monotonic()
    i=0
    while i<6 or (a.mode=='pmu' and time.monotonic()-begin<30):call('measured',i);i+=1
    if a.mode=='task':model.collective_rpc('activity_stop')
    else:model.collective_rpc('activity_flush')
    if collector is not None:
        code=collector.wait(timeout=180);collectlog.close()
        if code:raise RuntimeError(f'sample-only collector exited with {code}')
    for i in range(6):call('control-after',i)
    baseline=statistics.median(r['duration_ns'] for r in records if r['kind'].startswith('control-'))
    measured=statistics.median(r['duration_ns'] for r in records if r['kind']=='measured')
    result=dict(status='pass',args=dict({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},length=length,batch=batch,sample=sample,output=str(output)),warmup_stable=stable,
                control_median_ms=baseline/1e6,measured_median_ms=measured/1e6,overhead_fraction=measured/baseline-1,
                quality='valid' if measured/baseline<=1.1 else 'diagnostic_profiler_overhead',records=len(records))
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':main()
