#!/bin/bash
# msprof wrapper for Phase 1a with configurable params
LABEL="$1"
INJECT="$2"
SINK="$3"
PROF_DIR="/home/user7/npu-nvme/output/profiling_vec/${LABEL}"

source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash

export GLOG_v=0
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0

exec /root/miniconda3/envs/ms_2.5/bin/python \
  -c "
import os, sys, time, json, math
REPO='/home/user7/npu-nvme'
sys.path.insert(0,os.path.join(REPO,'python'))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops
ms.set_recursion_limit(10000)

DEVICE_ID=1; SEQ_LEN=1024; SINK_SIZE=${SINK}; TOTAL_STEPS=20; EPOCHS=2

ms.context.set_context(mode=ms.GRAPH_MODE,device_target='Ascend',device_id=DEVICE_ID)
ms.common.set_seed(42)

from mindformers import AutoModel, AutoConfig
cfg=AutoConfig.from_pretrained('gpt2_xl')
cfg.seq_length=SEQ_LEN; cfg.max_position_embeddings=SEQ_LEN
model=AutoModel.from_config(cfg)

ds=ms.dataset.MindDataset(REPO+'/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord',shuffle=True)
ds=ds.batch(1,drop_remainder=True).take(TOTAL_STEPS)

opt=nn.AdamWeightDecay(model.trainable_params(),learning_rate=1e-5)
all_params=list(model.trainable_params())
n_total=len(all_params)
inject_params=${INJECT}
covered=all_params[:inject_params] if inject_params>0 else []
n_inject=len(covered)

num_groups=max(1,min(int(math.ceil(n_inject/100)),10)) if n_inject>0 else 0
param_groups=[]; fp16_needed=[]
if n_inject>0:
    gs=max(1,int(math.ceil(n_inject/num_groups)))
    for g in range(num_groups):
        s=g*gs; e=min(s+gs,n_inject)
        if s<n_inject:
            pg=covered[s:e]
            param_groups.append(pg)
            fp16_needed.append([hasattr(p,'dtype') and p.dtype!=ms.float16 for p in pg])

total_elems=sum(int(np.prod(p.shape)) for p in covered) if inject_params else 0
print('[{label}] Total={} params, Inject={}, {:.2f}B elems, {} groups'.format(
    n_total,n_inject,total_elems/1e9,num_groups),flush=True)

class ProfiledCell(nn.Cell):
    def __init__(self,network,optimizer,param_groups,fp16_needed,inject):
        super().__init__(auto_prefix=False)
        self.network=network; self.network.set_grad()
        self.optimizer=optimizer
        self.grad_fn=ops.value_and_grad(self.network,grad_position=None,weights=self.optimizer.parameters)
        self.depend=ops.Depend()
        self.param_groups=param_groups; self.fp16_needed=fp16_needed
        self.inject=inject
    def construct(self,*inputs):
        loss,grads=self.grad_fn(*inputs)
        if self.inject:
            acc=Tensor([0.0],dtype=ms.float16)
            for gi,group in enumerate(self.param_groups):
                flags=self.fp16_needed[gi]
                flat_parts=[]
                for pi,p in enumerate(group):
                    pv=ops.Cast()(p,ms.float16) if flags[pi] else p
                    flat_parts.append(ops.Reshape()(pv,(-1,)))
                flat=flat_parts[0] if len(flat_parts)==1 else ops.Concat()(tuple(flat_parts))
                delta=ops.Sub()(flat,ops.ZerosLike()(flat))
                red=ops.ReduceSum()(delta)
                c32=ops.Cast()(red,ms.float32)
                c16=ops.Cast()(c32,ms.float16)
                acc=ops.Add()(acc,c16)
            loss=self.depend(loss,acc)
        opt_res=self.optimizer(grads)
        return self.depend(loss,opt_res)

t_build=time.perf_counter()
cell=ProfiledCell(model,opt,param_groups,fp16_needed,inject_params>0)
ms_model=ms.Model(cell)
build_s=time.perf_counter()-t_build
print('[{label}] Build={:.1f}s'.format(build_s),flush=True)

epoch_times_ms=[]
class CB(ms.Callback):
    def on_train_epoch_begin(self,rc):
        self.t0=time.perf_counter()
    def on_train_epoch_end(self,rc):
        epoch_times_ms.append((time.perf_counter()-self.t0)*1000)

print('[{label}] Starting {} steps (sink_size={})...'.format(TOTAL_STEPS,SINK_SIZE),flush=True)
t_total=time.perf_counter()
compiled_ok=True; error_msg=None
try:
    ms_model.train(epoch=EPOCHS,train_dataset=ds,callbacks=[CB()],dataset_sink_mode=True,sink_size=SINK_SIZE)
except Exception as e:
    compiled_ok=False; error_msg=str(e)[:300]
    print('[{label}] FAILED: {}'.format(error_msg),flush=True)

total_s=time.perf_counter()-t_total
compile_epoch=epoch_times_ms[0] if epoch_times_ms else 0
warm_epochs=epoch_times_ms[1:] if len(epoch_times_ms)>1 else []
avg_step=sum(warm_epochs)/len(warm_epochs)/SINK_SIZE if warm_epochs else 0
print('[{label}] compile={:.0f}ms warm_epochs={} avg_step={:.0f}ms total_s={:.1f}s'.format(
    compile_epoch,[round(e,0) for e in warm_epochs],avg_step,total_s),flush=True)

result={'test':'$LABEL','total_params':n_total,'inject_params':inject_params,
    'inject_elems_B':round(total_elems/1e9,3),'num_groups':num_groups,
    'sink_size':SINK_SIZE,'total_steps':TOTAL_STEPS,'epochs':EPOCHS,
    'compiled_ok':compiled_ok,'error':error_msg,
    'build_s':round(build_s,1),'total_wall_s':round(total_s,1),
    'compile_epoch_ms':round(compile_epoch,0),
    'warm_epochs_ms':[round(et,0) for et in warm_epochs],
    'avg_step_ms':round(avg_step,1)}

os.makedirs(REPO+'/experiments/output',exist_ok=True)
with open(REPO+'/experiments/output/phase1a_$LABEL.json','w') as f:
    json.dump(result,f,indent=2)
print('[$LABEL] DONE. -> phase1a_$LABEL.json',flush=True)
" --label "${LABEL}" --inject "${INJECT}"
