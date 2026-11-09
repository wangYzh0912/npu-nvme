"""Observation-only vLLM worker: launch scopes and lightweight task profiler."""
import json,os,time,functools
from pathlib import Path
import torch
import torch_npu
from vllm_ascend.worker.worker import NPUWorker
from aic_activity.phases import launch_phase

class ActivityWorker(NPUWorker):
    def init_device(self):
        super().init_device()
        self.activity_records=[]
        runner=self.model_runner
        original=runner.execute_model
        @functools.wraps(original)
        def execute(scheduler_output,*args,**kwargs):
            # Newly scheduled requests are explicitly prefill. Existing state
            # records how many prompt tokens remain; mixed batches stay mixed.
            phase=launch_phase(scheduler_output,runner.requests)
            begin=time.time_ns()
            with torch.profiler.record_function('AIC_ACTIVITY::'+phase):
                result=original(scheduler_output,*args,**kwargs)
            self.activity_records.append(dict(stage=phase,begin_ns=begin,end_ns=time.time_ns(),scheduled_tokens=int(scheduler_output.total_num_scheduled_tokens)))
            return result
        runner.execute_model=execute
        original_sample=runner._sample
        @functools.wraps(original_sample)
        def sample(*args,**kwargs):
            begin=time.time_ns()
            with torch.profiler.record_function('AIC_ACTIVITY::token_selection'):
                result=original_sample(*args,**kwargs)
            self.activity_records.append(dict(stage='token_selection',begin_ns=begin,end_ns=time.time_ns()))
            return result
        runner._sample=sample
        self.activity_profiler=None

    def activity_start(self,output):
        self.activity_records=[]
        self.activity_output=Path(output);self.activity_output.mkdir(parents=True,exist_ok=True)
        level=int(os.environ.get('AIC_ACTIVITY_LEVEL','1'))
        config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=(torch_npu.profiler.ProfilerLevel.Level1 if level else torch_npu.profiler.ProfilerLevel.Level0),
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,msprof_tx=False,
            l2_cache=False,op_attr=False,data_simplification=False)
        self.activity_profiler=torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            with_stack=False,record_shapes=False,profile_memory=False,experimental_config=config,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.activity_output),worker_name=f'rank{self.rank}'))
        self.activity_profiler.start()

    def activity_configure(self,output):
        if self.activity_profiler is not None:raise RuntimeError('cannot reconfigure an active profiler')
        os.environ['AIC_ACTIVITY_OUTPUT']=output
        self.activity_records=[]

    def activity_stop(self):
        if self.activity_profiler is None:raise RuntimeError('activity profiler not started')
        self.activity_profiler.stop();self.activity_profiler=None
        self.activity_flush()

    def activity_flush(self):
        out=Path(os.environ['AIC_ACTIVITY_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
        (out/f'worker-{self.rank}-stages.json').write_text(json.dumps(self.activity_records,indent=2)+'\n')
        return dict(rank=self.rank,stages=len(self.activity_records),device=str(self.device))
