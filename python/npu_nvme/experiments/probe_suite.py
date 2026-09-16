"""Reset-separated traffic probes sharing one compiled training graph."""
import time
from pathlib import Path
from npu_nvme.experiments.injection import ProbeController
from npu_nvme.experiments.initial_state import restore
from npu_nvme.runtime.training_catalog import write_checked


class ProbeSuite:
    suite=True
    pending=None

    def __init__(self, ms, network, options, *, rank, output):
        self.ms=ms;self.network=network;self.options=options;self.rank=rank;self.output=Path(output)
        self.specs=options['probe']['specs'];self.index=-1;self.controller=None
        self.timings=[];self.losses=[];self.failed=False;self.restore_evidence=None;self.baseline=[]

    def warmup(self):self._start()

    def _start(self):
        import gc
        from mindspore.communication.comm_func import barrier
        self.index+=1;self.losses=[];self.timings=[];self.controller=None;gc.collect()
        begin=time.monotonic_ns()
        self.restore_evidence=restore(self.ms,self.network,Path(self.options['initial_full'])/f'rank_{self.rank}',identity=self.options['initial_identity'])
        self.restore_evidence['elapsed_ns']=time.monotonic_ns()-begin
        barrier()
        spec=self.specs[self.index];self.directory=self.output/'suite'/spec['name']
        if spec.get('probe'):
            options=dict(self.options,probe=spec['probe'])
            self.controller=ProbeController(self.ms,self.network,options,rank=self.rank,output=self.directory)
            self.controller.warmup()
        self.ms.runtime.synchronize();barrier();self.ms.runtime.reset_peak_memory_stats()
        self.memory_start=self.ms.runtime.memory_allocated();self.begin=time.monotonic_ns()

    def observe(self, row):
        if self.index==0:self.baseline.append(row['loss'])
        elif row['loss']!=self.baseline[len(self.losses)]:
            raise ValueError('suite numerical trajectory changed after FULL reset')
        self.losses.append(dict(row,logical_step=len(self.losses)+1))

    def save(self, logical_step):
        local=(logical_step-1)%20+1
        if len(self.losses)!=local:raise ValueError('suite local optimizer step mismatch')
        row=self.controller.save(local) if self.controller else dict(step=local,payload_bytes=0,pending_requests=0)
        self.timings.append(row)
        if local==20:
            end=time.monotonic_ns();peak=self.ms.runtime.max_memory_allocated()
            drain=self.controller.close() if self.controller else 0
            done=time.monotonic_ns();spec=self.specs[self.index]
            report=dict(status='pass',rank=self.rank,losses=self.losses,initial_full_restore=self.restore_evidence,
                parent_run=str(self.output),suite_slot=self.index,
                measurement_scope='One configuration per reset-separated 20-step interval; shared process/compiler, not independent repeated runs.',
                incremental=dict(group='B0',warmup_steps=0,formal_begin_ns=self.begin,formal_end_ns=end,
                    all_done_ns=done,drain_ns=drain,steps=self.timings,memory_peak_bytes=peak,
                    memory_start_bytes=self.memory_start,pending_at_formal_end=0,suite=True))
            write_checked(self.directory/f'rank_{self.rank}'/'training.json',report)
            self.controller=None
            if self.index+1<len(self.specs):self._start()
        return row

    def close(self):
        if self.controller:self.controller.close();self.controller=None
        if self.index+1!=len(self.specs) or len(self.losses)!=20:
            raise ValueError('suite ended before all configurations completed')
        return 0
