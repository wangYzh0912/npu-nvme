"""Fixed-input auxiliary load for synchronized training boundary probes."""
import math
import time
import numpy as np

from npu_nvme.experiments.device_score import make_score_cell


class ScoreInjection:
    def __init__(self, ms, target_elements, factor, *, selection=False):
        self.ms=ms;self.selection=selection
        self.chunk=64*1024*1024
        self.elements=math.ceil(target_elements*factor)
        self.sizes=[min(self.chunk,self.elements-start) for start in range(0,self.elements,self.chunk)]
        self.inputs={};self.cells={}
        for size in set(self.sizes):
            a=np.ones(size,np.float32);b=np.zeros(size,np.float32)
            self.inputs[size]=(ms.Tensor(a),ms.Tensor(b));self.cells[size]=make_score_cell(ms,size,65536)
            self.cells[size](*self.inputs[size]).asnumpy()
        ms.runtime.synchronize()

    def run(self):
        start=time.monotonic_ns();scores=[]
        for size in self.sizes:scores.append(self.cells[size](*self.inputs[size]).asnumpy())
        score_end=time.monotonic_ns()
        combined=np.concatenate(scores)
        if self.selection:
            count=max(1,math.ceil(len(combined)*.1));indices=np.argpartition(combined,-count)[-count:]
            digest=float(combined[indices].sum(dtype=np.float64))
        else:digest=float(combined.sum(dtype=np.float64))
        if not math.isfinite(digest) or digest<=0:raise ValueError('probe output was not consumed')
        return dict(score_ns=score_end-start,selection_ns=time.monotonic_ns()-score_end,
                    elapsed_ns=time.monotonic_ns()-start,elements=self.elements,scan_bytes=self.elements*8,
                    digest=digest,placement='post-optimizer synchronous; no overlap claim',
                    working_set_bytes=sum(self.inputs[size][0].size*8 for size in self.inputs),
                    selection_scope='rank-local synthetic block selection; no TP score coordination')


class CopyInjection:
    """Consumed fixed-input loop-buffer traffic; reports its limited working set."""
    def __init__(self, library, rank, byte_count, granularity):
        import ctypes as C
        from npu_nvme.storage.bindings import load_backend
        self.C=C;self.acl=load_backend(library).acl_lib;self.bytes=byte_count;self.granularity=granularity
        self.size=min(256<<20,((byte_count+4095)//4096)*4096)
        self.stream=C.c_void_p();self.source=C.c_void_p();self.staging=C.c_void_p();self.reference=C.c_void_p();self.host=C.c_void_p()
        self.check(self.acl.aclrtSetDevice(rank))
        self.check(self.acl.aclrtCreateStream(C.byref(self.stream)))
        for value in (self.source,self.staging,self.reference):self.check(self.acl.aclrtMalloc(C.byref(value),self.size,0))
        self.check(self.acl.aclrtMallocHost(C.byref(self.host),self.bytes))
        C.memset(self.host,0x5a,self.bytes)
        self.copy(self.source,self.host,self.size,1);self.sync()

    def check(self, rc):
        if rc:raise RuntimeError('injection ACL operation failed: '+str(rc))

    def copy(self, target, source, size, kind):
        self.check(self.acl.aclrtMemcpyAsync(target,size,source,size,kind,self.stream))

    def sync(self):self.check(self.acl.aclrtSynchronizeStream(self.stream))

    def run(self, *, d2h=True):
        C=self.C;pack_ns=d2h_ns=0;records=[]
        begin=time.monotonic_ns()
        # Each staging chunk is consumed before reuse. Fine-grained sources
        # traverse the loop buffer backwards to create discontiguous reads.
        for base in range(0,self.bytes,self.size):
            amount=min(self.size,self.bytes-base);started=time.monotonic_ns()
            for offset in range(0,amount,self.granularity):
                count=min(self.granularity,amount-offset)
                source_offset=self.size-offset-count
                self.copy(C.c_void_p(self.staging.value+offset),C.c_void_p(self.source.value+source_offset),count,3)
            self.sync();pack_ns+=time.monotonic_ns()-started
            if d2h:
                started=time.monotonic_ns()
                self.copy(C.c_void_p(self.host.value+base),self.staging,amount,2);self.sync()
                d2h_ns+=time.monotonic_ns()-started
                records.append(dict(name='fixed_probe',element_offset=0,element_count=amount//4,
                                    payload_offset=base,payload_bytes=amount,itemsize=4,small=False))
        # A device-produced byte is consumed even in the pack-only rung.
        if not d2h:self.copy(self.host,self.staging,min(4096,self.size),2);self.sync()
        sample=memoryview((C.c_ubyte*min(4096,self.bytes)).from_address(self.host.value)).cast('B')
        if any(value!=0x5a for value in sample):raise ValueError('injected pack output differs')
        return dict(pack_ns=pack_ns,d2h_ns=d2h_ns,elapsed_ns=time.monotonic_ns()-begin,
                    bytes=self.bytes,granularity=self.granularity,records=records,
                    working_set_bytes=self.size*3,consumed_bytes=len(sample))

    def close(self):
        # Never release an allocation without proof that submitted DMA stopped.
        self.sync()
        self.check(self.acl.aclrtFreeHost(self.host))
        for value in (self.source,self.staging,self.reference):self.check(self.acl.aclrtFree(value))
        self.check(self.acl.aclrtDestroyStream(self.stream))


class ProbeController:
    """Training-boundary E2 probes and cumulative fixed-input E5 rungs."""
    def __init__(self, ms, network, options, *, rank, output):
        import ctypes
        import socket
        from pathlib import Path
        from npu_nvme.d2 import wire
        self.ms=ms;self.options=options;self.rank=rank;self.output=Path(output)
        self.probe=options['probe'];self.kind=self.probe['kind'];self.stage=self.probe.get('stage',0)
        self.deadline=time.monotonic()+options['timeout_seconds'];self.pending=None;self.operation=0;self.failed=False
        self.timings=[];self.score=None;self.copier=None;self.socket=None
        if self.kind.startswith('score') or (self.kind=='ablation' and self.stage>=2):
            self.score=ScoreInjection(ms,self.probe['target_elements'],self.probe.get('factor',1),
                                     selection=self.kind=='score_select' or self.stage>=3)
        if self.kind=='copy' or (self.kind=='ablation' and self.stage>=4):
            self.copier=CopyInjection(options['library'],rank,self.probe['output_bytes'],self.probe.get('granularity',4<<20))
            self.acl=self.copier.acl;self.stream=self.copier.stream
            self.reference_pointers={'fixed_probe':self.copier.reference.value}
        if self.kind=='ablation' and self.stage>=5:
            self.socket=socket.socket(socket.AF_UNIX);self.socket.connect(options['socket'])
            wire.send(self.socket,dict(kind='hello',rank=rank),b'',deadline=self.deadline,max_payload=0)

    def warmup(self):
        if self.copier:self.copier.run(d2h=self.kind=='copy' or self.stage>=5)

    def _finalize(self):
        if self.pending is None:return 0
        import ctypes
        from npu_nvme.experiments.runtime import _write
        start=time.monotonic_ns();pending=self.pending;pending['thread'].join()
        if pending.get('error'):raise pending['error']
        reference_start=time.monotonic_ns()
        if self.stage>=6:
            for row in pending['records']:
                self.copier.copy(self.copier.reference,ctypes.c_void_p(self.copier.host.value+row['payload_offset']),row['payload_bytes'],1)
            self.copier.sync()
        self.operation+=1
        _write(self.output/f'rank_{self.rank}'/'incremental-completions'/f"step-{pending['step']:04d}.json",
               dict(step=pending['step'],trigger_ns=pending['trigger_ns'],committed_ns=pending['committed_ns'],
                    reference_ready_ns=time.monotonic_ns(),reference_ns=time.monotonic_ns()-reference_start,
                    wait_and_reference_ns=time.monotonic_ns()-start))
        self.pending=None
        return time.monotonic_ns()-start

    def save(self, step):
        import ctypes
        import hashlib
        import threading
        from npu_nvme.experiments.runtime import IncrementalController
        start=time.monotonic_ns();self.ms.runtime.synchronize();stable=time.monotonic_ns()-start
        wait=self._finalize();row=dict(step=step,source_stable_ns=stable,wait_previous_ns=wait,payload_bytes=0,pending_requests=0,
            probe=self.probe,placement='post-optimizer synchronous; fixed loop-buffer inputs',scope='synthetic traffic; not a saved model',
            limitations=['rank-local synthetic selection omits TP coordination',
                         'fixed copy grain differs from actual model fragments',
                         'fixed payload metadata differs from model block metadata'])
        if self.score:row.update(self.score.run())
        records=[]
        if self.copier:
            value=self.copier.run(d2h=self.kind=='copy' or self.stage>=5);records=value.pop('records');row.update(value)
        if self.socket:
            view=memoryview((ctypes.c_ubyte*self.copier.bytes).from_address(self.copier.host.value)).cast('B')
            before=time.monotonic_ns();digest=hashlib.sha256(view).hexdigest();row['checksum_ns']=time.monotonic_ns()-before
            descriptor=dict(schema_version=1,group='synthetic-ablation',rank=self.rank,logical_step=step,records=records)
            self.pending=dict(host=self.copier.host,view=view,records=records,error=None,receipt=None,step=step,trigger_ns=start)
            thread=threading.Thread(target=IncrementalController._submit,args=(self,step,self.copier.host,view,descriptor,digest),daemon=True)
            self.pending['thread']=thread;thread.start();row.update(payload_bytes=len(view),pending_requests=1)
        row['critical_ns']=time.monotonic_ns()-start;self.timings.append(row);return row

    def close(self):
        from npu_nvme.d2 import wire
        drain=self._finalize()
        if self.socket:
            wire.send(self.socket,dict(kind='close',rank=self.rank,operation=self.operation),b'',deadline=self.deadline,max_payload=0)
            reply,payload=wire.receive(self.socket,deadline=self.deadline,max_payload=0)
            if payload or reply!=dict(kind='closed',rank=self.rank):raise ValueError('probe owner close differs')
            self.socket.close()
        if self.copier:self.copier.close()
        return drain
