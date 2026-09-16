"""Persistent rank-local FULL adapter for a training process."""
import json
import os
from pathlib import Path
import socket
import threading
import time

from .d2_state import BlockingState
from npu_nvme.d2 import wire
from npu_nvme.d2.rank_copy_executor import RankCopyExecutor
from npu_nvme.d2.training_session import TrainingRank
from npu_nvme.d2.tp_schema import validate_rank, digest
from npu_nvme.storage.bindings import load_backend


class TrainingCheckpoint:
    def __init__(self, config, *, rank, npu_id, output):
        self.config=config;self.rank=rank;self.npu=npu_id;self.out=Path(output)
        self.out.mkdir(parents=True,exist_ok=True)
        self.executor=None;self.client=None;self.state=None;self.connection=None
        self.status=dict(status='running',pid=os.getpid(),rank=rank)

    def record(self, **values):
        self.status.update(values)
        if self.executor:self.status['last_copy_receipt']=getattr(self.executor,'last_receipt',None)
        path=self.out/'d2-session-status.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.status,indent=2)+'\n');temporary.replace(path)

    def _open(self):
        if self.client is not None:return
        c=self.config
        self.record(phase='opening')
        self.executor=RankCopyExecutor(load_backend(c['library']),
            dict(chunk_bytes=c['chunk_bytes'],shm_id=c['rank_shm_base']+self.rank,profiling_dir=str(self.out)),
            npu_id=self.npu,depth=c['depth'],timeout_ms=c['copy_timeout_ms'])
        self.connection=socket.socket(socket.AF_UNIX)
        deadline=time.monotonic()+c['timeout_seconds']
        self.connection.settimeout(c['operation_timeout_seconds'])
        self.connection.connect(c['socket'])
        wire.send(self.connection,dict(kind='connect',rank=self.rank,epoch=c['epoch']),b'',deadline=deadline,max_payload=0)
        self.client=TrainingRank(self.connection,rank=self.rank,epoch=c['epoch'],identity=c['identity'],
            chunk_bytes=c['chunk_bytes'],deadline=deadline,operation_timeout=c['operation_timeout_seconds'])

    def transfer(self, framework, network, *, operation, step, partitions, controls=None,
                 generation=None, verify_controls=None, barrier=None):
        try:
            self._open()
            self.state=BlockingState(framework,network,rank=self.rank,executor=self.executor,
                host_tensor_budget=self.config['host_tensor_budget_bytes'],partitions=partitions)
            strategy=self.config['strategy']
            if digest(strategy)!=self.config['identity']['strategy_sha256']:
                raise ValueError('training strategy identity differs')
            validate_rank(self.state.schema,strategy,self.rank)
            sequence=self.client.sequence
            self.record(phase='transfer',operation=operation,step=step,sequence=sequence)
            started=time.monotonic()
            if operation=='save':
                if controls is None:raise ValueError('FULL save requires controls')
                receipt=self.client.save(self.state.schema,self.state.read_chunk,controls,step=step)
            elif operation=='restore':
                if verify_controls is None or barrier is None:raise ValueError('restore requires verification and barrier')
                def verify(value,stored_step):
                    self.state.verify_finished();verify_controls(value,stored_step)
                receipt=self.client.restore(self.state.schema,self.state.apply_chunk,verify,
                    step=step,generation=generation)
                barrier()
            else:raise ValueError('unsupported FULL operation')
            result=dict(receipt=receipt,step=step,operation=operation,sequence=sequence,
                seconds=time.monotonic()-started,placement=self.state.placement,
                state_bytes=sum(row['bytes'] for row in self.state.schema),
                native_dma_pool_bytes=self.executor.transport.capabilities.dma_pool_bytes)
            (self.out/f'd2-operation-{sequence:04d}.json').write_text(json.dumps(result,indent=2)+'\n')
            self.record(phase='idle',status='running')
            self.state=None
            return result
        except BaseException as error:
            if self.client:self.client.failed=True
            self.record(status='fail',error=repr(error))
            raise

    def close(self):
        try:
            if self.client:self.client.close()
            elif self.connection:self.connection.close()
        finally:
            if self.executor and not self.executor.close():
                self.record(status='retained',phase='close',closed=False)
                # Keep the entire last state/allocator alive until recovery can
                # establish a device stop proof. The supervisor sees retained.
                threading.Event().wait()
            self.state=None
            self.record(status='pass' if self.status['status']=='running' else self.status['status'],closed=True)
