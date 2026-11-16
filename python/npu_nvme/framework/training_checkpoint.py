"""Persistent rank-local FULL adapter for a training process."""
import json
import os
from pathlib import Path
import socket
import threading
import time
from contextlib import contextmanager

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
        self.sink_only=bool(config.get('sink_only'))
        self.sink_sequence=0
        event_root=config.get('io_event_root')
        self.event_path=(str(Path(event_root)/f'rank-{rank}.jsonl') if event_root else None)

    def event(self,name,*,step=None,**fields):
        if not self.event_path:return
        from npu_nvme.experiments.checkpoint_io import emit
        emit(self.event_path,event=name,run_id=self.config.get('run_id',''),rank=self.rank,
             physical_npu=self.config['identity']['physical_npu_ids'][self.rank],step=step,**fields)

    def record(self, **values):
        self.status.update(values)
        if self.executor:self.status['last_copy_receipt']=getattr(self.executor,'last_receipt',None)
        path=self.out/'d2-session-status.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.status,indent=2)+'\n');temporary.replace(path)

    def _open(self):
        if self.executor is not None:return
        c=self.config
        self.record(phase='opening')
        self.executor=RankCopyExecutor(load_backend(c['library']),
            dict(chunk_bytes=c['chunk_bytes'],shm_id=c['rank_shm_base']+self.rank,
                 profiling_dir=str(self.out),measure=bool(self.event_path)),
            npu_id=self.npu,depth=c['depth'],timeout_ms=c['copy_timeout_ms'])
        if self.sink_only:return
        self.connection=socket.socket(socket.AF_UNIX)
        deadline=time.monotonic()+c['timeout_seconds']
        self.connection.settimeout(c['operation_timeout_seconds'])
        self.connection.connect(c['socket'])
        if self.event_path:
            self.status['socket_buffers']=dict(send=self.connection.getsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF),
                receive=self.connection.getsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF))
        wire.send(self.connection,dict(kind='connect',rank=self.rank,epoch=c['epoch']),b'',deadline=deadline,max_payload=0)
        self.client=TrainingRank(self.connection,rank=self.rank,epoch=c['epoch'],identity=c['identity'],
            chunk_bytes=c['chunk_bytes'],deadline=deadline,operation_timeout=c['operation_timeout_seconds'])

    def transfer(self, framework, network, *, operation, step, partitions, controls=None,
                 generation=None, verify_controls=None, barrier=None, prepare=None):
        try:
            self.event('capture_setup_begin',step=step)
            self._open()
            self.state=BlockingState(framework,network,rank=self.rank,executor=self.executor,
                host_tensor_budget=self.config['host_tensor_budget_bytes'],partitions=partitions)
            strategy=self.config['strategy']
            if digest(strategy)!=self.config['identity']['strategy_sha256']:
                raise ValueError('training strategy identity differs')
            validate_rank(self.state.schema,strategy,self.rank)
            sequence=self.client.sequence if self.client else self.sink_sequence
            request=f"{self.client.epoch if self.client else self.config['epoch']}:{sequence}"
            capture_chunk=0
            @contextmanager
            def read_chunk(name,offset,length):
                nonlocal capture_chunk
                started_ns=time.monotonic_ns()
                with self.state.read_chunk(name,offset,length) as data:
                    device=self.state.placement[name]=='device'
                    native=self.executor.last_receipt.get('request_id') if device else None
                    fields=dict(step=step,raw_request_id=request,tensor=name,chunk=capture_chunk,
                                byte_count=length,native_request_id=native,
                                slot_generation=self.executor.buffer_generation if device else None,
                                buffer_id=f'rank:{self.rank}:host:{self.executor.base}' if device else None)
                    self.event('source_read_begin',timestamp_ns=started_ns,**fields)
                    self.event('data_ready',**fields)
                    yield data
                self.event('source_released',**fields)
                capture_chunk+=1
            self.record(phase='transfer',operation=operation,step=step,sequence=sequence)
            self.event('capture_begin',step=step,raw_request_id=request)
            self.event('capture_setup_end',step=step)
            started=time.monotonic()
            if operation=='save':
                if controls is None:raise ValueError('FULL save requires controls')
                if prepare is not None:
                    epoch=self.client.epoch if self.client else self.config['epoch']
                    self.event('prepared_journal_begin',step=step,raw_request_id=request)
                    prepare(dict(request_id=f'{epoch}:{sequence}',step=step,operation='save',
                        sequence=sequence,seconds=None,placement=self.state.placement,
                        state_bytes=sum(row['bytes'] for row in self.state.schema),
                        native_dma_pool_bytes=self.executor.transport.capabilities.dma_pool_bytes))
                    self.event('prepared_journal_end',step=step,raw_request_id=request)
                    self.event('prepared_barrier_begin',step=step,raw_request_id=request)
                    barrier()
                    self.event('prepared_barrier_end',step=step,raw_request_id=request)
                if self.sink_only:
                    from npu_nvme.experiments.checkpoint_io import consume_to_host_sink
                    chunk_index=0
                    def consume(name,offset,length,digest,data):
                        nonlocal chunk_index
                        self.event('sink_consumed_source_safe',step=step,tensor=name,chunk=chunk_index,
                                   byte_count=length,offset=offset,sha256=digest,
                                   raw_request_id=f"{self.config['epoch']}:{sequence}")
                        chunk_index+=1
                    consumed=consume_to_host_sink(self.state.schema,self.config['chunk_bytes'],
                        read_chunk,consume)
                    receipt=dict(sink='bounded_host_sha256',**consumed,
                                 source_safe=True,durable=False,published=False)
                    self.sink_sequence+=1
                else:
                    receipt=self.client.save(self.state.schema,read_chunk,controls,step=step,
                                             observer=self.event)
            elif operation=='restore':
                if verify_controls is None or barrier is None:raise ValueError('restore requires verification and barrier')
                restored_chunks=0
                def apply_chunk(name,offset,data,digest):
                    nonlocal restored_chunks
                    begin_ns=time.monotonic_ns()
                    self.state.apply_chunk(name,offset,data,digest)
                    device=self.state.placement[name]=='device'
                    fields=dict(step=step,generation=generation,raw_request_id=request,tensor=name,
                        chunk=restored_chunks,offset=offset,byte_count=len(data),sha256=digest,
                        native_request_id=self.executor.last_receipt['request_id'] if device else None,
                        buffer_id=f'rank:{self.rank}:host:{self.executor.base}' if device else None,
                        slot_generation=self.executor.buffer_generation if device else None)
                    self.event('restore_apply_begin',timestamp_ns=begin_ns,**fields)
                    self.event('restore_copy_safe',**fields)
                    restored_chunks+=1
                def verify(value,stored_step):
                    self.state.verify_finished();verify_controls(value,stored_step)
                receipt=self.client.restore(self.state.schema,apply_chunk,verify,
                    step=step,generation=generation,observer=self.event)
                barrier()
                self.event('restore_collective_ready',step=step,generation=generation,raw_request_id=request)
            else:raise ValueError('unsupported FULL operation')
            result=dict(receipt=receipt,step=step,operation=operation,sequence=sequence,
                seconds=time.monotonic()-started,placement=self.state.placement,
                state_bytes=sum(row['bytes'] for row in self.state.schema),
                native_dma_pool_bytes=self.executor.transport.capabilities.dma_pool_bytes)
            (self.out/f'd2-operation-{sequence:04d}.json').write_text(json.dumps(result,indent=2)+'\n')
            self.event('capture_end',step=step,raw_request_id=request,
                       byte_count=result['state_bytes'],generation=receipt.get('generation'))
            self.record(phase='idle',status='running')
            self.state=None
            if self.event_path:
                from npu_nvme.experiments.checkpoint_io_closure import native_snapshot
                self.record(native=native_snapshot(self.executor.transport),copy_api=self.executor.measurement)
            return result
        except BaseException as error:
            if self.client:self.client.failed=True
            self.record(status='fail',error=repr(error))
            self.event('capture_failed',step=step,status=repr(error))
            raise

    def close(self):
        try:
            self.record(close_begin_ns=time.monotonic_ns())
            if self.client:self.client.close()
            elif self.connection:self.connection.close()
            if self.event_path and self.executor:
                from npu_nvme.experiments.checkpoint_io_closure import native_snapshot,verify_drained
                snapshot=native_snapshot(self.executor.transport);verify_drained(snapshot)
                self.record(native=snapshot,copy_api=self.executor.measurement)
        except BaseException as error:
            self.record(status='fail',error=repr(error))
            raise
        finally:
            if self.executor and not self.executor.close():
                self.record(status='retained',phase='close',closed=False)
                # Keep the entire last state/allocator alive until recovery can
                # establish a device stop proof. The supervisor sees retained.
                threading.Event().wait()
            self.state=None
            self.record(status='pass' if self.status['status']=='running' else self.status['status'],
                        closed=True,close_end_ns=time.monotonic_ns())
