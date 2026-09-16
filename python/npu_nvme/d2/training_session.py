"""Multiple FULL operations over one owner/rank connection set.

Each operation has a new epoch/request identity. The underlying single-operation
protocol retains its commit, checksum and global-ready guarantees.
"""
import socket
import time
from . import wire
from .session import OwnerSession, RankSession


def operation_message(*, epoch, sequence, rank, operation, step=0, generation=None):
    if operation not in ('save', 'restore', 'close'):
        raise ValueError('unknown training operation')
    if type(sequence) is not int or sequence < 0 or type(step) is not int:
        raise ValueError('invalid operation sequence/step')
    if operation != 'close' and step <= 0:
        raise ValueError('checkpoint step must be positive')
    if operation == 'restore':
        if type(generation) is not int or generation <= 0:
            raise ValueError('restore must select an explicit committed generation')
    elif generation is not None:
        raise ValueError('generation is only valid for restore')
    return dict(kind='operation', epoch=epoch, sequence=sequence, rank=rank,
                operation=operation, step=step, generation=generation)


class TrainingOwner:
    def __init__(self, region, connections, *, epoch, identity, chunk_bytes,
                 deadline, operation_timeout, authoritative_schema=None):
        self.region=region; self.connections=connections; self.epoch=epoch
        self.identity=identity; self.chunk=chunk_bytes; self.deadline=deadline
        self.timeout=operation_timeout; self.strategy=authoritative_schema
        if set(connections) != set(range(len(connections))) or len(connections) not in (2,4):
            raise ValueError('training owner requires all rank connections')

    def run(self, on_result=None):
        sequence=0
        try:
            while True:
                common=None
                for rank, connection in sorted(self.connections.items()):
                    row, data=wire.receive(connection,deadline=self.deadline,max_payload=0)
                    expected=operation_message(epoch=self.epoch,sequence=sequence,rank=rank,
                        operation=row.get('operation'),step=row.get('step'),generation=row.get('generation'))
                    if data or row!=expected:
                        raise ValueError('training operation identity differs')
                    value={k:v for k,v in row.items() if k!='rank'}
                    if common is not None and value!=common:
                        raise ValueError('ranks requested different checkpoint operations')
                    common=value
                if common['operation']=='close':
                    for connection in self.connections.values():
                        wire.send(connection,dict(kind='closed',sequence=sequence),b'',deadline=self.deadline,max_payload=0)
                    return sequence
                epoch=f'{self.epoch}:{sequence}'
                session=OwnerSession(self.region,self.connections,epoch=epoch,request_id=epoch,
                    topology=dict(world_size=len(self.connections)),chunk_bytes=self.chunk,
                    deadline=min(self.deadline,time.monotonic()+self.timeout),identity=self.identity,
                    authoritative_schema=self.strategy)
                receipt=(session.save(step=common['step']) if common['operation']=='save'
                         else session.restore(generation=common['generation']))
                if on_result is not None: on_result(dict(common,receipt=receipt))
                sequence+=1
        except BaseException:
            # A protocol error invalidates the connection set. This only wakes
            # peers; native resources are released by their checked close.
            for connection in self.connections.values():
                try: connection.shutdown(socket.SHUT_RDWR)
                except OSError: pass
            raise


class TrainingRank:
    def __init__(self, connection, *, rank, epoch, identity, chunk_bytes, deadline, operation_timeout):
        self.connection=connection; self.rank=rank; self.epoch=epoch; self.identity=identity
        self.chunk=chunk_bytes; self.deadline=deadline; self.timeout=operation_timeout
        self.sequence=0; self.failed=False; self.closed=False

    def _session(self, operation, schema, step, generation=None):
        if self.closed or self.failed: raise RuntimeError('training connection is unavailable')
        row=operation_message(epoch=self.epoch,sequence=self.sequence,rank=self.rank,
            operation=operation,step=step,generation=generation)
        deadline=min(self.deadline,time.monotonic()+self.timeout)
        wire.send(self.connection,row,b'',deadline=deadline,max_payload=0)
        return RankSession(self.connection,rank=self.rank,epoch=f'{self.epoch}:{self.sequence}',
            identity=self.identity,schema=schema,chunk_bytes=self.chunk,deadline=deadline)

    def save(self, schema, read_chunk, controls, *, step):
        try:
            result=self._session('save',schema,step).save(read_chunk,controls,step=step)
            self.sequence+=1
            return result
        except BaseException:
            self.failed=True
            raise

    def restore(self, schema, apply_chunk, verify_controls, *, step, generation):
        try:
            def verified(controls, stored_step):
                if stored_step!=step: raise ValueError('selected generation step differs')
                verify_controls(controls,stored_step)
            result=self._session('restore',schema,step,generation).restore(apply_chunk,verified)
            self.sequence+=1
            return result
        except BaseException:
            self.failed=True
            raise

    def close(self):
        if self.closed:return
        try:
            if not self.failed:
                row=operation_message(epoch=self.epoch,sequence=self.sequence,rank=self.rank,operation='close')
                wire.send(self.connection,row,b'',deadline=self.deadline,max_payload=0)
                reply,data=wire.receive(self.connection,deadline=self.deadline,max_payload=0)
                if data or reply!=dict(kind='closed',sequence=self.sequence):raise ValueError('owner close ACK differs')
        finally:
            self.connection.close();self.closed=True
