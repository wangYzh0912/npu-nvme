"""Framework callback bridge; no model import or implicit environment selection."""
import json
import os
import socket
import time
from pathlib import Path
from .d2_state import BlockingState
from npu_nvme.d2 import wire
from npu_nvme.d2.session import RankSession
from npu_nvme.d2.rank_copy_executor import RankCopyExecutor
from npu_nvme.storage.bindings import load_backend


def checkpoint_session(framework,network,*,operation,rank,npu_id,connection,library,
                       partitions,controls=None,verify_controls=None,collective_barrier=None):
    """Save while optimizer updates are stopped, or restore before first update.

    ``verify_controls`` must apply and read back framework controls and compare
    restored tensor hashes. ``collective_barrier`` is mandatory on restore;
    every rank must join it after the verified socket release.
    """
    if operation not in ('save','restore'):raise ValueError('checkpoint operation')
    if operation=='restore' and (verify_controls is None or collective_barrier is None):
        raise ValueError('restore requires verification and framework collective barrier')
    cfg=json.loads(Path(connection).read_text())
    if cfg['capture']!='blocking' or cfg['world_size']!=4 or not 0<=rank<4 or npu_id!=rank:
        raise ValueError('only explicit blocking TP4 on NPU0..3 is admitted')
    deadline=time.monotonic()+cfg['timeout_seconds'];executor=None;sock=socket.socket(socket.AF_UNIX)
    started=time.monotonic()
    status_path=Path(connection).parent/f'rank_{rank}'/'d2-session-status.json'
    status_path.parent.mkdir(parents=True,exist_ok=True)
    status=dict(status='running',operation=operation,pid=os.getpid(),rank=rank)
    def record(**fields):
        status.update(fields)
        if executor:status['last_copy_receipt']=getattr(executor,'last_receipt',None)
        temporary=status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(status_path)
    record(phase='opening')
    try:
        executor=RankCopyExecutor(load_backend(library),dict(chunk_bytes=cfg['chunk_bytes'],
            shm_id=cfg['rank_shm_base']+rank),npu_id=npu_id,depth=cfg['depth'],timeout_ms=120000)
        state=BlockingState(framework,network,rank=rank,executor=executor,
                            host_tensor_budget=cfg['host_tensor_budget_bytes'],partitions=partitions)
        if cfg.get('strategy_path'):
            from npu_nvme.d2.tp_schema import validate_rank,digest
            strategy=json.loads(Path(cfg['strategy_path']).read_text())
            if digest(strategy)!=cfg['identity'].get('strategy_sha256'):raise ValueError('strategy identity differs')
            validate_rank(state.schema,strategy,rank)
        sock.settimeout(max(.001,deadline-time.monotonic()));sock.connect(cfg['socket'])
        wire.send(sock,dict(kind='connect',rank=rank,epoch=cfg['epoch']),b'',deadline=deadline,max_payload=0)
        session=RankSession(sock,rank=rank,epoch=cfg['epoch'],identity=cfg['identity'],schema=state.schema,
                            chunk_bytes=cfg['chunk_bytes'],deadline=deadline)
        record(phase='transfer',state_bytes=sum(v['bytes'] for v in state.schema))
        if operation=='save':
            if controls is None:raise ValueError('full-state controls required')
            receipt=session.save(state.read_chunk,controls,step=cfg['step'])
        else:
            def verified(value,step):
                state.verify_finished();verify_controls(value,step)
            receipt=session.restore(state.apply_chunk,verified)
            collective_barrier()
        return dict(receipt=receipt,seconds=time.monotonic()-started,capture='blocking',
            state_bytes=sum(v['bytes'] for v in state.schema),placement=state.placement,
            native_dma_pool_bytes=executor.transport.capabilities.dma_pool_bytes,
            host_bridge_bytes=cfg['chunk_bytes'])
    except BaseException as error:
        record(status='fail',error=repr(error))
        raise
    finally:
        sock.close()
        if executor and not executor.close():
            # Do not permit the callback to unwind and release borrowed tensors.
            record(status='retained',phase='close',error=status.get('error','native close lacks stop proof'))
            import threading
            threading.Event().wait()
        record(status='pass' if status['status']=='running' else status['status'],closed=True)
