"""Explicit read-only V2 migration into a distinct D2 region."""
import hashlib

def migrate(*,source,record,destination,request_id,identity,chunk_bytes,source_layout,validate_source=None):
    if validate_source is None:
        from npu_nvme.runtime.d1_schema import validate_record
        validate_source=validate_record
    validate_source(record,source_layout) # Before begin or any source/destination I/O.
    if record.get('strict_contract')!='D1' or record.get('type')!='TRAINING_STATE_FULL':raise ValueError('unsupported V2 record')
    if type(chunk_bytes) is not int or chunk_bytes<=0 or chunk_bytes%4096:raise ValueError('chunk geometry')
    replay=destination.begin(request_id,identity)
    if replay is not None:return replay
    rows=[];whole_state=hashlib.sha256()
    try:
        for name,info in sorted(record['params'].items()):
            tensor=hashlib.sha256();cursor=0;logical=0;pending=bytearray()
            def emit():
                nonlocal logical
                data=bytes(pending);ref=destination.payload(data)
                rows.append(dict(rank=0,name=name,shape=info['shape'],dtype=info['dtype'],partition='per_rank_control' if name.startswith('control/') else 'replicated',logical_offset=logical,logical_bytes=len(data),payload=ref))
                logical+=len(data);pending.clear()
            for chunk in info['chunks']:
                if chunk['offset']!=cursor or chunk['size']<=0:raise ValueError('V2 chunk order')
                old_digest=hashlib.sha256();consumed=0
                while consumed<chunk['size']:
                    size=min(chunk_bytes-len(pending),chunk['size']-consumed)
                    data=source.read(info['offset']+cursor+consumed,size)
                    if len(data)!=size:raise ValueError('V2 short read')
                    old_digest.update(data);tensor.update(data);whole_state.update(data)
                    pending.extend(data);consumed+=size
                    if len(pending)==chunk_bytes:emit()
                if old_digest.hexdigest()!=chunk['sha256']:raise ValueError('V2 payload corruption')
                cursor+=chunk['size']
            if pending:emit()
            if cursor!=info['size'] or tensor.hexdigest()!=info['sha256']:raise ValueError('V2 tensor mismatch')
        if whole_state.hexdigest()!=record['data_sha256']:raise ValueError('V2 state mismatch')
        return destination.commit(rows,step=record['state_step'],topology=dict(world_size=1,tp=1,dp=1,pp=1))
    except BaseException:
        destination.poisoned=True
        raise
