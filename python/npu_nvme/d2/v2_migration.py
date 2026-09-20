"""Explicit read-only V2 migration into a distinct D2 region."""
import hashlib

def migrate(*,source,record,destination,request_id,identity,chunk_bytes,source_layout,validate_source=None):
    if validate_source is None:
        from npu_nvme.runtime.d1_schema import validate_record
        validate_source=validate_record
    validate_source(record,source_layout) # Before begin or any source/destination I/O.
    if record.get('strict_contract')!='D1' or record.get('type')!='TRAINING_STATE_FULL':raise ValueError('unsupported V2 record')
    if type(chunk_bytes) is not int or chunk_bytes<=0 or chunk_bytes%4096:raise ValueError('chunk geometry')
    # V2 source and D2 destination may share the authorized physical device.
    # Reject overlap before admitting any destination write.
    if getattr(source, 'storage_identity', ('raw',))[0] != 'file':
        extents = [(info['offset'], (info['size']+4095)//4096*4096) for info in record['params'].values()]
        if source_layout is not None:
            extents += [(0, source_layout.full_end), (source_layout.delta_base, source_layout.delta_bytes)]
        for start, length in extents:
            if start < destination.end and destination.base < start+length:
                raise ValueError('V2 source overlaps D2 destination')
    from .region_budget import region_budget
    budget = region_budget(state_bytes=sum(v['size'] for v in record['params'].values()),
        descriptor_count=sum((v['size']+chunk_bytes-1)//chunk_bytes for v in record['params'].values()),
        page_bytes=destination.codec.page_bytes,retention=destination.retention,active_pins=0)
    reserve=(budget['per_generation_upper_bytes']+4*destination.codec.page_bytes+4095)//4096*4096
    replay=destination.begin(request_id,identity,reserve_bytes=reserve)
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
