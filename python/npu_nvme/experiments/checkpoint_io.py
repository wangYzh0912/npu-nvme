"""Append-only checkpoint I/O event records for the M0/M1 campaign."""
import json
import hashlib
import os
import time
from pathlib import Path


FIELDS=('run_id','rank','physical_npu','step','generation','raw_request_id',
        'native_request_id','tensor','chunk','offset','sha256','buffer_id','slot_generation',
        'event','clock_domain','timestamp_ns','bytes','status')


def emit(path, *, event, run_id, rank, physical_npu, step=None, generation=None,
         raw_request_id=None, native_request_id=None, tensor=None, chunk=None, offset=None, sha256=None,
         buffer_id=None, slot_generation=None, byte_count=None, status='ok',
         clock_domain='CLOCK_MONOTONIC', timestamp_ns=None):
    if not path:return
    row=dict(run_id=run_id,rank=rank,physical_npu=physical_npu,step=step,
             generation=generation,raw_request_id=raw_request_id,
             native_request_id=native_request_id,tensor=tensor,chunk=chunk,offset=offset,sha256=sha256,
             buffer_id=buffer_id,slot_generation=slot_generation,event=event,
             clock_domain=clock_domain,timestamp_ns=time.monotonic_ns() if timestamp_ns is None else timestamp_ns,
             bytes=byte_count,status=status)
    if tuple(row)!=FIELDS:raise AssertionError('event schema differs')
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
    fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    try:
        written=os.write(fd,payload)
        if written!=len(payload):raise OSError('short event append')
    finally:os.close(fd)


def consume_to_host_sink(schema, chunk_size, read_chunk, consume):
    """Synchronously consume validated capture chunks before releasing sources."""
    if type(chunk_size) is not int or chunk_size<=0:raise ValueError('chunk size')
    chunks=0;logical_bytes=0
    for tensor in sorted(schema,key=lambda row:row['name']):
        if type(tensor.get('bytes')) is not int or tensor['bytes']<=0:
            raise ValueError('tensor byte geometry')
        for offset in range(0,tensor['bytes'],chunk_size):
            length=min(chunk_size,tensor['bytes']-offset)
            with read_chunk(tensor['name'],offset,length) as data:
                if len(data)!=length:raise ValueError('capture short read')
                digest=hashlib.sha256(data).hexdigest()
                consume(tensor['name'],offset,length,digest,data)
            chunks+=1;logical_bytes+=length
    return dict(chunks=chunks,logical_bytes=logical_bytes)
