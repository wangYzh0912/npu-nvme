"""Draft bounded rank IPC framing; explicit epoch/offset leases and deadlines."""
from __future__ import annotations
import hashlib
import json
import select
import socket
import struct
import time

HEADER=struct.Struct('!8sIIQ32s')
MAGIC=b'NPURANK3'
MAX_CONTROL=1024*1024


def _ready(sock,write,deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('rank IPC observation deadline')
    try:
        readable,writable,_=select.select([] if write else [sock],[sock] if write else [],[],remaining)
    except InterruptedError:
        return _ready(sock,write,deadline)
    if not (writable if write else readable):raise TimeoutError('rank IPC observation deadline')


def _read(sock,size,deadline):
    chunks=bytearray(size);view=memoryview(chunks);cursor=0
    while cursor<size:
        _ready(sock,False,deadline)
        try:count=sock.recv_into(view[cursor:], flags=socket.MSG_DONTWAIT)
        except (BlockingIOError,InterruptedError):continue
        if not count:raise EOFError('rank disconnected')
        cursor+=count
    return bytes(chunks)


def _write(sock,data,deadline):
    view=memoryview(data);cursor=0
    while cursor<len(view):
        _ready(sock,True,deadline)
        try:count=sock.send(view[cursor:], socket.MSG_DONTWAIT)
        except (BlockingIOError,InterruptedError):continue
        if not count:raise EOFError('rank disconnected')
        cursor+=count


def send(sock,control,payload,*,deadline,max_payload):
    raw=json.dumps(control,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if len(raw)>MAX_CONTROL or len(payload)>max_payload:raise ValueError('IPC frame exceeds credit')
    digest=hashlib.sha256(raw+payload).digest()
    _write(sock,HEADER.pack(MAGIC,3,len(raw),len(payload),digest),deadline)
    _write(sock,raw,deadline);_write(sock,payload,deadline)


def receive(sock,*,deadline,max_payload):
    magic,version,size,length,digest=HEADER.unpack(_read(sock,HEADER.size,deadline))
    # Reject declared oversize before allocating or reading payload.
    if magic!=MAGIC or version!=3 or size>MAX_CONTROL or length>max_payload:raise ValueError('invalid IPC header')
    raw=_read(sock,size,deadline);payload=_read(sock,length,deadline)
    if hashlib.sha256(raw+payload).digest()!=digest:raise ValueError('IPC digest mismatch')
    def pairs(items):
        d={}
        for key,value in items:
            if key in d:raise ValueError('duplicate IPC key')
            d[key]=value
        return d
    control=json.loads(raw,object_pairs_hook=pairs)
    if not isinstance(control,dict):raise ValueError('IPC control must be object')
    return control,payload


class LeasePool:
    """Slot epochs prevent a delayed rank ACK from freeing reused Host memory."""
    def __init__(self,slots,slot_bytes,epoch):
        if slots<=0 or slot_bytes<=0 or not epoch:raise ValueError('pool geometry')
        self.slots=slots;self.bytes=slot_bytes;self.epoch=epoch;self.generation=[0]*slots;self.owners={}
    def acquire(self,rank,request,length):
        if type(rank) is not int or rank<0 or length<=0 or length>self.bytes:raise ValueError('lease bounds')
        slot=next((i for i in range(self.slots) if i not in self.owners),None)
        if slot is None:raise BlockingIOError('no Host credit')
        self.generation[slot]+=1
        lease=dict(epoch=self.epoch,slot=slot,generation=self.generation[slot],rank=rank,request_id=request,length=length,offset=slot*self.bytes)
        self.owners[slot]=lease;return dict(lease)
    def release(self,lease,*,transport_safe):
        if transport_safe is not True:raise RuntimeError('lease lacks DMA stop proof')
        slot=lease.get('slot')
        if slot not in self.owners or self.owners[slot]!=lease:raise ValueError('stale/wrong lease')
        del self.owners[slot]
