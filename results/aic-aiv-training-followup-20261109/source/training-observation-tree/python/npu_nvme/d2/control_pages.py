"""Page rank control JSON independently of per-page row limits."""
import base64
import hashlib
import json

def encode(values,*,budget,slice_bytes=16384):
    total=0;seen=set()
    for value in values:
        rank=value['rank']
        if type(rank) is not int or rank<0 or rank in seen:raise ValueError('duplicate/invalid control rank')
        seen.add(rank)
        raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        total+=len(raw)
        if total>budget:raise ValueError('control aggregate budget')
        digest=hashlib.sha256(raw).hexdigest()
        for offset in range(0,len(raw),slice_bytes):
            yield dict(rank=rank,offset=offset,bytes=len(raw),sha256=digest,data=base64.b64encode(raw[offset:offset+slice_bytes]).decode())

def decode(rows,*,budget):
    buffers={};identities={};total=0
    for row in rows:
        if type(row) is not dict or set(row)!={'rank','offset','bytes','sha256','data'}:raise ValueError('control page fields')
        rank=row['rank'];size=row['bytes'];offset=row['offset']
        if type(rank) is not int or rank<0 or type(size) is not int or size<=0 or type(offset) is not int or offset<0:raise ValueError('control page geometry')
        if rank not in buffers:
            total+=size
            if total>budget:raise ValueError('control decode budget')
            buffers[rank]=bytearray();identities[rank]=(size,row['sha256'])
        if identities[rank]!=(size,row['sha256']) or offset!=len(buffers[rank]):raise ValueError('control slice identity/order')
        data=base64.b64decode(row['data'],validate=True)
        if not data or offset+len(data)>size:raise ValueError('control slice bounds')
        buffers[rank].extend(data)
    result=[]
    def pairs(items):
        value={}
        for k,v in items:
            if k in value:raise ValueError('duplicate control key')
            value[k]=v
        return value
    for rank,raw in sorted(buffers.items()):
        size,digest=identities[rank]
        if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('control integrity')
        value=json.loads(raw,object_pairs_hook=pairs)
        if type(value) is not dict or value.get('rank')!=rank:raise ValueError('control rank mismatch')
        result.append(value)
    return result
