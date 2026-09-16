#!/usr/bin/env python3
"""Read and hash every safetensors payload with bounded memory, without a runtime."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

ITEM_BYTES = {'BOOL':1,'U8':1,'I8':1,'U16':2,'I16':2,'F16':2,'BF16':2,
              'U32':4,'I32':4,'F32':4,'U64':8,'I64':8,'F64':8}
SCALARS = {'I32':'<i','I64':'<q','F32':'<f','F64':'<d','U64':'<Q'}


def unique_object(pairs):
    result = {}
    for key,value in pairs:
        if key in result: raise ValueError('duplicate JSON key: '+key)
        result[key] = value
    return result


def inspect(path):
    path = Path(path).resolve()
    before = path.stat()
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8: raise ValueError('truncated safetensors prefix')
        length = struct.unpack('<Q',prefix)[0]
        if not 2 <= length <= 32*1024**2 or length > before.st_size-8:
            raise ValueError('invalid safetensors header length')
        raw = stream.read(length)
        header = json.loads(raw,object_pairs_hook=unique_object)
        if not isinstance(header,dict): raise ValueError('header must be an object')
        entries=[]
        for name,value in header.items():
            if name=='__metadata__': continue
            if not isinstance(value,dict) or set(value)!={'dtype','shape','data_offsets'}:
                raise ValueError('invalid tensor descriptor')
            dtype,shape,offsets = value['dtype'],value['shape'],value['data_offsets']
            if dtype not in ITEM_BYTES: raise ValueError('unsupported dtype')
            if not isinstance(shape,list) or len(shape)>32 or any(type(d) is not int or d<0 for d in shape):
                raise ValueError('invalid shape')
            if not isinstance(offsets,list) or len(offsets)!=2 or any(type(x) is not int or x<0 for x in offsets):
                raise ValueError('invalid offsets')
            start,end=offsets
            size=math.prod(shape)*ITEM_BYTES[dtype]
            if end<start or end-start!=size or end>before.st_size-8-length:
                raise ValueError('tensor extent/shape/dtype mismatch')
            entries.append(dict(name=name,shape=shape,dtype=dtype,offset=start,end=end,bytes=size))
        if not entries: raise ValueError('empty checkpoint')
        cursor=0
        for entry in sorted(entries,key=lambda x:(x['offset'],x['end'])):
            if entry['offset']!=cursor: raise ValueError('overlap or hole in tensor data')
            cursor=entry['end']
        if cursor!=before.st_size-8-length: raise ValueError('trailing/truncated tensor payload')
        digest=hashlib.sha256(prefix+raw)
        for entry in sorted(entries,key=lambda x:(x['offset'],x['end'])):
            remaining=entry['bytes'];tensor_hash=hashlib.sha256();scalar=b''
            while remaining:
                block=stream.read(min(8*1024**2,remaining))
                if not block: raise ValueError('truncated tensor payload')
                digest.update(block);tensor_hash.update(block);remaining-=len(block)
                if entry['bytes']<=8: scalar+=block
            entry['sha256']=tensor_hash.hexdigest()
            if math.prod(entry['shape'])==1 and entry['dtype'] in SCALARS:
                entry['scalar_value']=struct.unpack(SCALARS[entry['dtype']],scalar)[0]
        after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('checkpoint changed during audit')
    return dict(path=str(path),bytes=before.st_size,sha256=digest.hexdigest(),
                header_sha256=hashlib.sha256(raw).hexdigest(),logical_bytes=cursor,
                tensors=entries,evidence='actual payload bytes read',
                limitations=['does not prove saved state consistency or fresh-process restoration'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+',type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    files=[]
    for i,path in enumerate(a.paths):
        result=inspect(path)
        output=a.out/f'file-{i:03d}.json';output.write_text(json.dumps(result,indent=2)+'\n')
        files.append(dict(path=str(path),manifest=output.name,bytes=result['bytes'],sha256=result['sha256']))
        print(f"Audited {path}: {result['bytes']} bytes",flush=True)
    (a.out/'manifest.json').write_text(json.dumps(dict(schema_version=1,files=files),indent=2)+'\n')

if __name__=='__main__': main()
