"""Draft strict D2 manifest schema validator using actual per-rank geometry."""
import math

WIDTHS={'float32':4,'float16':2,'bfloat16':2,'float64':8,'int64':8,'int32':4,'int16':2,'int8':1,'uint8':1,'bool':1}


def validate(rows,expected,*,world_size,chunk_bytes,max_manifest_bytes=64*1024**2):
    import json
    if world_size not in (1,2,4) or chunk_bytes<=0 or chunk_bytes%4096:raise ValueError('manifest topology/chunk')
    schema={}
    for item in expected:
        rank=item['rank'];name=item['name'];shape=item['shape'];dtype=item['dtype'];partition=item['partition']
        if type(rank) is not int or not 0<=rank<world_size or not isinstance(name,str) or not 0<len(name)<=1024:raise ValueError('rank/name')
        if dtype not in WIDTHS or not isinstance(shape,list) or any(type(x) is not int or x<0 for x in shape):raise ValueError('tensor geometry')
        if partition not in ('replicated','sharded','per_rank_control'):raise ValueError('partition')
        size=math.prod(shape)*WIDTHS[dtype]
        if size<=0 or size!=item['bytes'] or (rank,name) in schema:raise ValueError('tensor byte count or duplicate')
        schema[(rank,name)]=item
    cursors={};encoded=0;count=0
    for row in rows:
        encoded+=len(json.dumps(row,separators=(',',':'),allow_nan=False).encode())
        if encoded>max_manifest_bytes:raise ValueError('aggregate manifest budget')
        key=(row['rank'],row['name'])
        if key not in schema:raise ValueError('unexpected rank tensor')
        item=schema[key];offset=cursors.get(key,0)
        if row['shape']!=item['shape'] or row['dtype']!=item['dtype'] or row['partition']!=item['partition']:raise ValueError('schema changed')
        if type(row['logical_offset']) is not int or row['logical_offset']!=offset:raise ValueError('duplicate/missing offset')
        size=min(chunk_bytes,item['bytes']-offset)
        if type(row['logical_bytes']) is not int or row['logical_bytes']!=size or size<=0:raise ValueError('chunk size')
        ref=row['payload']
        if ref['logical_bytes']!=size or ref['length']!=(size+4095)//4096*4096:raise ValueError('payload geometry')
        for name in ('sha256','logical_sha256'):
            digest=ref[name]
            if not isinstance(digest,str) or len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest):raise ValueError('payload digest')
        cursors[key]=offset+size;count+=1
    if set(cursors)!=set(schema) or any(cursors[k]!=v['bytes'] for k,v in schema.items()):raise ValueError('incomplete rank schema')
    return dict(tensors=len(schema),chunks=count,bytes=sum(cursors.values()),encoded_bytes=encoded)
