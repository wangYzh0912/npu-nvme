"""Authoritative fixed-topology TP geometry, independent of framework imports."""
import hashlib
import json
import math

DTYPES={'F32':'float32','F16':'float16','BF16':'bfloat16','I64':'int64','I32':'int32','BOOL':'bool'}

def digest(schema):
    return hashlib.sha256(json.dumps(schema,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate(schema):
    if schema.get('topology')!={'tp':4,'dp':1,'pp':1}:
        raise ValueError('authoritative TP topology differs')
    tensors={}
    for row in schema['tensors']:
        name=row['name']
        if row['partition']=='per_rank_control':
            if name in tensors or row['role']!='per_rank_control_scalar' or not 0<row['logical_bytes_per_rank']<=1024:
                raise ValueError('invalid authoritative control')
            tensors[name]=row
            continue
        shape=row['global_shape'];local=row['local_shape'];parts=row['shards']
        if name in tensors or not name or row['partition'] not in ('replicated','sharded'):
            raise ValueError('duplicate tensor or partition')
        if any(type(v) is not int or v<=0 for v in [*shape,*local]) or len(shape)!=len(local):
            raise ValueError('invalid global/local shape')
        if len(parts)!=4 or {p['rank'] for p in parts}!={0,1,2,3}:
            raise ValueError('incomplete authoritative shard ranks')
        volume=0
        for p in parts:
            start,end=p['start'],p['end']
            if len(start)!=len(shape) or len(end)!=len(shape) or any(type(v) is not int for v in start+end):
                raise ValueError('invalid shard coordinates')
            if any(a<0 or a>=b or b>n for a,b,n in zip(start,end,shape)) or [b-a for a,b in zip(start,end)]!=local:
                raise ValueError('shard extent/local shape differs')
            if row['partition']=='replicated' and (start!=[0]*len(shape) or end!=shape):
                raise ValueError('replicated extent differs')
            volume+=math.prod(local)
        if row['partition']=='sharded':
            for i,a in enumerate(parts):
                for b in parts[i+1:]:
                    if all(max(x,y)<min(u,v) for x,y,u,v in zip(a['start'],b['start'],a['end'],b['end'])):
                        raise ValueError('overlapping TP shards')
            if volume!=math.prod(shape):raise ValueError('incomplete global coverage')
        tensors[name]=row
    if not tensors:raise ValueError('empty authoritative schema')
    for row in tensors.values():
        if row['role'] in ('adam_m','adam_v'):
            model=tensors.get(row['model_parameter'])
            if model is None or any(row[k]!=model[k] for k in ('global_shape','local_shape','partition')):
                raise ValueError('optimizer/model partition differs')
            if [(p['rank'],p['start'],p['end']) for p in row['shards']]!=[(p['rank'],p['start'],p['end']) for p in model['shards']]:
                raise ValueError('optimizer/model shard mapping differs')
    return tensors

def validate_rank(rows,schema,rank):
    expected=validate(schema);seen=set()
    for row in rows:
        name=row['name']
        if name in seen or row['rank']!=rank:raise ValueError('runtime tensor rank/identity differs')
        seen.add(name)
        if name not in expected:
            if row['partition']!='per_rank_control' or row['bytes']>1024:
                raise ValueError('unexpected runtime TP tensor')
            continue
        item=expected[name]
        if row['shape']!=item['local_shape'] or row['dtype']!=DTYPES.get(item['dtype'],item['dtype']) or row['partition']!=item['partition'] or row['bytes']!=item['logical_bytes_per_rank']:
            raise ValueError('runtime/authoritative tensor geometry differs: '+name)
    if not set(expected)<=seen:raise ValueError('runtime missing authoritative tensors')
