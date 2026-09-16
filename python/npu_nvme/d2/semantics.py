"""Strict structural validation before accepting a checksummed D2 catalog."""
import re
HASH=re.compile(r'^[0-9a-f]{64}$')
def obj(value,keys):
    if type(value) is not dict or set(value)!=set(keys):raise ValueError('invalid object fields')
def integer(value,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError('invalid integer')
def digest(value):
    if type(value) is not str or not HASH.fullmatch(value):raise ValueError('invalid digest')
def ref(value,payload=False):
    obj(value,('offset','length','sha256','logical_bytes','logical_sha256') if payload else ('offset','length','sha256'))
    integer(value['offset']);integer(value['length'],1);digest(value['sha256'])
    if payload:
        integer(value['logical_bytes'],1);digest(value['logical_sha256'])
        if value['logical_bytes']>value['length']:raise ValueError('logical payload bounds')
def catalog(anchor,state,retention,*,allow_shared=False):
    obj(anchor,('sequence','region_id','catalog'));integer(anchor['sequence'],1);ref(anchor['catalog'])
    obj(state,('sequence','generations','receipts'));integer(state['sequence'],1)
    if type(state['generations']) is not list or not 1<=len(state['generations'])<=retention:raise ValueError('generation count')
    if type(state['receipts']) is not list or not 1<=len(state['receipts'])<=128:raise ValueError('receipt count')
    sequences=[];physical=[(anchor['catalog'],'catalog')]
    for g in state['generations']:
        obj(g,('generation','step','topology','pages','extent_pages','control_pages','extents','manifest_sha256'))
        integer(g['generation'],1);integer(g['step']);digest(g['manifest_sha256'])
        if type(g['topology']) is not dict:raise ValueError('topology object')
        if type(g['pages']) is not list or type(g['extents']) is not list:raise ValueError('generation arrays')
        sequences.append(g['generation'])
        for p in g['pages']+g['extent_pages']+g['control_pages']:ref(p);physical.append((p,'page'))
        for p in g['extents']:ref(p,True);physical.append((p,'payload'))
    if sequences[0]!=state['sequence'] or sequences!=sorted(set(sequences),reverse=True):raise ValueError('generation ordering')
    ids=set();previous=0
    for r in state['receipts']:
        obj(r,('request_id','identity','generation'));integer(r['generation'],1)
        if type(r['request_id']) is not str or not 0<len(r['request_id'])<=128 or r['request_id'] in ids:raise ValueError('receipt request identity')
        if not previous<r['generation']<=state['sequence']:raise ValueError('receipt ordering')
        previous=r['generation'];ids.add(r['request_id'])
    if previous!=state['sequence']:raise ValueError('latest receipt missing')
    # No implicit sharing: each immutable generation owns distinct extents.
    physical.sort(key=lambda p:p[0]['offset'])
    for (left,kind),(right,next_kind) in zip(physical,physical[1:]):
        if left['offset']+left['length']>right['offset']:
            if not (allow_shared and kind==next_kind=='payload' and left==right):raise ValueError('overlapping physical extents')
