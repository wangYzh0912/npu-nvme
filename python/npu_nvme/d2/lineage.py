"""Pure FULL/delta closure planning; does not publish durable receipts."""
import copy


def retained_closure(nodes,heads,*,max_chain_length):
    if type(max_chain_length) is not int or max_chain_length<=0:raise ValueError('chain budget')
    result=set()
    for head in heads:
        current=head;chain=[];expected_root=None;manifest=None
        while True:
            if type(current) is not int or current<=0 or current not in nodes:raise ValueError('missing lineage node')
            if len(chain)>=max_chain_length:raise BufferError('FULL compaction required before chain limit')
            node=nodes[current]
            if node.get('generation')!=current:raise ValueError('lineage identity')
            kind=node.get('kind');root=node.get('root');parent=node.get('parent')
            digest=node.get('manifest_sha256')
            if type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('lineage manifest digest')
            if manifest is None:manifest=digest;expected_root=root
            if digest!=manifest or root!=expected_root:raise ValueError('lineage schema/root changed')
            chain.append(current)
            if kind=='FULL':
                if parent is not None or root!=current:raise ValueError('invalid FULL root')
                break
            if kind!='DELTA' or type(parent) is not int or not 0<parent<current or type(root) is not int or not 0<root<=parent:
                raise ValueError('invalid delta parent')
            current=parent
        result.update(chain)
    return result


def plan_retention(nodes,heads,*,retention,fallback_heads=(),pinned_heads=(),pending=None,max_chain_length=128):
    """Return complete ancestors and physical extents for every protected head.

    A pending node is planned before publication. It cannot make any ancestor
    reclaimable until its anchor is durable and previous-reader pins are gone.
    """
    if retention not in (2,3):raise ValueError('retention')
    nodes=copy.deepcopy(nodes)
    if pending is not None:
        generation=pending.get('generation')
        if generation in nodes:raise ValueError('pending generation already committed')
        nodes[generation]=copy.deepcopy(pending)
    ordered=list(heads)
    if ordered!=sorted(set(ordered),reverse=True):raise ValueError('head ordering')
    selected=ordered[:retention]
    protected=set(selected)|set(fallback_heads)|set(pinned_heads)
    if pending is not None:protected.add(pending['generation'])
    retained=retained_closure(nodes,protected,max_chain_length=max_chain_length)
    extents={}
    for generation in retained:
        for extent in nodes[generation].get('extents',[]):
            offset=extent['offset'];length=extent['length'];digest=extent['sha256']
            if type(offset) is not int or type(length) is not int or offset<0 or length<=0 or offset%4096 or length%4096:
                raise ValueError('lineage extent geometry')
            key=(offset,length)
            if key in extents and extents[key]!=digest:raise ValueError('conflicting shared extent identity')
            extents[key]=digest
    previous_end=0
    for offset,length in sorted(extents):
        if offset<previous_end:raise ValueError('overlapping lineage extents')
        previous_end=offset+length
    return dict(heads=selected,retained=sorted(retained),reclaimable=sorted(set(nodes)-retained),
                physical_bytes=sum(length for offset,length in extents),
                extents=[dict(offset=o,length=n,sha256=extents[(o,n)]) for o,n in sorted(extents)])
