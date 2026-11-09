"""Logical blocks mapped onto contiguous TP shards without counting replicas."""
from __future__ import annotations

import math


def aggregate_scores(rank_rows):
    """Sum partial logical-block scores in a fixed rank order."""
    result = {}
    seen = set()
    for rank, rows in sorted(rank_rows.items()):
        for row in rows:
            key = (row['name'], row['block_index'])
            if (rank, key) in seen or not math.isfinite(row['score']) or row['score'] < 0:
                raise ValueError('duplicate or invalid logical partial score')
            seen.add((rank, key))
            result[key] = result.get(key, 0.0) + row['score']
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError('logical block score overflow')
    return result


def logical_fragments(tensor, block_elements=65536):
    """Return owned fragments keyed by global flattened parameter block.

    Rows of a column-sharded matrix may contribute to the same logical block
    on several ranks. Their squared-difference sums must be added before Top-K.
    """
    if tensor['role'] != 'model' or block_elements <= 0:
        raise ValueError('model tensor and positive block size required')
    shape = tensor.get('global_shape', tensor['local_shape'])
    total = math.prod(shape)
    if tensor['partition'] == 'replicated':
        return _split(tensor['name'], 0, 0, 0, total, total, block_elements)
    if tensor['partition'] != 'sharded':
        raise ValueError('unsupported model partition')
    strides = [math.prod(shape[i+1:]) for i in range(len(shape))]
    fragments = []
    for shard in tensor['shards']:
        start, end = shard['start'], shard['end']
        if len(start) != len(shape) or any(not 0 <= a < b <= n for a,b,n in zip(start,end,shape)):
            raise ValueError('invalid shard geometry')
        lengths = [b-a for a,b in zip(start,end)]
        # Collapse trailing dimensions that cover the global extent.
        axis = len(shape)-1
        while axis > 0 and start[axis] == 0 and end[axis] == shape[axis]:
            axis -= 1
        run = lengths[axis] * strides[axis]
        prefix_count = math.prod(lengths[:axis])
        for index in range(prefix_count):
            remainder = index; global_offset = start[axis]*strides[axis]
            for dim in reversed(range(axis)):
                coordinate = remainder % lengths[dim]; remainder //= lengths[dim]
                global_offset += (start[dim]+coordinate)*strides[dim]
            fragments.extend(_split(tensor['name'], shard['rank'], global_offset,
                                    index*run, run, total, block_elements))
    return fragments


def _split(name, rank, global_offset, local_offset, count, total, block):
    result=[]; cursor=0
    small=total < block
    while cursor<count:
        index=(global_offset+cursor)//block
        take=min(count-cursor, block-(global_offset+cursor)%block)
        result.append(dict(name=name,rank=rank,block_index=index,
            global_element_offset=global_offset+cursor,local_element_offset=local_offset+cursor,
            element_count=take,small=small))
        cursor+=take
    return result
