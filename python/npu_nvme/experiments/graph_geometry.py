"""Real logical-block subsets and their TP-local contiguous tiles (no runtime)."""
import math
from functools import reduce

from npu_nvme.experiments.tp_blocks import logical_fragments


def geometry(schema, rank, fraction=1.0, block_elements=65536):
    if rank not in range(4) or fraction not in (0.125, 0.25, 0.5, 1.0):
        raise ValueError('unsupported rank or scan fraction')
    stride = round(1 / fraction)
    rows = []
    base = 0
    for parameter_index, tensor in enumerate(sorted(
            (t for t in schema['tensors'] if t['role'] == 'model'), key=lambda t: t['name'])):
        total = math.prod(tensor['global_shape'])
        if total < block_elements:
            continue  # Fixed small-parameter policy: excluded from candidate detection.
        blocks = math.ceil(total / block_elements)
        # Every parameter/type/layer contributes; cycle the phase by parameter.
        chosen = list(range(parameter_index % min(stride, blocks), blocks, stride))
        selected = {b: i for i, b in enumerate(chosen)}
        fragments = [f for f in logical_fragments(tensor, block_elements)
                     if f['rank'] == rank and f['block_index'] in selected]
        all_local = [f for f in logical_fragments(tensor, block_elements) if f['rank'] == rank]
        unit = reduce(math.gcd, (f['element_count'] for f in all_local), block_elements)
        tile_indices, segment_ids, slot_ids = [], [], []
        for fragment in fragments:
            first = fragment['local_element_offset'] // unit
            count = fragment['element_count'] // unit
            if fragment['local_element_offset'] % unit:
                raise ValueError('unaligned tile')
            tile_indices.extend(range(first, first + count))
            segment_ids.extend([selected[fragment['block_index']]] * count)
            first_slot = (fragment['global_element_offset'] % block_elements) // unit
            slot_ids.extend(range(first_slot, first_slot + count))
        rows.append(dict(name=tensor['name'], parameter_index=parameter_index,
                         local_shape=tensor['local_shape'], unit=unit,
                         tile_indices=tile_indices, segment_ids=segment_ids, slot_ids=slot_ids,
                         global_blocks=chosen, score_offset=base,
                         block_count=len(chosen), fragment_count=len(fragments),
                         elements=len(tile_indices) * unit,
                         all_local_elements=math.prod(tensor['local_shape']),
                         global_elements=total, partition=tensor['partition']))
        base += len(chosen)
    return rows


def summarize(rows):
    return dict(candidate_blocks=sum(r['block_count'] for r in rows),
                local_elements=sum(r['elements'] for r in rows),
                local_input_bytes=8 * sum(r['elements'] for r in rows),
                local_reference_bytes=4 * sum(r['all_local_elements'] for r in rows),
                tp_fragments=sum(r['fragment_count'] for r in rows),
                parameters=[{k: v for k, v in r.items() if k not in ('tile_indices', 'segment_ids', 'slot_ids', 'global_blocks')}
                            for r in rows])
