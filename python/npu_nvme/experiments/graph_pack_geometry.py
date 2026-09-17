"""TP block packing tables; zero slots denote fragments owned by another rank."""
import math
from functools import reduce
from npu_nvme.experiments.tp_blocks import logical_fragments


def packing_geometry(schema, rows, rank, block_elements=65536):
    unit = reduce(math.gcd, (r['unit'] for r in rows), block_elements)
    slots = block_elements // unit
    count = sum(r['block_count'] for r in rows)
    lookup = [[0] * slots for _ in range(count)]
    tensors = {t['name']: t for t in schema['tensors'] if t['role']=='model'}
    offset = 1  # concatenated buffer starts with one zero tile
    result=[]
    for row in rows:
        expanded = [i * (row['unit']//unit) + j for i in row['tile_indices'] for j in range(row['unit']//unit)]
        packed_index = {tile: offset+i for i,tile in enumerate(expanded)}
        selected = {b:row['score_offset']+i for i,b in enumerate(row['global_blocks'])}
        reference_mask_indices = [0] * (row['all_local_elements']//unit)
        for fragment in logical_fragments(tensors[row['name']],block_elements):
            if fragment['rank'] != rank or fragment['block_index'] not in selected:continue
            block=selected[fragment['block_index']]
            for j in range(fragment['element_count']//unit):
                tile=fragment['local_element_offset']//unit+j
                slot=(fragment['global_element_offset']%block_elements)//unit+j
                if lookup[block][slot]:raise ValueError('overlapping TP fragment')
                lookup[block][slot]=packed_index[tile]
                reference_mask_indices[tile]=block+1
        result.append(dict(name=row['name'],selected_local_tiles=expanded,reference_mask_indices=reference_mask_indices))
        offset+=len(expanded)
    return dict(unit=unit,slots_per_block=slots,lookup=lookup,parameters=result,
                candidate_buffer_elements=offset*unit,
                packing='TP rank partials in fixed global block slots, zero padding for unowned fragments; 4x global dense output bytes')
