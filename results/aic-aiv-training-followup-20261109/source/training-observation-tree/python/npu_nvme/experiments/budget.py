"""Parameter-inventory budgets, without materializing model tensors."""
import math
from collections import defaultdict


def estimate(schema, *, block_elements=65536, ratios=(.05,.1,.2), staging_bytes=256 << 20):
    scopes=defaultdict(lambda: dict(logical_bytes=0, physical_bytes=0, elements=0))
    blocks=[]
    small_bytes=0
    for row in schema['tensors']:
        role=row['role']
        local=row['logical_bytes_per_rank']
        logical=local if row['partition']=='replicated' else local*schema['topology']['tp']
        scopes[role]['logical_bytes']+=logical
        scopes[role]['physical_bytes']+=local*schema['topology']['tp']
        if role!='model':
            continue
        count=math.prod(row.get('global_shape',row['local_shape']))
        itemsize=local//math.prod(row['local_shape'])
        scopes[role]['elements']+=count
        if count<block_elements:
            small_bytes+=count*itemsize
        else:
            blocks.extend(min(block_elements,count-start)*itemsize for start in range(0,count,block_elements))
    weights=scopes['model']['logical_bytes']
    return dict(scope_bytes=dict(scopes), weight_bytes=weights, candidate_blocks=len(blocks),
                small_bytes=small_bytes, score_scalar_ops=3*scopes['model']['elements'],
                score_scan_bytes=2*(weights-small_bytes),
                full_state_scan_bytes=2*sum(row['physical_bytes'] for row in scopes.values()),
                outputs=[dict(ratio=q,selected_blocks=math.ceil(q*len(blocks)),
                              approximate_payload_bytes=q*(weights-small_bytes)+small_bytes,
                              metadata_bytes=None) for q in ratios],
                hbm_extra=dict(reference_physical_bytes=scopes['model']['physical_bytes'],
                               staging_physical_bytes=staging_bytes*schema['topology']['tp'],
                               score_bytes=len(blocks)*8,workspace_bytes=None,snapshot_bytes=0),
                note='Output estimates exclude metadata; actual selected tails determine exact bytes. '
                     'Workspace and framework temporaries require measurement.')
