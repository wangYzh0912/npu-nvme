import numpy as np
from npu_nvme.experiments.graph_geometry import geometry
from npu_nvme.experiments.graph_pack_geometry import packing_geometry
from test_graph_topk_geometry import tensor


def test_pack_and_reference_mask_cover_selected_owned_elements_only():
    schema={'tensors':[tensor('a',[16,48],1),tensor('b',[10,20],1),tensor('c',[32,16],0)]}
    arrays={t['name']:np.arange(np.prod(t['global_shape']),dtype=np.float32).reshape(t['global_shape'])+1 for t in schema['tensors']}
    combined=None
    chosen=[0,2]
    for rank in range(4):
        rows=geometry(schema,rank,.5,64)
        tables=packing_geometry(schema,rows,rank,64)
        sources=[np.zeros((1,tables['unit']),np.float32)]
        for t, mapping in zip(schema['tensors'],tables['parameters']):
            shard=t['shards'][rank]
            local=arrays[t['name']][tuple(slice(a,b) for a,b in zip(shard['start'],shard['end']))].reshape(-1,tables['unit'])
            sources.append(local[mapping['selected_local_tiles']])
            mask=np.isin(mapping['reference_mask_indices'],np.asarray(chosen)+1)
            # Updated references retain exactly all nonselected values.
            before=np.full_like(local,-99)
            after=np.where(mask[:,None],local,before)
            assert np.array_equal(after[~mask],before[~mask])
        source=np.concatenate(sources)
        packed=source[np.asarray(tables['lookup'])[chosen]].reshape(len(chosen),64)
        combined=packed if combined is None else combined+packed
    oracle=[]
    for index in chosen:
        row=next(r for r in rows if r['score_offset']<=index<r['score_offset']+r['block_count'])
        block=row['global_blocks'][index-row['score_offset']]
        values=arrays[row['name']].reshape(-1)[block*64:(block+1)*64]
        oracle.append(np.pad(values,(0,64-len(values))))
    np.testing.assert_array_equal(combined,np.asarray(oracle))
