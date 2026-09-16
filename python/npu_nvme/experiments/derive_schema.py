"""Derive another Qwen3 size while retaining the accepted TP partition axes."""
import math


def derive_model_schema(template, shapes, *, source_run, tied_embeddings=False):
    model={row['name']:row for row in template['tensors'] if row['role']=='model'}
    if tied_embeddings and 'output_layer.weight' not in shapes:
        model.pop('output_layer.weight',None)
    if set(model)!=set(shapes):
        raise ValueError('auxiliary model parameter names differ from TP template')
    rows=[]
    for name,row in sorted(model.items()):
        local=list(shapes[name])
        if len(local)!=len(row['local_shape']) or any(type(value) is not int or value<=0 for value in local):
            raise ValueError('auxiliary local rank differs: '+name)
        global_shape=list(local)
        if row['partition']=='sharded':
            axes=[i for i,(left,right) in enumerate(zip(row['global_shape'],row['local_shape'])) if left!=right]
            if len(axes)!=1 or row['global_shape'][axes[0]]!=4*row['local_shape'][axes[0]]:
                raise ValueError('template requires one TP4 shard axis')
            global_shape[axes[0]]*=4
        shards=[]
        for rank in range(4):
            start=[0]*len(local);end=list(global_shape)
            if row['partition']=='sharded':
                axis=axes[0];start[axis]=rank*local[axis];end[axis]=(rank+1)*local[axis]
            shards.append(dict(rank=rank,start=start,end=end))
        rows.append(dict(name=name,dtype='F32',local_shape=local,
            logical_bytes_per_rank=math.prod(local)*4,role='model',partition=row['partition'],
            global_shape=global_shape,device_matrix=[1,1,4],tensor_map=row['tensor_map'],shards=shards))
    return dict(schema_version=1,source_run=source_run,checkpoint_step=0,
                topology=dict(tp=4,dp=1,pp=1),scope='phase-one weights only',tensors=rows)
