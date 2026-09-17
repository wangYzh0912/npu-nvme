"""Device consumer experiment: padded TP partial buffers, no persistence."""
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter, ParameterTuple
from mindspore.ops import functional as F
from mindspore.ops.operations import Morph
from mindspore.common.initializer import initializer
from mindformers.parallel_core.training_graph.device_matrix import layout
from npu_nvme.experiments.graph_chain import weight_layout
from npu_nvme.experiments.graph_pack_geometry import packing_geometry


class CandidateTiles(nn.Cell):
    def __init__(self, tensor, mapping, unit):
        super().__init__(auto_prefix=False)
        self.unit = unit
        self.count = len(mapping['selected_local_tiles'])
        self.indices = Tensor(np.asarray(mapping['selected_local_tiles'],np.int32))
        self.morph = Morph(self.tiles,self.infer_shape,self.infer_dtype).add_prim_attr('self_define_shard',True)
        self.morph.shard(in_strategy=(weight_layout(tensor),),out_strategy=(layout('None','None'),))

    def infer_shape(self, shape):return (self.count,self.unit)
    def infer_dtype(self, dtype):return dtype
    def tiles(self, weight):return ops.gather(ops.reshape(weight,(-1,self.unit)),self.indices,0)
    def construct(self, weight):return self.morph(weight)


class ReferenceProjection(nn.Cell):
    def __init__(self, tensor, mapping, unit):
        super().__init__(auto_prefix=False)
        self.unit=unit
        self.mask_indices=Tensor(np.asarray(mapping['reference_mask_indices'],np.int32))
        self.morph=Morph(self.project,self.infer_shape,self.infer_dtype).add_prim_attr('self_define_shard',True)
        self.morph.shard(in_strategy=(weight_layout(tensor),weight_layout(tensor),layout('None')),
                         out_strategy=(weight_layout(tensor),))

    def infer_shape(self,w,r,m):return w
    def infer_dtype(self,w,r,m):return w
    def project(self,weight,reference,mask):
        take=ops.reshape(ops.gather(mask,self.mask_indices,0),(-1,1))
        result=ops.where(take,ops.reshape(weight,(-1,self.unit)),ops.reshape(reference,(-1,self.unit)))
        return ops.reshape(result,reference.shape)
    def construct(self,w,r,mask):return self.morph(w,r,mask)


class DeviceConsumer(nn.Cell):
    def __init__(self,sources,references,schema,rows,rank,k,advance):
        super().__init__(auto_prefix=False)
        tables=packing_geometry(schema,rows,rank)
        tensors={t['name']:t for t in schema['tensors'] if t['role']=='model'}
        self.sources=ParameterTuple(sources)
        self.references=ParameterTuple(references)
        self.present=tuple(i for i,m in enumerate(tables['parameters']) if m['selected_local_tiles'])
        self.tiles=nn.CellList([CandidateTiles(tensors[rows[i]['name']],tables['parameters'][i],tables['unit'])
                               for i in self.present],auto_prefix=False)
        self.projections=nn.CellList([ReferenceProjection(tensors[r['name']],m,tables['unit'])
                                      for r,m in zip(rows,tables['parameters'])],auto_prefix=False)
        self.lookup=Tensor(np.asarray(tables['lookup'],np.int32))
        self.zero=Tensor(np.zeros((1,tables['unit']),np.float32))
        self.output=Parameter(initializer('zeros',(k,65536),ms.float32),name='graph_device_output',requires_grad=False)
        self.empty_mask=Tensor(np.zeros(len(tables['lookup'])+1,np.bool_))
        self.ones=Tensor(np.ones(k,np.bool_))
        self.advance=advance
        self.k=k
        self.budget=dict(output_bytes=k*65536*4,candidate_buffer_bytes=tables['candidate_buffer_elements']*4,
                         lookup_bytes=self.lookup.nbytes,packing=tables['packing'],
                         reference_update_implementation='full-sized where + assign; unselected values preserved exactly')

    def construct(self,indices,ready):
        pieces=(self.zero,)
        for j in range(len(self.tiles)):
            pieces+=(self.tiles[j](F.depend(self.sources[self.present[j]],ready)),)
        candidates=ops.concat(pieces,0)
        tile_indices=ops.gather(self.lookup,indices,0)
        packed=ops.reshape(ops.gather(candidates,tile_indices,0),(self.k,65536))
        stored=F.assign(self.output,packed)
        consumed=ops.ReduceSum()(F.depend(self.output,stored))
        if self.advance:
            mask=ops.tensor_scatter_update(self.empty_mask,ops.reshape(indices+1,(-1,1)),self.ones)
            mask=F.depend(mask,consumed)
            updates=()
            for i in range(len(self.projections)):
                new_reference=self.projections[i](self.sources[i],self.references[i],mask)
                updates+=(F.assign(self.references[i],new_reference),)
            consumed=F.depend(consumed,updates)
        return consumed
