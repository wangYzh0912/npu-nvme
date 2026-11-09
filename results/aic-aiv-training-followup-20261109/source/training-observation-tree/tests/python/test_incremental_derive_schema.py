import pytest

from npu_nvme.experiments.derive_schema import derive_model_schema


def test_derives_sharded_axis_and_replica():
    template=dict(tensors=[
        dict(name='w',role='model',partition='sharded',local_shape=[2,3],global_shape=[8,3],tensor_map=[0,-1]),
        dict(name='n',role='model',partition='replicated',local_shape=[3],global_shape=[3],tensor_map=[-1])])
    value=derive_model_schema(template,{'w':[4,5],'n':[5]},source_run='aux')
    assert value['tensors'][1]['global_shape']==[16,5]
    assert value['tensors'][1]['shards'][3]['start']==[12,0]
    assert value['tensors'][0]['global_shape']==[5]
    with pytest.raises(ValueError):derive_model_schema(template,{'w':[4,5]},source_run='aux')


def test_tied_output_can_be_absent_only_when_declared():
    template=dict(tensors=[dict(name=name,role='model',partition='sharded',
        local_shape=[2,3],global_shape=[8,3],tensor_map=[0,-1])
        for name in ('embedding.word_embeddings.weight','output_layer.weight')])
    shapes={'embedding.word_embeddings.weight':[4,5]}
    with pytest.raises(ValueError):derive_model_schema(template,shapes,source_run='aux')
    value=derive_model_schema(template,shapes,source_run='aux',tied_embeddings=True)
    assert len(value['tensors'])==1
