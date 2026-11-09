import json
import numpy as np
import pytest
from npu_nvme.runtime.training_catalog import write_checked
from npu_nvme.experiments.media_shadow import MediaShadow
from npu_nvme.experiments.tp_blocks import logical_fragments


def setup(tmp_path):
    tensors=[dict(name='w',role='model',partition='sharded',global_shape=[4,8],local_shape=[4,2],
                  shards=[dict(rank=r,start=[0,2*r],end=[4,2*r+2]) for r in range(4)]),
             dict(name='n',role='model',partition='replicated',global_shape=[2],local_shape=[2])]
    schema=tmp_path/'schema.json';schema.write_text(json.dumps(dict(tensors=tensors)))
    initial=tmp_path/'initial'
    for rank in range(4):
        directory=initial/f'rank_{rank}';directory.mkdir(parents=True);entries={}
        for row in tensors:
            a=np.zeros(row['local_shape'],np.float32);name=row['name'];np.save(directory/f'{name}.npy',a)
            entries[name]=dict(file=f'{name}.npy',bytes=a.nbytes)
        write_checked(directory/'manifest.json',dict(tensors=entries))
    return MediaShadow(initial,schema,tmp_path/'shadow',8),tensors


def frames(tensors,rank):
    records=[];payload=b''
    for tensor in tensors:
        name=tensor['name'];values=(np.arange(32,dtype=np.float32).reshape(4,8)[:,rank*2:rank*2+2].copy()
                                   if name=='w' else np.array([7,8],np.float32))
        for fragment in logical_fragments(tensor,8):
            if fragment['rank']!=rank or (not fragment['small'] and fragment['block_index'] not in (0,2)):continue
            start=fragment['local_element_offset'];count=fragment['element_count'];raw=values.reshape(-1)[start:start+count].tobytes()
            records.append(dict(name=name,element_offset=start,element_count=count,
                                payload_offset=len(payload),payload_bytes=len(raw),fragments=[fragment]))
            payload+=raw
    return dict(rank=rank,logical_step=1,records=records),payload


def test_actual_byte_stream_replays_across_value_and_record_boundaries(tmp_path):
    shadow,tensors=setup(tmp_path);shadow.begin(1)
    for rank in range(4):
        descriptor,payload=frames(tensors,rank)
        for offset in range(0,len(payload),3):shadow.consume(rank,descriptor,offset,payload[offset:offset+3])
    shadow.finish()
    reconstructed=np.concatenate([shadow.arrays[r]['w'] for r in range(4)],axis=1)
    expected=np.zeros((4,8),np.float32);expected[[0,2]]=np.arange(32,dtype=np.float32).reshape(4,8)[[0,2]]
    assert np.array_equal(reconstructed,expected)
    assert all(np.array_equal(shadow.arrays[r]['n'],[7,8]) for r in range(4))


def test_rejects_wrong_logical_mapping_before_ready_publication(tmp_path):
    shadow,tensors=setup(tmp_path);shadow.begin(1);descriptor,payload=frames(tensors,0)
    descriptor['records'][0]['fragments'][0]['global_element_offset']=999
    with pytest.raises(ValueError,match='mapping'):shadow.consume(0,descriptor,0,payload)
    assert not (shadow.output/'ready.json').exists()


def test_shadow_mapping_cache_is_bounded_and_flushes_evicted_values(tmp_path):
    from npu_nvme.experiments.media_shadow import ShadowArrays
    paths={}
    for i in range(40):
        p=tmp_path/f'{i}.npy';np.save(p,np.zeros(2,np.float32));paths[str(i)]=p
    arrays=ShadowArrays(paths,limit=2)
    for i in range(40):
        arrays[str(i)][0]=i+1
        assert len(arrays.cache)<=2
    for i in range(40):assert arrays[str(i)][0]==i+1
