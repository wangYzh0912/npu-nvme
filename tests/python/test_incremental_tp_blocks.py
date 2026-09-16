import pytest

from npu_nvme.experiments.tp_blocks import aggregate_scores, logical_fragments


def test_column_shards_share_logical_block_and_cover_it_once():
    tensor=dict(name='w',role='model',partition='sharded',global_shape=[4,8],local_shape=[4,4],
                shards=[dict(rank=0,start=[0,0],end=[4,4]),dict(rank=1,start=[0,4],end=[4,8])])
    fragments=logical_fragments(tensor,16)
    for block in range(2):
        rows=[r for r in fragments if r['block_index']==block]
        assert sum(r['element_count'] for r in rows)==16
        assert {r['rank'] for r in rows}=={0,1}
    for rank in range(2):
        rows=[r for r in fragments if r['rank']==rank]
        assert [r['local_element_offset'] for r in rows]==[0,4,8,12]


def test_replicated_parameter_has_only_one_owner():
    tensor=dict(name='norm',role='model',partition='replicated',local_shape=[8])
    rows=logical_fragments(tensor,16)
    assert len(rows)==1 and rows[0]['rank']==0 and rows[0]['small']


def test_partial_scores_are_merged_by_logical_identity():
    rows={0:[dict(name='w',block_index=0,score=2.0)],
          1:[dict(name='w',block_index=0,score=3.0),dict(name='w',block_index=1,score=1.0)]}
    assert aggregate_scores(rows)=={('w',0):5.0,('w',1):1.0}
    with pytest.raises(ValueError):
        aggregate_scores({0:rows[0]+rows[0]})
