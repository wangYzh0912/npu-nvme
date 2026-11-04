"""CPU tensor checks; model/cache-writer acceptance is a separate NPU check."""
import pytest

torch=pytest.importorskip('torch')
from tools.aic_kv.geometry import dense_cache_views


def test_same_storage_dense_views_keep_k_v_and_blocks_disjoint():
    raw=torch.zeros((4,2,3,2,2))
    key,value=dense_cache_views(raw[:,0],raw[:,1])
    assert key.is_contiguous() and value.is_contiguous()
    assert key.untyped_storage().data_ptr()==raw.untyped_storage().data_ptr()
    key[2,1].fill_(7);value[3,0].fill_(11)
    assert key[2,1].eq(7).all() and value[3,0].eq(11).all()
    assert key.sum()==28 and value.sum()==44
    assert value[2,1].eq(0).all() and key[3,0].eq(0).all()


def test_reject_padded_backing_without_reinterpreting_other_regions():
    backing=torch.zeros(4*2*3*2*2+8)
    views=backing[:4*2*3*2*2].view(4,2,3,2,2)
    with pytest.raises(ValueError,match='unpadded'):
        dense_cache_views(views[:,0],views[:,1])
