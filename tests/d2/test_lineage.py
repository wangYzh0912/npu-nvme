import pytest
from npu_nvme.d2.lineage import plan_retention,retained_closure

def node(g,root=None,parent=None):
    return dict(generation=g,kind='FULL' if parent is None else 'DELTA',root=g if root is None else root,
        parent=parent,manifest_sha256='a'*64,extents=[dict(offset=g*4096,length=4096,sha256='b'*64)])

def test_retained_heads_keep_full_ancestors_and_pins():
    nodes={1:node(1),2:node(2,1,1),3:node(3,1,2),4:node(4),5:node(5,4,4),6:node(6,4,5)}
    result=plan_retention(nodes,[6,5,4,3],retention=2,pinned_heads=[3],fallback_heads=[4])
    assert result['retained']==[1,2,3,4,5,6] and result['physical_bytes']==6*4096
    result=plan_retention(nodes,[6,5,4,3],retention=2)
    assert result['retained']==[4,5,6] and result['reclaimable']==[1,2,3]

def test_generation_gaps_allowed_when_parent_is_explicit():
    assert retained_closure({1:node(1),1000:node(1000,1,1)},[1000],max_chain_length=2)=={1,1000}

@pytest.mark.parametrize('root,parent',[(2,1),(1,9),(1,2)])
def test_foreign_or_cyclic_parent_rejected(root,parent):
    with pytest.raises(ValueError):retained_closure({1:node(1),2:node(2,root,parent)},[2],max_chain_length=3)

def test_missing_ancestor_and_chain_admission():
    with pytest.raises(ValueError):retained_closure({2:node(2,1,1)},[2],max_chain_length=3)
    with pytest.raises(BufferError):retained_closure({1:node(1),2:node(2,1,1)},[2],max_chain_length=1)

def test_pending_retains_base_and_does_not_mutate_input():
    nodes={1:node(1),2:node(2)}
    result=plan_retention(nodes,[2,1],retention=2,pending=node(3,1,1))
    assert result['retained']==[1,2,3] and set(nodes)=={1,2}
