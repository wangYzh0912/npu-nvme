"""CPU geometry-only streams, never dereference the synthetic pointers."""
import pytest
from npu_nvme.storage.chunks import iter_chunk_windows


def test_whole_checkpoint_exceeds_request_limits_without_one_large_array():
    mib=1024**2;size=95520*mib+517
    windows=iter_chunk_windows([dict(ptr=4096,offset=0,size=size,name='state')],mib,max_items=4096)
    count=total=peak=0
    for window in windows:
        peak=max(peak,len(window));count+=len(window)
        for ptr,offset,length,name in window:
            assert offset.value==total and ptr.value==4096+total
            total+=length.value
    assert peak==4096 and count==95521 and total==size and total>64*1024**3


def test_request_bytes_checkpoint_bytes_and_logical_tail_are_distinct():
    params=[dict(ptr=4096,offset=0,size=9000,name='state')]
    windows=list(iter_chunk_windows(params,4096,max_bytes=8192,checkpoint_bytes=9000))
    assert [[x[2].value for x in w] for w in windows]==[[4096,4096],[808]]
    with pytest.raises(ValueError,match='checkpoint'):
        list(iter_chunk_windows(params,4096,checkpoint_bytes=8999))


@pytest.mark.parametrize('change',[dict(ptr=(1<<64)-1),dict(offset=(1<<64)-4096),dict(size=True)])
def test_overflow_rejected_before_window_allocation(change):
    item=dict(ptr=4096,offset=0,size=8192,name='state');item.update(change)
    with pytest.raises(ValueError):next(iter_chunk_windows([item],4096))
