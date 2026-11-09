"""Views for zero-initialized, unbound ordinary GQA cache allocations."""


def dense_cache_views(key,value):
    if key.is_contiguous() and value.is_contiguous():return key,value
    lane=key.numel()//key.shape[0]
    if (key.shape!=value.shape or key.untyped_storage().data_ptr()!=value.untyped_storage().data_ptr()
        or key.stride()[0]!=2*lane or value.storage_offset()!=key.storage_offset()+lane
        or key.untyped_storage().nbytes()!=2*key.numel()*key.element_size()):
        raise ValueError('dense control requires an unpadded block-interleaved K/V allocation')
    strides=[];stride=1
    for size in reversed(key.shape):
        strides.append(stride);stride*=size
    strides=tuple(reversed(strides))
    return (key.as_strided(key.shape,strides,key.storage_offset()),
            value.as_strided(value.shape,strides,key.storage_offset()+key.numel()))
