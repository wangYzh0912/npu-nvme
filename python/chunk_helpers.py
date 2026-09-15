"""Compatibility exports for chunk planning and framework reconstruction."""
from npu_nvme.storage.chunks import *


def rebuild_chunks_from_meta(models, params_meta, chunk_size):
    from direct_checkpoint import get_dev_ptr
    from npu_nvme.framework.capture import rebuild_chunks_from_meta as rebuild
    return rebuild(models, params_meta, chunk_size, pointer_of=get_dev_ptr)
