from types import SimpleNamespace
import threading

import pytest

from npu_nvme.experiments.raw_owner import RawOwner
from npu_nvme.experiments.raw_store import ALIGN, SUPER_BYTES


def test_owner_checks_offset_before_transport():
    writes=[]
    owner=RawOwner.__new__(RawOwner)
    owner.lock=threading.Lock()
    owner.config=dict(chunk_bytes=ALIGN)
    owner.transport=SimpleNamespace(total_bytes=SUPER_BYTES+ALIGN*2,
                                     write=lambda offset,data:writes.append((offset,data)))
    for offset in (0, SUPER_BYTES+1, SUPER_BYTES+ALIGN*2):
        with pytest.raises(ValueError):
            owner._write(offset,b'x')
    assert writes==[]
    owner._write(SUPER_BYTES,bytes(ALIGN))
    assert len(writes)==1
