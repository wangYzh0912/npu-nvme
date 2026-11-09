import hashlib
from types import SimpleNamespace

import pytest

from npu_nvme.experiments.raw_store import ALIGN,SUPER_BYTES,pack_frame_prefix,pack_commit
from npu_nvme.experiments.verify_media import verify_receipt


def test_readback_detects_corrupt_payload():
    disk=bytearray(SUPER_BYTES+ALIGN*20);cursor=SUPER_BYTES;rows=[]
    for rank in range(4):
        payload=bytes([rank])*ALIGN;sha=hashlib.sha256(payload).hexdigest()
        prefix=pack_frame_prefix(generation=1,step=1,descriptor=dict(rank=rank),payload_bytes=ALIGN,payload_sha256=sha)
        frame_sha=hashlib.sha256(prefix+payload).hexdigest();disk[cursor:cursor+len(prefix)+ALIGN]=prefix+payload
        commit_offset=cursor+len(prefix)+ALIGN
        disk[commit_offset:commit_offset+ALIGN]=pack_commit(generation=1,frame_offset=cursor,frame_bytes=len(prefix)+ALIGN,frame_sha256=frame_sha)
        rows.append(dict(rank=rank,frame_offset=cursor,prefix_bytes=len(prefix),payload_offset=cursor+len(prefix),
            payload_bytes=ALIGN,payload_sha256=sha,frame_bytes=len(prefix)+ALIGN,commit_offset=commit_offset,frame_sha256=frame_sha))
        cursor=commit_offset+ALIGN
    transport=SimpleNamespace(total_bytes=len(disk),read=lambda offset,size:bytes(disk[offset:offset+size]))
    receipt=dict(generation=1,step=1,ranks=rows)
    assert verify_receipt(transport,receipt,chunk_bytes=ALIGN)['status']=='pass'
    disk[rows[2]['payload_offset']]=255
    with pytest.raises(ValueError,match='checksum'):verify_receipt(transport,receipt,chunk_bytes=ALIGN)
