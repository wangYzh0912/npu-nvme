import hashlib

import pytest

from npu_nvme.experiments.raw_store import (ALIGN, SUPER_BYTES, Layout,
    pack_commit, pack_frame_prefix, pack_superblock, unpack_commit,
    unpack_frame_prefix, unpack_superblock)


def test_superblock_frame_and_commit_roundtrip():
    config = "a" * 64
    superblock = pack_superblock(namespace_bytes=4 << 30, config_sha256=config, campaign_id="run")
    assert len(superblock) == SUPER_BYTES
    assert unpack_superblock(superblock)["config_sha256"] == config
    payload = b"payload"
    payload_sha = hashlib.sha256(payload).hexdigest()
    prefix = pack_frame_prefix(generation=2, step=7, descriptor={"blocks": [1]},
                               payload_bytes=len(payload), payload_sha256=payload_sha)
    assert len(prefix) % ALIGN == 0
    frame = unpack_frame_prefix(prefix)
    assert frame["payload_sha256"] == payload_sha and frame["descriptor"] == {"blocks": [1]}
    whole = prefix + payload
    digest = hashlib.sha256(whole).hexdigest()
    commit = pack_commit(generation=2, frame_offset=SUPER_BYTES,
                         frame_bytes=len(whole), frame_sha256=digest)
    assert unpack_commit(commit)["frame_sha256"] == digest

    large = pack_frame_prefix(generation=3, step=8, descriptor={"x": "a" * 9000},
                              payload_bytes=1, payload_sha256=hashlib.sha256(b"x").hexdigest())
    assert len(large) > ALIGN and unpack_frame_prefix(large)["descriptor"]["x"] == "a" * 9000


def test_layout_keeps_archive_and_double_scratch_disjoint():
    layout = Layout(4 << 40, 33 << 30)
    offset = layout.reserve_archive(2 << 40)
    assert offset == SUPER_BYTES
    first = layout.scratch_slot(0); second = layout.scratch_slot(1)
    assert first[0] + first[1] == second[0]
    assert layout.cursor <= first[0]
    with pytest.raises(ValueError):
        layout.scratch_slot(2)
    with pytest.raises(MemoryError):
        layout.validate_frame(layout.scratch_start - ALIGN, ALIGN, ALIGN)
    assert layout.validate_frame(SUPER_BYTES, ALIGN, 1) == SUPER_BYTES + ALIGN * 2
    assert layout.scratch_cursor(0, 1)[1]-layout.scratch_cursor(0, 1)[0]==ALIGN
    with pytest.raises(MemoryError):
        layout.scratch_cursor(0, layout.scratch_bytes)


def test_commit_rejects_non_digest_identity():
    with pytest.raises(ValueError):
        pack_commit(generation=1, frame_offset=SUPER_BYTES, frame_bytes=1, frame_sha256="x")
