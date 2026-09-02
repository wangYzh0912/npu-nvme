import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))

from multirank_protocol import GlobalCommit, GlobalState


def commit():
    return GlobalCommit(world_size=2, step=10, generation=7)


def test_commit_requires_every_rank_and_publishes_shards():
    value = commit()
    value.prepare(0, 10, 7, "m0")
    assert value.state == GlobalState.PREPARING
    value.prepare(1, 10, 7, "m1")
    value.persisted_ready(0, "d0")
    assert value.state == GlobalState.PERSISTING
    value.persisted_ready(1, "d1")
    metadata = value.commit()
    assert metadata["state"] == "COMMITTED"
    assert metadata["ranks"]["1"]["data_digest"] == "d1"


@pytest.mark.parametrize("action", ["duplicate", "mismatch", "early_commit"])
def test_invalid_global_commit_aborts_or_rejects(action):
    value = commit()
    value.prepare(0, 10, 7, "m0")
    if action == "duplicate":
        with pytest.raises(ValueError):
            value.prepare(0, 10, 7, "m0-again")
        assert value.state == GlobalState.ABORTED
    elif action == "mismatch":
        with pytest.raises(ValueError):
            value.prepare(1, 11, 7, "m1")
        assert value.state == GlobalState.ABORTED
    else:
        with pytest.raises(RuntimeError):
            value.commit()
        assert value.state != GlobalState.COMMITTED


def test_abort_metadata_is_not_publishable():
    value = commit()
    value.prepare(0, 10, 7, "m0")
    value.prepare(1, 10, 7, "m1")
    value.abort("rank 1 checksum failure")
    assert value.state == GlobalState.ABORTED
    with pytest.raises(RuntimeError):
        value.commit()
    assert value.metadata()["error"] == "rank 1 checksum failure"

