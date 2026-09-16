"""Regression cases exercise the production Python path without device I/O."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import chunk_helpers
import direct_checkpoint as checkpoint
from full_checkpoint_protocol import CheckpointState, validate_result_gate


def test_wait_timeout_preserves_request_and_allows_late_completion():
    owner = SimpleNamespace()
    handle = checkpoint.CheckpointHandle(owner, 'r1', 1, 1)
    handle.state = CheckpointState.METADATA_COMMITTING
    before = handle.state
    with pytest.raises(TimeoutError):
        handle.wait(timeout=0)
    assert handle.state == before
    assert not handle.done()
    handle._complete()
    assert handle.wait(timeout=0) is handle


def test_cancelled_wait_is_not_timeout():
    handle = checkpoint.CheckpointHandle(SimpleNamespace(), 'r1', 1, 1)
    handle.status = handle.CANCELLED
    handle.state = CheckpointState.CANCELLED
    handle._done.set()
    with pytest.raises(RuntimeError, match='cancel'):
        handle.wait(timeout=0)


@pytest.mark.parametrize('changes', [{}, {'persisted': 1}, {'restore_verified': 'no'}])
def test_result_requires_explicit_status_and_boolean_proofs(changes):
    result = dict(mode='serial', request_id='r1', generation=1,
                  persisted=True, restore_verified=True)
    if changes:
        result['status'] = 'pass'
        result.update(changes)
    with pytest.raises(ValueError):
        validate_result_gate(result)


@pytest.mark.parametrize('field,value', [('size', -1), ('offset', -4096),
                                       ('offset', 1), ('ptr', -1)])
def test_chunk_input_rejected_before_ctypes(field, value):
    item = dict(ptr=4096, size=4, offset=4096, name='x')
    item[field] = value
    with pytest.raises(ValueError):
        chunk_helpers.build_chunks([item], 4096)


def test_zero_chunk_rejected_before_loop(monkeypatch):
    count = 0
    pointer = chunk_helpers.ctypes.c_void_p
    def bounded_pointer(value):
        nonlocal count
        count += 1
        if count > 5:
            raise AssertionError('chunk loop did not reject zero chunk size')
        return pointer(value)
    monkeypatch.setattr(chunk_helpers.ctypes, 'c_void_p', bounded_pointer)
    with pytest.raises(ValueError):
        chunk_helpers.build_chunks([dict(ptr=4096, size=4, offset=4096)], 0)


def test_restore_size_mismatch_rejected_before_allocation(monkeypatch):
    model = SimpleNamespace(parameters_and_names=lambda: [('x', SimpleNamespace(shape=(1,)))])
    monkeypatch.setattr(checkpoint, 'get_dev_ptr', lambda param: 0)
    def forbidden(*args, **kwargs):
        raise AssertionError('allocated unvalidated metadata')
    monkeypatch.setattr(np, 'empty', forbidden)
    with pytest.raises(ValueError):
        chunk_helpers.rebuild_chunks_from_meta(model, {'x':dict(shape=[1],dtype='float32',size=4096,offset=4096)},4096)


@pytest.mark.parametrize('field,value',[('size',0),('size',1<<64),('ptr',0),('ptr',1<<64),('offset',(1<<64)-4096),('name','x'*1025)])
def test_input_extremes_and_descriptor_budget(field,value):
    item=dict(ptr=4096,size=8192,offset=0,name='x');item[field]=value
    with pytest.raises(ValueError):chunk_helpers.build_chunks([item],4096)


def test_last_byte_pointer_and_partial_chunk_are_legal():
    chunks,total=chunk_helpers.build_chunks([dict(ptr=chunk_helpers.MAX_POINTER,size=1,offset=4096)],4096)
    assert total==1 and len(chunks)==1 and chunks[0][0].value==chunk_helpers.MAX_POINTER


def test_restore_chunk_budget_checked_before_host_allocation(monkeypatch):
    size=(chunk_helpers.MAX_BATCH_ITEMS+1)*4096
    model=SimpleNamespace(parameters_and_names=lambda:[('x',SimpleNamespace(shape=(size,)))])
    monkeypatch.setattr(checkpoint,'get_dev_ptr',lambda p:0)
    def forbidden(*a,**kw):raise AssertionError('allocated before item budget validation')
    monkeypatch.setattr(np,'empty',forbidden)
    with pytest.raises(ValueError):chunk_helpers.rebuild_chunks_from_meta(model,dict(x=dict(shape=[size],dtype='uint8',size=size,offset=0)),4096)
