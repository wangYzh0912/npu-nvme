"""Regression cases exercise the production Python path without device I/O."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import npu_nvme.storage.chunks as chunk_helpers
from npu_nvme.types import validate_result_gate






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




@pytest.mark.parametrize('field,value',[('size',0),('size',1<<64),('ptr',0),('ptr',1<<64),('offset',(1<<64)-4096),('name','x'*1025)])
def test_input_extremes_and_descriptor_budget(field,value):
    item=dict(ptr=4096,size=8192,offset=0,name='x');item[field]=value
    with pytest.raises(ValueError):chunk_helpers.build_chunks([item],4096)


def test_last_byte_pointer_and_partial_chunk_are_legal():
    chunks,total=chunk_helpers.build_chunks([dict(ptr=chunk_helpers.MAX_POINTER,size=1,offset=4096)],4096)
    assert total==1 and len(chunks)==1 and chunks[0][0].value==chunk_helpers.MAX_POINTER
