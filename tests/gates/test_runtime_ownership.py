import pytest
"""Ownership boundaries exercised with real scheduler threads and copied buffers."""
import threading
from types import SimpleNamespace

import numpy as np

from npu_nvme.framework.capture import FrozenCapture




def test_framework_capture_freezes_host_values_without_a_facade():
    source = np.arange(4, dtype=np.float32)
    capture = FrozenCapture(rank_id=0, device_id=0, framework=SimpleNamespace(),
                            acl=SimpleNamespace(aclrtMalloc=object()), pointer_of=lambda p: 0)
    frozen = capture.snapshot([dict(name='x', ptr=0, size=source.nbytes, np_arr=source)], 9)
    source[:] = -1
    np.testing.assert_array_equal(frozen[0]['np_arr'], np.arange(4, dtype=np.float32))
    assert frozen[0]['generation'] == 9
    capture.release(frozen)



def test_d1_reservation_is_single_owner_and_unknown_is_explicit():
    from npu_nvme.runtime.d1_commit import D1CommitCoordinator, D1CommitError, OutcomeUnknown
    from npu_nvme.runtime.commit import MetadataState
    class IO:
        def persist(self, state, generation=None): state.metadata_generation = generation
    from npu_nvme.storage.layout import make_layout
    layout = make_layout(total_bytes=1 << 30, full_slot_bytes=1 << 20,
                         full_slot_count=3, delta_slot_bytes=4096, delta_slot_count=1)
    c = D1CommitCoordinator(metadata_io=IO(), state=MetadataState(layout=layout,
        meta_dict={'strict_contract':'D1','catalog_revision':0,'checkpoints':{}}))
    with pytest.raises(D1CommitError): D1CommitCoordinator(metadata_io=IO(), state=MetadataState(), rank_id=1)
    r = c.reserve(step=2)
    with pytest.raises(OutcomeUnknown): c.resolve('missing')
    c.abort(r)
    assert isinstance(c.resolve(r.request_id), D1CommitError)


def test_current_drain_does_not_wait_for_later_handles():
    from npu_nvme.runtime.d1_runtime import FullRuntime
    runtime = object.__new__(FullRuntime)
    runtime.changed = threading.Condition()
    runtime._busy = False
    runtime.quarantined = []
    runtime.restore_session = SimpleNamespace(quarantined=[])
    first, later = object(), object()
    runtime._handles = {first}
    entered = threading.Event()
    original_wait = runtime.changed.wait
    def wait(timeout):
        entered.set()
        return original_wait(timeout)
    runtime.changed.wait = wait
    errors = []
    def drain():
        try: runtime.drain(2)
        except BaseException as error: errors.append(error)
    worker = threading.Thread(target=drain)
    worker.start()
    assert entered.wait(1)
    with runtime.changed:
        runtime._handles.add(later)
        runtime._handles.remove(first)
        runtime.changed.notify_all()
    worker.join(2)
    assert not worker.is_alive() and not errors
    assert runtime._handles == {later}
