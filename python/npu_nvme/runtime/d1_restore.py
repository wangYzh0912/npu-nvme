"""Pinned, bounded strict restore; observation timeout never frees an I/O lease."""
from __future__ import annotations
import hashlib
import math
import threading
import time
import uuid
from npu_nvme.types import RestoreReceipt
from .d1_schema import validate_record, spec_validate


class StrictRestoreError(RuntimeError): pass


class ObservationTimeout(TimeoutError):
    def __init__(self, handle):
        super().__init__(f'observation timed out; operation {handle.request_id} is still owned')
        self.handle = handle


class RestoreHandle:
    """A target cannot escape this handle until verification and ready succeed."""
    def __init__(self, operation, *, request_id=None, clock=time.monotonic):
        self.request_id = request_id or str(uuid.uuid4())
        self._clock = clock
        self._done = threading.Event()
        self._result = None
        self._error = None
        self._thread = threading.Thread(target=self._run, args=(operation,), daemon=False)

    def _run(self, operation):
        try:
            self._result = operation(self.request_id)
        except BaseException as error:
            self._error = error
        finally:
            self._done.set()

    def start(self):
        self._thread.start()
        return self

    @property
    def done(self): return self._done.is_set()

    def result(self, deadline=None):
        deadline = self._clock() + 120 if deadline is None else float(deadline)
        if not math.isfinite(deadline): raise ValueError('deadline must be finite monotonic absolute time')
        if not self._done.wait(max(0, deadline - self._clock())):
            raise ObservationTimeout(self)
        if self._error is not None: raise self._error
        return self._result


class StrictRestoreSession:
    def __init__(self, *, coordinator, reader, layout, quiescent=lambda:True, applier=None):
        self.coordinator, self.reader, self.layout = coordinator, reader, layout
        self.quiescent = quiescent
        self.applier = applier
        self.quarantined = []

    def restore_full_state(self, target_factory, expected_spec, step=None, *, request_id=None):
        spec_validate(expected_spec)
        with self.coordinator.selected(step) as record:
            # No target creation or payload read is allowed before planning.
            total, count = validate_record(record, self.layout, expected_spec)
            target = target_factory(expected_spec)
            try:
                if target.ready: raise StrictRestoreError('factory returned a ready target')
                target.prepare_restore(expected_spec)
                overall = hashlib.sha256()
                read_bytes = read_chunks = 0
                for name in sorted(record['params']):
                    info = record['params'][name]
                    whole = hashlib.sha256()
                    for chunk in info['chunks']:
                        data = self.reader(info['offset'] + chunk['offset'], chunk['size'])
                        if not isinstance(data, (bytes, bytearray, memoryview)) or len(data) != chunk['size']:
                            raise StrictRestoreError('reader returned an invalid byte count')
                        if hashlib.sha256(data).hexdigest() != chunk['sha256']:
                            raise StrictRestoreError(f'chunk checksum mismatch: {name}@{chunk["offset"]}')
                        whole.update(data); overall.update(data)
                        if self.applier is None: target.apply_chunk(name, chunk['offset'], data)
                        else: self.applier(target,name,chunk['offset'],data)
                        read_bytes += len(data); read_chunks += 1
                    if whole.hexdigest() != info['sha256']:
                        raise StrictRestoreError(f'tensor checksum mismatch: {name}')
                if (read_bytes, read_chunks) != (total, count) or overall.hexdigest() != record['data_sha256']:
                    raise StrictRestoreError('FULL stream digest mismatch')
                target.finish_restore(expected_spec)
                if not target.verify_controls(expected_spec):
                    raise StrictRestoreError('control state readback differs')
                if not self.quiescent(): raise StrictRestoreError('restore transport is not quiescent')
                target.mark_ready()
                if target.ready is not True: raise StrictRestoreError('target did not publish ready')
                return target, RestoreReceipt(request_id or str(uuid.uuid4()), record['generation'],
                    record['state_step'], read_bytes, read_chunks, overall.hexdigest(), True,
                    tuple(expected_spec['control_names']))
            except BaseException:
                target.ready = False
                if self.quiescent() and getattr(target, 'transport_safe', True):
                    target.discard()
                else:
                    # Retain the selected context manager and its pin until
                    # the runtime owner has a positive transport-stop proof.
                    self.quarantined.append(target)
                    # Retain an additional pin: selected() releases only its
                    # original reference when unwinding this failure.
                    self.coordinator.quarantine_pin(record['generation'])
                raise
