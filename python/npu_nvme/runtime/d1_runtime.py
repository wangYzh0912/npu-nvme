"""FULL orchestration with injected capture and transport services."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import math
import threading
import time
from npu_nvme.types import TransferReceipt
from .d1_schema import spec_validate, align, manifest_digest, CHUNK_BYTES
from .d1_restore import RestoreHandle, ObservationTimeout, StrictRestoreSession
from .scheduler import CheckpointBusyError


class FullHandle(RestoreHandle):
    def __init__(self, reservation, operation):
        super().__init__(operation, request_id=reservation.request_id)
        self.generation = reservation.generation
        self.step = reservation.step
        self.status = 'DISPATCHED'
        self.snapshot_state_digest = None
        self.metadata_generation = None
        self.events = []

    def _run(self, operation):
        super()._run(operation)
        # result() uses the event; publish these fields in operation instead.

    def wait(self, timeout=None):
        self.result(time.monotonic() + (120 if timeout is None else float(timeout)))
        return self

    def as_dict(self):
        return dict(request_id=self.request_id, generation=self.generation, step=self.step,
                    status=self.status, state=self.status, metadata_generation=self.metadata_generation,
                    snapshot_state_digest=self.snapshot_state_digest, events=list(self.events),
                    api_enter_ns=self.api_enter_ns)


class FullRuntime:
    def __init__(self, coordinator, transport, *, prepare, freeze, release, chunk_size=CHUNK_BYTES):
        self.commit, self.transport = coordinator, transport
        self.prepare, self.freeze, self.release = prepare, freeze, release
        self.chunk_size = chunk_size
        self.changed = threading.Condition()
        self._busy = False
        self._closing = False
        self._handles = set()
        self._restore_busy = False
        self.quarantined = []
        self.restore_session = StrictRestoreSession(coordinator=coordinator, reader=transport.read,
            layout=coordinator.state.layout, quiescent=transport.quiescent)

    def save(self, components, controls, step, spec, *, admission='block', timeout=120):
        api_enter_ns = time.monotonic_ns()
        spec = deepcopy(spec_validate(spec))
        if admission not in ('block', 'try'): raise ValueError('admission must be block or try')
        timeout = 120 if timeout is None else float(timeout)
        if not math.isfinite(timeout) or timeout < 0: raise ValueError('invalid admission timeout')
        deadline = time.monotonic() + timeout
        with self.changed:
            while self._busy:
                if self._closing: raise RuntimeError('admission is closed')
                if admission == 'try': raise CheckpointBusyError('FULL slot is busy')
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise TimeoutError('admission timed out')
                self.changed.wait(remaining)
            if self._closing: raise RuntimeError('admission is closed')
            reservation = self.commit.reserve(step=step)
            self._busy = True
        frozen = None
        try:
            params = self.prepare(components, controls, spec)
            frozen = self.freeze(params, reservation.generation)
            handle = FullHandle(reservation, lambda rid: self._save_worker(handle, reservation, frozen, spec))
            handle.api_enter_ns = api_enter_ns
            handle.events.append(dict(state="CREATED", monotonic_ns=time.monotonic_ns()))
            with self.changed: self._handles.add(handle)
            handle.start()
            return handle
        except BaseException as error:
            try:
                if frozen is not None: self._release_or_retain(frozen)
            finally:
                self.commit.abort(reservation, error)
                with self.changed:
                    if 'handle' in locals(): self._handles.discard(handle)
                    self._busy = False
                    self.changed.notify_all()
            raise

    def _save_worker(self, handle, reservation, frozen, spec):
        try:
            base = self.commit.state.layout.full_base + reservation.slot * self.commit.state.layout.full_slot_bytes
            record = dict(strict_contract='D1', type='TRAINING_STATE_FULL', schema_version=1,
                          rank_id=0, world_size=1, request_id=reservation.request_id,
                          writer_epoch=reservation.writer_epoch, generation=reservation.generation,
                          state_step=reservation.step, slot=reservation.slot, chunk_size=self.chunk_size,
                          spec=spec, params={})
            overall = hashlib.sha256()
            total = count = 0
            items = sorted(frozen, key=lambda p:p['name'])
            for item in items:
                info = {k:item[k] for k in ('shape', 'dtype', 'size')}
                if item['name'].startswith('control/'): info['codec'] = 'json-tagged-v1'
                info.update(offset=base, chunks=[])
                whole = hashlib.sha256()
                for offset in range(0, item['size'], self.chunk_size):
                    size = min(self.chunk_size, item['size'] - offset)
                    data = self.transport.frozen_bytes(item, offset, size)
                    if len(data) != size: raise ValueError('frozen byte count differs')
                    whole.update(data); overall.update(data)
                    info['chunks'].append(dict(offset=offset, size=size, sha256=hashlib.sha256(data).hexdigest()))
                    total += size; count += 1
                info['sha256'] = whole.hexdigest()
                record['params'][item['name']] = info
                base += align(item['size'])
            record['data_sha256'] = overall.hexdigest()
            record['manifest_sha256'] = manifest_digest(record)
            self.commit.preflight(reservation, record)
            for item in items:
                info = record['params'][item['name']]
                for chunk in info['chunks']:
                    data = self.transport.frozen_bytes(item, chunk['offset'], chunk['size'])
                    if hashlib.sha256(data).hexdigest() != chunk['sha256']:
                        raise ValueError('frozen source changed before transport')
                    self.transport.write(info['offset'] + chunk['offset'], data)
            self.transport.flush()
            receipt = self.commit.publish(reservation, TransferReceipt(reservation.request_id,
                reservation.generation, total, count, True, overall.hexdigest()), record)
            handle.status = 'PERSISTED'
            handle.metadata_generation = receipt.metadata_generation
            handle.snapshot_state_digest = overall.hexdigest()
            handle.events.append(dict(state='PERSISTED', monotonic_ns=time.monotonic_ns()))
            return receipt
        except BaseException as error:
            handle.status = 'FAILED'
            self.commit.abort(reservation, error)
            raise
        finally:
            try:
                self._release_or_retain(frozen)
            finally:
                with self.changed:
                    self._busy = False
                    self._handles.discard(handle)
                    self.changed.notify_all()

    def _release_or_retain(self, frozen):
        try:
            if self.transport.quiescent():
                self.release(frozen)
                return
        except BaseException:
            self.quarantined.append(frozen)
            self.commit.poisoned = True
            raise
        self.quarantined.append(frozen)
        self.commit.poisoned = True

    def begin_restore(self, factory, spec, step=None):
        spec = deepcopy(spec_validate(spec))
        with self.changed:
            if self._closing: raise RuntimeError('admission is closed')
            if self._restore_busy: raise CheckpointBusyError('one restore session is already active')
            self._restore_busy = True
            def operation(request_id):
                try: return self.restore_session.restore_full_state(factory, spec, step, request_id=request_id)
                finally:
                    with self.changed:
                        self._restore_busy = False
                        self._handles.discard(handle)
                        self.changed.notify_all()
            handle = RestoreHandle(operation)
            self._handles.add(handle)
            try: return handle.start()
            except BaseException:
                self._handles.discard(handle)
                self._restore_busy = False
                raise

    def drain(self, timeout=120):
        timeout = 120 if timeout is None else float(timeout)
        if not math.isfinite(timeout) or timeout < 0: raise ValueError('invalid drain timeout')
        deadline = time.monotonic() + timeout
        with self.changed:
            watermark = set(self._handles)
            while watermark & self._handles or self._busy and not watermark:
                remaining = deadline - time.monotonic()
                if remaining <= 0: raise TimeoutError('FULL drain timed out')
                self.changed.wait(remaining)
        if self.quarantined or self.restore_session.quarantined:
            raise RuntimeError('runtime retains quarantined resources')

    def close(self, timeout=120):
        with self.changed:
            self._closing = True
            self.changed.notify_all()
        started = time.monotonic()
        self.drain(timeout)
        self.transport.close(max(0, timeout - (time.monotonic()-started)))
