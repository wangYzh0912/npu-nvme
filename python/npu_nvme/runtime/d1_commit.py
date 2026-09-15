"""Single-owner FULL catalog; physical ownership belongs to the native context."""
from __future__ import annotations
from collections import Counter, OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import threading
import uuid
import hashlib
from npu_nvme.types import Reservation, TransferReceipt, CommitReceipt
from .d1_schema import validate_record, integer, canonical


class D1CommitError(RuntimeError): pass
class NoSpace(D1CommitError): pass
class OutcomeUnknown(D1CommitError): pass


@dataclass(frozen=True)
class CatalogSnapshot:
    metadata_generation: int
    active_generation: int
    previous_generation: int | None
    records: tuple


class D1CommitCoordinator:
    """One pending publication, two retained generations, three physical slots.

    Metadata revisions also advance when retiring stale fallback references.
    They are intentionally independent of checkpoint generations.
    """
    def __init__(self, *, metadata_io, state, rank_id=0, world_size=1,
                 keep_last_n=2, slot_count=3, writer_epoch=None):
        if rank_id != 0 or world_size != 1:
            raise D1CommitError('D1 requires rank 0/world_size 1')
        if slot_count != 3 or keep_last_n != 2:
            raise D1CommitError('D1 requires three slots and two retained generations')
        self.io, self.state = metadata_io, state
        self.writer_epoch = str(writer_epoch or uuid.uuid4())
        self._lock = threading.RLock()
        self._terminal = OrderedDict()
        self._pins = Counter()
        self._reservations = {}
        self._serial = 0
        self.poisoned = False
        self._check_catalog()
        self._generation = max((r['generation'] for r in self._records().values()), default=0)

    def _records(self):
        return self.state.meta_dict.get('checkpoints', {})

    def _check_catalog(self):
        meta = self.state.meta_dict
        if meta.get('strict_contract') != 'D1' or meta.get('catalog_revision') != self.state.metadata_generation:
            raise D1CommitError('strict D1 catalog required; initialize an empty D1 store explicitly')
        if self.state.layout.full_slot_count != 3:
            raise D1CommitError('D1 requires three physical FULL slots')
        records = self._records()
        if not isinstance(records, dict) or len(records) > 2:
            raise D1CommitError('invalid retained catalog')
        generations, slots = set(), set()
        for key, record in records.items():
            validate_record(record, self.state.layout)
            if key != 'generation_' + str(record['generation']):
                raise D1CommitError('catalog key differs from record generation')
            if record['generation'] in generations or record['slot'] in slots:
                raise D1CommitError('duplicate generation or physical slot')
            generations.add(record['generation']); slots.add(record['slot'])

    def _persist(self, records):
        candidate = deepcopy(self.state)
        revision = self.state.metadata_generation + 1
        candidate.meta_dict = {'strict_contract': 'D1', 'catalog_revision': revision,
                               'checkpoints': deepcopy(records)}
        try:
            self.io.persist(candidate, revision)
        except BaseException:
            # An error at the final barrier cannot establish non-commit.
            self.poisoned = True
            raise
        self.state.meta_dict = candidate.meta_dict
        self.state.metadata_generation = candidate.metadata_generation
        self.state.active_meta_slot = candidate.active_meta_slot
        self.state.layout = candidate.layout

    def reserve(self, *, step, request_id=None):
        integer(step, 'step')
        with self._lock:
            if self.poisoned: raise D1CommitError('catalog outcome unknown; reopen required')
            if self._reservations: raise NoSpace('one pending FULL already owns the spare slot')
            if request_id is not None:
                raise D1CommitError('request identity is assigned by admission')
            records = self._records()
            # A pinned oldest generation cannot be evicted by this commit.
            ordered = sorted(records.values(), key=lambda r: r['generation'])
            if len(ordered) == 2 and self._pins[ordered[0]['generation']]:
                raise NoSpace('oldest retained generation is pinned')
            used = {r['slot'] for r in records.values()}
            free = sorted(set(range(3)) - used)
            if not free: raise NoSpace('all slots are protected')
            # Overwrite the stale fallback with an identical current catalog
            # and commit that revision BEFORE allowing any payload reuse.
            self._persist(records)
            self._generation += 1
            self._serial += 1
            rid = f'{self.writer_epoch}:{self._serial}'
            reservation = Reservation(rid, self._generation, self.writer_epoch, 0, step, free[0])
            self._reservations[rid] = reservation
            return reservation

    def _next_records(self, record):
        records = deepcopy(self._records())
        records['generation_' + str(record['generation'])] = deepcopy(record)
        ordered = sorted(records, key=lambda k: records[k]['generation'])
        for key in ordered[:-2]:
            if self._pins[records[key]['generation']]: raise NoSpace('generation is pinned')
            del records[key]
        return records

    def preflight(self, reservation, record):
        with self._lock:
            if self._reservations.get(reservation.request_id) != reservation:
                raise D1CommitError('reservation is not owned')
            validate_record(record, self.state.layout)
            for name, value in (('request_id', reservation.request_id),
                                ('generation', reservation.generation), ('writer_epoch', reservation.writer_epoch),
                                ('state_step', reservation.step), ('slot', reservation.slot)):
                if record.get(name) != value: raise D1CommitError('manifest differs from reservation: ' + name)
            records = self._next_records(record)
            from npu_nvme.storage.format import pack_metadata
            revision = self.state.metadata_generation + 1
            pack_metadata({'strict_contract':'D1', 'catalog_revision':revision, 'checkpoints':records}, revision)
            return records

    def publish(self, reservation, transfer, record):
        with self._lock:
            old = self._terminal.get(reservation.request_id)
            if old is not None:
                receipt, fingerprint = old
                if fingerprint == (reservation, transfer, hashlib.sha256(canonical(record)).digest()):
                    return receipt
                raise D1CommitError('conflicting duplicate publication')
            if self.poisoned: raise OutcomeUnknown(reservation.request_id)
            if reservation.request_id not in self._reservations:
                raise OutcomeUnknown(reservation.request_id)
            records = self.preflight(reservation, record)
            total, chunks = validate_record(record, self.state.layout)
            if (transfer.request_id != reservation.request_id or transfer.generation != reservation.generation
                    or transfer.durable is not True or transfer.digest != record['data_sha256']
                    or transfer.bytes_written != total or transfer.chunks != chunks):
                raise D1CommitError('transfer receipt does not prove this manifest durable')
            self._persist(records)
            receipt = CommitReceipt(reservation.request_id, reservation.generation,
                                    self.state.metadata_generation, reservation.slot,
                                    tuple(sorted(r['generation'] for r in records.values())))
            self._terminal[reservation.request_id] = (receipt, (reservation, transfer, hashlib.sha256(canonical(record)).digest()))
            self._reservations.pop(reservation.request_id)
            while len(self._terminal) > 1024: self._terminal.popitem(last=False)
            return receipt

    def abort(self, reservation, error=None):
        with self._lock:
            if self.poisoned: return  # uncertain publication remains owned
            if self._reservations.get(reservation.request_id) != reservation:
                raise D1CommitError('reservation is not owned')
            self._reservations.pop(reservation.request_id)
            self._terminal[reservation.request_id] = (error or D1CommitError('cancelled before publication'), None)
            while len(self._terminal) > 1024: self._terminal.popitem(last=False)

    @contextmanager
    def selected(self, step=None):
        with self._lock:
            candidates = [r for r in self._records().values() if step is None or r['state_step'] == step]
            if not candidates: raise FileNotFoundError(f'no committed checkpoint for step {step}')
            record = max(candidates, key=lambda r: r['generation'])
            # Do not acquire a new pin on an eviction candidate after reserve.
            if self._reservations and len(self._records()) == 2:
                oldest = min(r['generation'] for r in self._records().values())
                if record['generation'] == oldest: raise NoSpace('generation is retiring')
            generation = record['generation']
            self._pins[generation] += 1
            frozen = deepcopy(record)
        try:
            yield frozen
        finally:
            with self._lock:
                self._pins[generation] -= 1
                if not self._pins[generation]: del self._pins[generation]

    def resolve(self, request_id):
        with self._lock:
            if request_id in self._terminal: return self._terminal[request_id][0]
            if request_id in self._reservations and not self.poisoned: return 'PENDING'
            raise OutcomeUnknown(request_id)

    def quarantine_pin(self, generation):
        with self._lock:
            if not self._pins[generation]: raise D1CommitError('cannot quarantine an unowned pin')
            self._pins[generation] += 1
            self.poisoned = True

    def snapshot(self):
        with self._lock:
            records = tuple(deepcopy(r) for r in sorted(self._records().values(), key=lambda r:r['generation'], reverse=True))
            gens = [r['generation'] for r in records]
            return CatalogSnapshot(self.state.metadata_generation, gens[0] if gens else 0,
                                   gens[1] if len(gens)>1 else None, records)
