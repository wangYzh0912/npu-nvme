"""D1 single-owner/rank FULL commit coordinator over the existing V2 store."""
from __future__ import annotations
from dataclasses import dataclass
import fcntl
import hashlib
import os
from pathlib import Path
import threading
from typing import Dict, Tuple
from npu_nvme.types import Reservation, TransferReceipt, CommitReceipt
from .owner import OwnerLock

class D1CommitError(RuntimeError): pass
class NoSpace(D1CommitError): pass
class OutcomeUnknown(D1CommitError): pass

@dataclass(frozen=True)
class CatalogSnapshot:
    metadata_generation: int
    active_generation: int
    previous_generation: int | None
    records: Tuple[dict, ...]

class D1CommitCoordinator:
    """Formal frozen FULL publication; Delta/live/multirank are rejected."""
    def __init__(self, *, metadata_io, state, rank_id=0, world_size=1,
                 keep_last_n=2, slot_count=3, writer_epoch=1,
                 pci_addr='0000:83:00.0', namespace=1, owner_lock=None):
        if rank_id != 0 or world_size != 1: raise D1CommitError('D1 accepts single rank/owner only')
        if slot_count != 3 or keep_last_n != 2: raise D1CommitError('D1 requires three FULL slots and two retained generations')
        self.io, self.state = metadata_io, state
        self.rank_id, self.world_size = rank_id, world_size
        self.keep_last_n, self.slot_count, self.writer_epoch = keep_last_n, slot_count, writer_epoch
        self.owner = owner_lock or OwnerLock(pci_addr, namespace)
        self._lock = threading.RLock(); self._request = 0; self._terminal: Dict[str, object] = {}; self._pins=set(); self._reservations={}

    def reserve(self, *, step, request_id=None):
        with self._lock:
            if request_id is None:
                self._request += 1; request_id = f'd1-e{self.writer_epoch}-r{self._request}'
            if request_id in self._terminal or request_id in self._reservations: raise D1CommitError('request_id already used')
            self.owner.acquire()
            try:
                generation = max((int(v.get('generation',0)) for v in self.state.meta_dict.get('checkpoints',{}).values()), default=0)+1
                slot = generation % self.slot_count
                active_slots = {int(v.get('slot',-1)) for v in self.state.meta_dict.get('checkpoints',{}).values() if int(v.get('generation',0)) in sorted((int(x.get('generation',0)) for x in self.state.meta_dict.get('checkpoints',{}).values()), reverse=True)[:self.keep_last_n]}
                if slot in active_slots or any(int(v.get('slot',-1)) == slot and int(v.get('generation',0)) in self._pins for v in self.state.meta_dict.get('checkpoints',{}).values()): raise NoSpace('all retained or pinned slots are protected')
                reservation=Reservation(request_id,generation,self.writer_epoch,0,int(step),slot); self._reservations[request_id]=reservation; return reservation
            except BaseException:
                self.owner.release(); raise

    def publish(self, reservation: Reservation, transfer: TransferReceipt):
        with self._lock:
            if self._reservations.get(reservation.request_id) != reservation: raise D1CommitError('reservation is not owned')
            if reservation.writer_epoch != self.writer_epoch or transfer.request_id != reservation.request_id or transfer.generation != reservation.generation or not transfer.durable: raise D1CommitError('invalid owner, receipt, or durability token')
            try:
                records=dict(self.state.meta_dict.get('checkpoints',{})); key=f'step_{reservation.step}'
                records[key]={'type':'FULL','strict_contract':'D1','generation':reservation.generation,'slot':reservation.slot,'request_id':reservation.request_id,'digest':transfer.digest,'bytes':transfer.bytes_written,'chunks':transfer.chunks}
                generations=sorted((int(v.get('generation',0)),k) for k,v in records.items())
                for gen,k in generations[:-self.keep_last_n]:
                    if gen in self._pins: raise NoSpace('pinned generation cannot be evicted')
                    records.pop(k,None)
                self.state.meta_dict['checkpoints']=records
                self.io.persist(self.state, reservation.generation)
                retained=tuple(sorted(int(v.get('generation',0)) for v in records.values()))
                receipt=CommitReceipt(reservation.request_id,reservation.generation,self.state.metadata_generation,reservation.slot,retained)
                self._terminal[reservation.request_id]=receipt; return receipt
            finally:
                self._reservations.pop(reservation.request_id,None); self.owner.release()

    def abort(self, reservation):
        with self._lock:
            if self._reservations.pop(reservation.request_id,None) is not None: self.owner.release()
    def pin(self,generation):
        with self._lock: self._pins.add(int(generation))
    def unpin(self,generation):
        with self._lock: self._pins.discard(int(generation))
    def resolve(self,request_id):
        with self._lock:
            if request_id not in self._terminal: raise OutcomeUnknown(request_id)
            return self._terminal[request_id]
    def snapshot(self):
        with self._lock:
            records=tuple(dict(v) for v in self.state.meta_dict.get('checkpoints',{}).values()); gens=sorted((int(v.get('generation',0)) for v in records),reverse=True)
            return CatalogSnapshot(self.state.metadata_generation,gens[0] if gens else 0,gens[1] if len(gens)>1 else None,records)
