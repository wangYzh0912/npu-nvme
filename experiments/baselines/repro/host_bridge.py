"""Owned shared-memory slots for cross-environment Host workers."""

from __future__ import annotations

import json
import os
import uuid
from multiprocessing import shared_memory

import numpy as np

from .state_bridge import Snapshot


class SharedSnapshot:
    def __init__(self, name, size, create=False):
        self.shm = shared_memory.SharedMemory(name=name, create=create, size=size)
        self.name = self.shm.name
        self.size = int(size)
        self.closed = False

    @classmethod
    def from_snapshot(cls, snapshot, prefix="npu_nvme_repro"):
        size = int(snapshot.total_bytes)
        name = f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:12]}"
        owner = cls(name, size, create=True)
        offset = 0
        fields = []
        for field_name in sorted(snapshot.arrays):
            array = snapshot.arrays[field_name]
            raw = array.tobytes()
            owner.shm.buf[offset:offset + len(raw)] = raw
            fields.append({"name": field_name, "dtype": array.dtype.str,
                           "shape": list(array.shape), "offset": offset,
                           "nbytes": len(raw)})
            offset += len(raw)
        control_raw = snapshot.controls_payload.tobytes()
        owner.shm.buf[offset:offset + len(control_raw)] = control_raw
        descriptor = {
            "name": owner.name,
            "size": size,
            "fields": fields,
            "controls": {"offset": offset, "nbytes": len(control_raw),
                         "metadata": snapshot.controls_metadata},
            "schema": snapshot.schema,
        }
        return owner, descriptor

    def close(self, unlink=False):
        if not self.closed:
            self.shm.close()
            self.closed = True
        if unlink:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


def snapshot_from_descriptor(descriptor):
    owner = SharedSnapshot(descriptor["name"], descriptor["size"])
    arrays = {}
    try:
        for field in descriptor["fields"]:
            begin = int(field["offset"])
            end = begin + int(field["nbytes"])
            arrays[field["name"]] = np.frombuffer(
                owner.shm.buf[begin:end], dtype=np.dtype(field["dtype"])).reshape(
                    tuple(field["shape"])).copy()
        control = descriptor["controls"]
        begin = int(control["offset"])
        end = begin + int(control["nbytes"])
        payload = np.frombuffer(owner.shm.buf[begin:end], dtype=np.uint8).copy()
        return Snapshot(arrays, payload, control["metadata"], descriptor["schema"])
    finally:
        owner.close()

