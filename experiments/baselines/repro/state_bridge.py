"""Canonical FULL training-state schema, fixture, and durable raw format."""

from __future__ import annotations

import hashlib
import json
import os
import random
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np

from python.training_state import (capture_training_controls,
                                   decode_control_value,
                                   encode_control_value,
                                   restore_training_controls)


FORMAT = "npu-nvme-repro-raw-v1"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def fsync_directory(path):
    descriptor = os.open(str(Path(path)), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def iter_unique_parameters(components):
    seen_objects = set()
    seen_storage = {}
    for category, component in components.items():
        for name, parameter in component.parameters_and_names():
            if id(parameter) in seen_objects:
                continue
            seen_objects.add(id(parameter))
            canonical = f"{category}/{name}"
            pointer = None
            try:
                data = parameter.data if hasattr(parameter, "data") else parameter
                pointer = int(data._data_ptr()) if hasattr(data, "_data_ptr") else None
            except Exception:
                pointer = None
            alias = seen_storage.get(pointer) if pointer else None
            if pointer and alias is None:
                seen_storage[pointer] = canonical
            yield canonical, category, parameter, alias


@dataclass
class Snapshot:
    arrays: Dict[str, np.ndarray]
    controls_payload: np.ndarray
    controls_metadata: Dict
    schema: Dict

    @property
    def total_bytes(self):
        return sum(array.nbytes for array in self.arrays.values()) + \
            self.controls_payload.nbytes

    def digest(self):
        digest = hashlib.sha256()
        for name in sorted(self.arrays):
            digest.update(name.encode("utf-8"))
            digest.update(np.ascontiguousarray(self.arrays[name]).tobytes())
        digest.update(self.controls_payload.tobytes())
        return digest.hexdigest()


def capture_snapshot(ms, components, optimizer, step, seed):
    arrays = {}
    fields = []
    for name, category, parameter, alias in iter_unique_parameters(components):
        array = np.asarray(parameter.value().asnumpy())
        if not array.flags.c_contiguous:
            raise ValueError(f"non-contiguous state field is unsupported: {name}")
        array = np.ascontiguousarray(array)
        arrays[name] = array
        fields.append({
            "name": name,
            "category": category,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
            "device_kind": "npu",
            "address": None,
            "host_offset": None,
            "alias_group": alias or name,
        })
    controls = capture_training_controls(
        ms, optimizer, {"epoch": 0, "sample": int(step), "next_step": int(step) + 1},
        np.float32(1.0), int(seed))
    payload, metadata = encode_control_value(controls)
    fields.append({
        "name": "controls/state",
        "category": "control",
        "dtype": payload.dtype.str,
        "shape": list(payload.shape),
        "nbytes": int(payload.nbytes),
        "device_kind": "host",
        "address": None,
        "host_offset": None,
        "alias_group": "controls/state",
    })
    return Snapshot(arrays, payload, metadata, {
        "schema_version": 1,
        "format": FORMAT,
        "fields": sorted(fields, key=lambda item: item["name"]),
    })


def parameter_map(components):
    return {name: parameter for name, _category, parameter, _alias
            in iter_unique_parameters(components)}


def apply_snapshot(ms, snapshot, components, optimizer):
    destinations = parameter_map(components)
    if set(destinations) != set(snapshot.arrays):
        missing = sorted(set(destinations) - set(snapshot.arrays))
        extra = sorted(set(snapshot.arrays) - set(destinations))
        raise ValueError(f"state field mismatch missing={missing} extra={extra}")
    for name, destination in destinations.items():
        current = np.asarray(destination.value().asnumpy())
        saved = snapshot.arrays[name]
        if current.dtype != saved.dtype:
            raise ValueError(f"dtype mismatch for {name}: {current.dtype} != {saved.dtype}")
        # MindSpore may materialize a scalar learning-rate Parameter as [] in
        # one fresh graph and [1] in another.  Preserve its bytes while
        # allowing this representation-only difference for singleton fields.
        if current.shape != saved.shape:
            if current.size != saved.size or current.size != 1:
                raise ValueError(f"dtype/shape mismatch for {name}")
            saved = saved.reshape(current.shape)
        destination.set_data(ms.Tensor(saved))
    controls = decode_control_value(snapshot.controls_payload,
                                    snapshot.controls_metadata)
    return restore_training_controls(ms, optimizer, controls)


def save_raw_snapshot(snapshot, generation_dir, phase_callback=None,
                      extra_metadata=None, chunk_bytes=None):
    generation_dir = Path(generation_dir)
    generation_dir.mkdir(parents=True, exist_ok=True)
    data_tmp = generation_dir / "data.bin.tmp"
    metadata_tmp = generation_dir / "metadata.json.tmp"
    offsets = {}
    with data_tmp.open("wb", buffering=0) as stream:
        offset = 0
        for name in sorted(snapshot.arrays):
            raw = memoryview(np.ascontiguousarray(
                snapshot.arrays[name])).cast("B")
            length = len(raw)
            stride = int(chunk_bytes or length or 1)
            for begin in range(0, length, stride):
                stream.write(raw[begin:begin + stride])
            offsets[name] = [offset, length]
            offset += length
        control_raw = memoryview(np.ascontiguousarray(
            snapshot.controls_payload)).cast("B")
        control_length = len(control_raw)
        stream.write(control_raw)
        offsets["controls/state"] = [offset, control_length]
        if phase_callback:
            phase_callback("flush_begin")
        stream.flush()
        os.fsync(stream.fileno())
    if phase_callback:
        phase_callback("metadata_begin")
    metadata = {
        "format": FORMAT,
        "schema": snapshot.schema,
        "offsets": offsets,
        "controls_metadata": snapshot.controls_metadata,
        "sha256": snapshot.digest(),
        "write_chunk_bytes": int(chunk_bytes) if chunk_bytes else None,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with metadata_tmp.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(data_tmp, generation_dir / "data.bin")
    os.replace(metadata_tmp, generation_dir / "metadata.json")
    fsync_directory(generation_dir)
    return metadata


def load_raw_snapshot(generation_dir, expected_generation=None):
    generation_dir = Path(generation_dir)
    metadata = json.loads((generation_dir / "metadata.json").read_text())
    if metadata.get("format") != FORMAT:
        raise ValueError("unsupported snapshot format")
    if expected_generation is not None and int(metadata.get(
            "generation", -1)) != int(expected_generation):
        raise ValueError("checkpoint generation mismatch")
    raw = (generation_dir / "data.bin").read_bytes()
    arrays = {}
    by_name = {field["name"]: field for field in metadata["schema"]["fields"]}
    for name, (offset, length) in metadata["offsets"].items():
        payload = raw[int(offset):int(offset) + int(length)]
        if len(payload) != int(length):
            raise ValueError(f"truncated field: {name}")
        if name == "controls/state":
            controls_payload = np.frombuffer(payload, dtype=np.uint8).copy()
            continue
        field = by_name[name]
        arrays[name] = np.frombuffer(payload, dtype=np.dtype(field["dtype"])).reshape(
            tuple(field["shape"])).copy()
    snapshot = Snapshot(arrays, controls_payload,
                        metadata["controls_metadata"], metadata["schema"])
    if snapshot.digest() != metadata["sha256"]:
        raise ValueError("snapshot checksum mismatch")
    return snapshot


def create_batches(path, steps, seq_len, vocab_size=50257):
    path = Path(path)
    ids, masks = [], []
    for step in range(1, int(steps) + 1):
        start = (step * 104729) % vocab_size
        ids.append((np.arange(seq_len, dtype=np.int32) + start) % vocab_size)
        masks.append(np.ones(seq_len, dtype=np.int32))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, input_ids=np.asarray(ids), attention_mask=np.asarray(masks))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "steps": int(steps)}


def load_batch(ms, path, step):
    with np.load(path) as batches:
        index = int(step) - 1
        return (ms.Tensor(batches["input_ids"][index:index + 1]),
                ms.Tensor(batches["attention_mask"][index:index + 1]))
