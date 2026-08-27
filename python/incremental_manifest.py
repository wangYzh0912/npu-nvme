"""Stable complete-training-state manifests for incremental checkpoints.

The manifest is deliberately independent from MindSpore.  Objects only need
``parameters_and_names()`` and each parameter needs the usual ``shape`` and
``dtype`` attributes.  This keeps the naming/blocking contract testable on
CPU while allowing the same manifest to be used by the NPU graph and the
recovery process.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np


MANIFEST_SCHEMA_VERSION = 1


def _component_order(components: Mapping[str, Any]):
    preferred = [name for name in ("model", "optimizer") if name in components]
    preferred.extend(sorted(name for name in components
                           if name not in {"model", "optimizer"}))
    return preferred


def _parameter_dtype(parameter) -> np.dtype:
    dtype = getattr(parameter, "dtype", None)
    if dtype is None:
        raise TypeError("parameter has no dtype")
    try:
        return np.dtype(dtype)
    except TypeError:
        # MindSpore dtype objects expose a stable string representation such
        # as Float32/float32; callers may also provide a NumPy-compatible name.
        aliases = {"Float16": np.float16, "Float32": np.float32,
                   "Float64": np.float64, "Int32": np.int32,
                   "Int64": np.int64, "UInt8": np.uint8,
                   "Int8": np.int8, "Bool": np.bool_}
        key = str(dtype).split(".")[-1]
        if key not in aliases:
            raise TypeError(f"unsupported parameter dtype: {dtype}")
        return np.dtype(aliases[key])


def _local_shape(parameter) -> Tuple[int, ...]:
    shape = tuple(int(value) for value in getattr(parameter, "shape", ()))
    sliced = getattr(parameter, "sliced_shape", None)
    if sliced:
        shape = tuple(int(value) for value in sliced)
    data = getattr(parameter, "data", None)
    data_shape = getattr(data, "shape", None)
    if data_shape is not None and np.prod(data_shape) < np.prod(shape):
        shape = tuple(int(value) for value in data_shape)
    if any(value < 0 for value in shape):
        raise ValueError("parameter shape contains a negative dimension")
    return shape


@dataclass(frozen=True)
class ManifestBlock:
    state_index: int
    block_index: int
    element_offset: int
    element_count: int

    def as_dict(self) -> dict:
        return {"state_index": self.state_index,
                "block_index": self.block_index,
                "element_offset": self.element_offset,
                "element_count": self.element_count}


@dataclass(frozen=True)
class ManifestField:
    state_index: int
    namespace: str
    source_name: str
    canonical_name: str
    state_id: str
    dtype: str
    shape: Tuple[int, ...]
    element_count: int
    byte_count: int
    small: bool
    blocks: Tuple[ManifestBlock, ...]

    def as_dict(self) -> dict:
        return {
            "state_index": self.state_index,
            "namespace": self.namespace,
            "source_name": self.source_name,
            "canonical_name": self.canonical_name,
            "state_id": self.state_id,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "element_count": self.element_count,
            "byte_count": self.byte_count,
            "small": self.small,
            "blocks": [block.as_dict() for block in self.blocks],
        }


class TrainingStateManifest:
    """Immutable, deterministic manifest for model and optimizer state."""

    def __init__(self, fields, block_size, small_threshold):
        self.schema_version = MANIFEST_SCHEMA_VERSION
        self.block_size = int(block_size)
        self.small_threshold = int(small_threshold)
        self.fields = tuple(fields)
        payload = {
            "schema_version": self.schema_version,
            "block_size": self.block_size,
            "small_threshold": self.small_threshold,
            "fields": [field.as_dict() for field in self.fields],
        }
        self._payload = payload
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self._by_name = {field.canonical_name: field for field in self.fields}

    def as_dict(self) -> dict:
        return {**self._payload, "digest": self.digest}

    def field(self, canonical_name: str) -> ManifestField:
        try:
            return self._by_name[canonical_name]
        except KeyError as error:
            raise KeyError(f"unknown training-state field: {canonical_name}") from error

    def block(self, state_index: int, block_index: int) -> ManifestBlock:
        field = self.fields[int(state_index)]
        for block in field.blocks:
            if block.block_index == int(block_index):
                return block
        raise KeyError(f"unknown block {state_index}/{block_index}")


def build_training_state_manifest(components: Mapping[str, Any],
                                  block_size: int = 524288,
                                  small_threshold: int = 10000):
    """Build a stable manifest and de-duplicate aliased Parameters.

    Names are retained in framework order within each component only long
    enough to collect them; final state indices are assigned by canonical
    name, making construction order irrelevant.
    """
    if not isinstance(components, Mapping) or not components:
        raise ValueError("components must be a non-empty mapping")
    if block_size <= 0 or small_threshold < 0:
        raise ValueError("invalid block or small-field dimensions")

    collected = []
    seen_objects = set()
    seen_names = set()
    for namespace in _component_order(components):
        component = components[namespace]
        if not hasattr(component, "parameters_and_names"):
            raise TypeError(f"component {namespace!r} has no parameters_and_names()")
        for source_name, parameter in component.parameters_and_names():
            if id(parameter) in seen_objects:
                continue
            seen_objects.add(id(parameter))
            canonical_name = f"{namespace}/{source_name}"
            if canonical_name in seen_names:
                raise ValueError(f"duplicate training-state field: {canonical_name}")
            seen_names.add(canonical_name)
            dtype = _parameter_dtype(parameter)
            shape = _local_shape(parameter)
            elements = int(np.prod(shape, dtype=np.int64))
            if elements <= 0:
                continue
            collected.append((namespace, str(source_name), canonical_name, dtype,
                              shape, elements))

    collected.sort(key=lambda item: item[2])
    fields = []
    for state_index, (namespace, source_name, canonical_name, dtype, shape,
                      elements) in enumerate(collected):
        dtype_name = dtype.newbyteorder("<").name
        state_id = hashlib.sha256(
            json.dumps({"namespace": namespace, "name": canonical_name,
                        "dtype": dtype_name, "shape": list(shape)},
                       sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        small = elements < small_threshold
        offsets = (0,) if small else range(0, elements, block_size)
        blocks = tuple(
            ManifestBlock(state_index, block_index, offset,
                          elements if small else min(block_size, elements - offset))
            for block_index, offset in enumerate(offsets)
        )
        fields.append(ManifestField(
            state_index=state_index, namespace=namespace,
            source_name=source_name, canonical_name=canonical_name,
            state_id=state_id, dtype=dtype_name, shape=shape,
            element_count=elements, byte_count=elements * dtype.itemsize,
            small=small, blocks=blocks))
    if not fields:
        raise ValueError("components contain no persistable parameters")
    return TrainingStateManifest(fields, block_size, small_threshold)


__all__ = ["MANIFEST_SCHEMA_VERSION", "ManifestBlock", "ManifestField",
           "TrainingStateManifest", "build_training_state_manifest"]
