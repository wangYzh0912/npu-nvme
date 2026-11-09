"""Deterministic logical block selection and Host-side fidelity oracle."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np


@dataclass(frozen=True, order=True)
class BlockId:
    parameter: str
    block_index: int


@dataclass(frozen=True)
class Block:
    identity: BlockId
    element_offset: int
    element_count: int
    dtype: str
    small: bool

    @property
    def byte_count(self):
        return self.element_count * np.dtype(self.dtype).itemsize


def build_blocks(parameters: Mapping[str, np.ndarray], *, block_elements=65536,
                 small_parameter_elements=65536):
    if block_elements <= 0 or small_parameter_elements != block_elements:
        raise ValueError("invalid phase-one block geometry")
    result = []
    for name in sorted(parameters):
        value = np.asarray(parameters[name])
        if value.size <= 0 or value.dtype.hasobject or value.dtype.fields:
            raise ValueError("unsupported parameter: " + name)
        dtype = value.dtype.newbyteorder("<").str
        small = value.size < small_parameter_elements
        offsets = (0,) if small else range(0, value.size, block_elements)
        for index, offset in enumerate(offsets):
            result.append(Block(BlockId(name, index), offset,
                value.size if small else min(block_elements, value.size - offset), dtype, small))
    return tuple(result)


def score_blocks(current, reference, blocks):
    scores = {}
    for block in blocks:
        if block.small:
            continue
        left = np.asarray(current[block.identity.parameter]).reshape(-1)
        right = np.asarray(reference[block.identity.parameter]).reshape(-1)
        if left.shape != right.shape:
            raise ValueError("current/reference geometry differs")
        start = block.element_offset
        diff = left[start:start + block.element_count].astype(np.float64) - right[start:start + block.element_count]
        score = float(np.dot(diff, diff))
        if not math.isfinite(score):
            raise ValueError("nonfinite block score")
        scores[block.identity] = score
    return scores


def select(scores, ratio):
    if type(ratio) not in (int, float) or not 0 < ratio <= 1 or not scores:
        raise ValueError("invalid Top-K request")
    if any(not isinstance(key, BlockId) or not math.isfinite(value) or value < 0
           for key, value in scores.items()):
        raise ValueError('invalid block score')
    count = int(math.ceil(float(ratio) * len(scores)))
    ordered = sorted(scores, key=lambda key: (-scores[key], key.parameter, key.block_index))
    return tuple(ordered[:count])


def encode_delta(current, blocks, selected):
    selected = set(selected)
    records = []
    payload = bytearray()
    for block in blocks:
        if not block.small and block.identity not in selected:
            continue
        value = np.asarray(current[block.identity.parameter]).reshape(-1)
        raw = np.ascontiguousarray(value[block.element_offset:block.element_offset + block.element_count]).tobytes()
        offset = len(payload)
        payload.extend(raw)
        records.append({
            "parameter": block.identity.parameter,
            "block_index": block.identity.block_index,
            "element_offset": block.element_offset,
            "element_count": block.element_count,
            "dtype": block.dtype,
            "small": block.small,
            "payload_offset": offset,
            "payload_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return records, bytes(payload)


def apply_delta(shadow, records, payload):
    payload = memoryview(payload)
    planned = []; identities = set(); occupied = []
    for row in records:
        identity = (row['parameter'], row['block_index'])
        if identity in identities:
            raise ValueError('duplicate increment block')
        identities.add(identity)
        for field in ('block_index', 'element_offset', 'payload_offset', 'element_count', 'payload_bytes'):
            if type(row[field]) is not int or row[field] < (1 if field in ('element_count', 'payload_bytes') else 0):
                raise ValueError('invalid increment geometry')
        start = row["payload_offset"]
        end = start + row["payload_bytes"]
        dtype = np.dtype(row['dtype'])
        if dtype.hasobject or dtype.fields or dtype.kind not in 'fiu' or row['element_count'] * dtype.itemsize != row['payload_bytes']:
            raise ValueError('invalid increment dtype/length')
        raw = payload[start:end]
        if len(raw) != row["payload_bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("increment payload checksum differs")
        value = np.frombuffer(raw, dtype=np.dtype(row["dtype"]), count=row["element_count"])
        target = np.asarray(shadow[row["parameter"]]).reshape(-1)
        offset = row["element_offset"]
        if target.dtype != dtype or offset + row['element_count'] > target.size:
            raise ValueError('increment target geometry differs')
        occupied.append((row['parameter'], offset, offset + row['element_count']))
        planned.append((target, offset, value))
    occupied.sort()
    if any(a[0] == b[0] and a[2] > b[1] for a, b in zip(occupied, occupied[1:])):
        raise ValueError('overlapping increment targets')
    for target, offset, value in planned:
        target[offset:offset + value.size] = value


def relative_l2(shadow, current):
    error = norm = 0.0
    layers = {}
    for name in sorted(current):
        left = np.asarray(shadow[name]).astype(np.float64, copy=False).reshape(-1)
        right = np.asarray(current[name]).astype(np.float64, copy=False).reshape(-1)
        if left.shape != right.shape:
            raise ValueError("shadow/current geometry differs")
        diff = left - right
        e = float(np.dot(diff, diff)); n = float(np.dot(right, right))
        error += e; norm += n
        layer = _layer_name(name)
        bucket = layers.setdefault(layer, [0.0, 0.0])
        bucket[0] += e; bucket[1] += n
    metric = lambda pair: math.sqrt(pair[0]) / max(math.sqrt(pair[1]), 1e-12)
    by_layer = {name: metric(value) for name, value in layers.items()}
    return {"global_relative_l2": metric((error, norm)), "layers": by_layer,
            "worst_layer": max(by_layer, key=by_layer.get), "worst_layer_relative_l2": max(by_layer.values())}


def _layer_name(name):
    parts = name.split(".")
    for index in range(len(parts) - 1):
        if parts[index] == "layers" and parts[index + 1].isdigit():
            return ".".join(parts[:index + 2])
    return parts[0]
