"""Versioned, non-pickle encoding for checkpoint control state.

The payload codec is intentionally independent from MindSpore and SPDK so it
can be unit-tested without hardware.  It preserves tuples, NumPy arrays,
NumPy scalars, and bytes used by Python/NumPy RNG states.
"""

import base64
import hashlib
import json
from typing import Any, Dict, Mapping, Tuple

import numpy as np


TRAINING_STATE_SCHEMA_VERSION = 1


def _to_jsonable(value: Any):
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "items": [_to_jsonable(v) for v in value]}
    if isinstance(value, list):
        return {"__kind__": "list", "items": [_to_jsonable(v) for v in value]}
    if isinstance(value, dict):
        return {
            "__kind__": "dict",
            "items": [[str(key), _to_jsonable(item)]
                      for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))],
        }
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "__kind__": "ndarray",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return {
            "__kind__": "numpy_scalar",
            "dtype": value.dtype.str,
            "data": base64.b64encode(value.tobytes()).decode("ascii"),
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "__kind__": "bytes",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"unsupported control-state value: {type(value).__name__}")


def _from_jsonable(value: Any):
    if not isinstance(value, dict) or "__kind__" not in value:
        return value
    kind = value["__kind__"]
    if kind == "tuple":
        return tuple(_from_jsonable(v) for v in value["items"])
    if kind == "list":
        return [_from_jsonable(v) for v in value["items"]]
    if kind == "dict":
        return {key: _from_jsonable(item) for key, item in value["items"]}
    if kind == "ndarray":
        raw = base64.b64decode(value["data"], validate=True)
        dtype = np.dtype(value["dtype"])
        shape = tuple(int(dim) for dim in value["shape"])
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if len(raw) != expected:
            raise ValueError("encoded ndarray byte length does not match shape/dtype")
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if kind == "numpy_scalar":
        raw = base64.b64decode(value["data"], validate=True)
        dtype = np.dtype(value["dtype"])
        if len(raw) != dtype.itemsize:
            raise ValueError("encoded NumPy scalar has invalid byte length")
        return np.frombuffer(raw, dtype=dtype)[0]
    if kind == "bytes":
        return base64.b64decode(value["data"], validate=True)
    raise ValueError(f"unsupported control-state kind: {kind}")


def encode_control_value(value: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return an owned uint8 payload and its self-describing metadata."""
    document = {
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "value": _to_jsonable(value),
    }
    raw = json.dumps(document, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    payload = np.frombuffer(raw, dtype=np.uint8).copy()
    return payload, {
        "codec": "json-tagged-v1",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def decode_control_value(payload: np.ndarray, metadata: Mapping[str, Any]):
    raw = np.ascontiguousarray(payload, dtype=np.uint8).tobytes()
    expected_hash = metadata.get("sha256")
    if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("control-state checksum mismatch")
    if metadata.get("codec") != "json-tagged-v1":
        raise ValueError(f"unsupported control-state codec: {metadata.get('codec')}")
    document = json.loads(raw.decode("utf-8"))
    if document.get("schema_version") != TRAINING_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported control-state schema version")
    return _from_jsonable(document["value"])


def validate_state_names(components: Mapping[str, Any],
                         control_state: Mapping[str, Any]) -> None:
    if not isinstance(components, Mapping) or not components:
        raise ValueError("components must be a non-empty mapping")
    if not isinstance(control_state, Mapping):
        raise TypeError("control_state must be a mapping")
    for namespace in components:
        if not isinstance(namespace, str) or not namespace or "/" in namespace:
            raise ValueError(f"invalid component namespace: {namespace!r}")
    for name in control_state:
        if not isinstance(name, str) or not name or "/" in name:
            raise ValueError(f"invalid control-state name: {name!r}")
