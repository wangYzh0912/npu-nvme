"""Versioned, non-pickle encoding for checkpoint control state.

The payload codec is intentionally independent from MindSpore and SPDK so it
can be unit-tested without hardware.  It preserves tuples, NumPy arrays,
NumPy scalars, and bytes used by Python/NumPy RNG states.
"""

import base64
import hashlib
import json
import random
from typing import Any, Dict, Mapping, Tuple

import numpy as np


TRAINING_STATE_SCHEMA_VERSION = 1
TRAINING_CONTROL_FIELDS = frozenset({
    "global_step", "loss_scale", "python_rng", "numpy_rng",
    "mindspore_seed", "mindspore_rng", "data_cursor",
})


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


def _scheduler_state(scheduler):
    if scheduler is None:
        return None
    if hasattr(scheduler, "state_dict"):
        return scheduler.state_dict()
    if hasattr(scheduler, "get_state"):
        return scheduler.get_state()
    raise TypeError("mutable scheduler must expose state_dict() or get_state()")


def capture_training_controls(ms, optimizer, data_cursor, loss_scale,
                              seed: int, scheduler=None) -> Dict[str, Any]:
    """Capture the complete restart control state used by FULL and Delta.

    Tensor state remains in the model/optimizer component manifests.  The
    duplicated global step here is intentional: recovery validates that the
    tensor and control generations describe the same logical training step.
    """
    controls = {
        "global_step": np.asarray(optimizer.global_step.asnumpy()).copy(),
        "loss_scale": np.asarray(loss_scale).copy()
        if isinstance(loss_scale, np.ndarray) else np.float32(loss_scale),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "mindspore_seed": int(seed),
        "mindspore_rng": np.asarray(ms.get_rng_state().asnumpy()).copy(),
        "data_cursor": dict(data_cursor),
    }
    scheduler_state = _scheduler_state(scheduler)
    if scheduler_state is not None:
        controls["scheduler"] = scheduler_state
    return controls


def restore_training_controls(ms, optimizer, controls: Mapping[str, Any],
                              scheduler=None) -> Dict[str, Any]:
    """Restore RNG/cursor controls and validate the required field set."""
    expected = set(TRAINING_CONTROL_FIELDS)
    if scheduler is not None:
        expected.add("scheduler")
    if set(controls) != expected:
        raise ValueError(
            "training control fields differ: expected=%s actual=%s" %
            (sorted(expected), sorted(controls)))

    saved_step = np.asarray(controls["global_step"])
    current_step = np.asarray(optimizer.global_step.asnumpy())
    if saved_step.dtype != current_step.dtype or saved_step.size != current_step.size:
        raise ValueError("global_step dtype/size differs from optimizer state")
    saved_step = saved_step.reshape(current_step.shape)
    if not np.array_equal(current_step, saved_step):
        if not hasattr(optimizer.global_step, "set_data"):
            raise ValueError("optimizer global_step cannot be restored")
        optimizer.global_step.set_data(ms.Tensor(saved_step))

    random.setstate(controls["python_rng"])
    np.random.set_state(controls["numpy_rng"])
    if hasattr(ms, "common") and hasattr(ms.common, "set_seed"):
        ms.common.set_seed(int(controls["mindspore_seed"]))
    elif hasattr(ms, "set_seed"):
        ms.set_seed(int(controls["mindspore_seed"]))
    ms.set_rng_state(ms.Tensor(np.asarray(controls["mindspore_rng"])))

    if scheduler is not None:
        if hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(controls["scheduler"])
        elif hasattr(scheduler, "set_state"):
            scheduler.set_state(controls["scheduler"])
        else:
            raise TypeError(
                "mutable scheduler must expose load_state_dict() or set_state()")
    return {
        "data_cursor": controls["data_cursor"],
        "loss_scale": controls["loss_scale"],
        "global_step": np.array(saved_step, copy=True),
    }
