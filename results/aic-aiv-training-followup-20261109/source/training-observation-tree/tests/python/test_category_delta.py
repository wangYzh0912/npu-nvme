import sys

import numpy as np
import pytest

sys.path.insert(0, "python")
from category_delta import (CategoryAwarePolicy, CategoryConfig, apply_frame,
                            unpack_frame)


def state():
    return {
        "model/w": np.arange(16, dtype=np.float32),
        "optimizer/m/w": np.linspace(-1, 1, 16, dtype=np.float32),
        "optimizer/v/w": np.linspace(0, 2, 16, dtype=np.float32),
        "optimizer/global_step": np.array([1], dtype=np.int32),
    }


def test_category_frame_roundtrip_and_accounting():
    base = state()
    current = {name: value + (1 if value.dtype.kind == "f" else 1)
               for name, value in base.items()}
    policy = CategoryAwarePolicy(base, 4, {
        "model": CategoryConfig(1.0, "raw"),
        "adam_m": CategoryConfig(1.0, "fp16"),
        "adam_v": CategoryConfig(1.0, "int8"),
        "other": CategoryConfig(1.0, "raw"),
    })
    pending = policy.observe(current, 2)
    frame, accounting = policy.pack(current)
    restored, parsed = apply_frame(base, frame)
    policy.ack(pending["generation"])
    assert len(frame) % 4096 == 0
    assert accounting["physical_bytes"] == len(frame)
    assert parsed["generation"] == 1
    np.testing.assert_array_equal(restored["model/w"], current["model/w"])
    np.testing.assert_array_equal(restored["optimizer/global_step"],
                                  current["optimizer/global_step"])
    np.testing.assert_allclose(restored["optimizer/m/w"],
                               current["optimizer/m/w"], rtol=1e-3)
    np.testing.assert_allclose(restored["optimizer/v/w"],
                               current["optimizer/v/w"], atol=0.02)


def test_crc_and_generation_guards():
    base = state()
    policy = CategoryAwarePolicy(base, 4, {})
    current = {name: value + 1 for name, value in base.items()}
    pending = policy.observe(current, 1)
    frame, _ = policy.pack(current)
    damaged = bytearray(frame)
    parsed = unpack_frame(frame)
    damaged[parsed["header_bytes"]] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        unpack_frame(bytes(damaged))
    with pytest.raises(ValueError, match="stale"):
        policy.ack(pending["generation"] + 1)
