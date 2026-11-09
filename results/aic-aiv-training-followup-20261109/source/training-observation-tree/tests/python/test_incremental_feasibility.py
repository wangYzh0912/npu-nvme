import json
from pathlib import Path

import numpy as np

from npu_nvme.experiments.config import load
from npu_nvme.experiments.topk import (BlockId, apply_delta, build_blocks,
    encode_delta, relative_l2, score_blocks, select)


ROOT = Path(__file__).resolve().parents[2]


def test_phase_one_config_is_strict_and_resolved():
    config = load(ROOT / "config/incremental_feasibility.json", ROOT)
    assert config["block_elements"] == 65536
    assert config["models"]["main"] == "/models/Qwen3-8B"
    assert len(config["config_sha256"]) == 64


def test_blocks_do_not_cross_parameters_and_tail_is_exact():
    state = {"large": np.zeros(65536 * 2 + 7, np.float32), "small": np.zeros(9, np.float32)}
    blocks = build_blocks(state)
    large = [block for block in blocks if block.identity.parameter == "large"]
    small = [block for block in blocks if block.identity.parameter == "small"]
    assert [block.element_count for block in large] == [65536, 65536, 7]
    assert not any(block.small for block in large)
    assert len(small) == 1 and small[0].small and small[0].element_count == 9


def test_topk_uses_ceil_and_deterministic_identity_ties():
    scores = {BlockId("b", 0): 1.0, BlockId("a", 1): 1.0,
              BlockId("a", 0): 1.0, BlockId("c", 0): 0.0}
    assert select(scores, .20) == (BlockId("a", 0),)
    assert select(scores, .50) == (BlockId("a", 0), BlockId("a", 1))


def test_actual_delta_roundtrip_and_metrics():
    reference = {"decoder.layers.0.weight": np.zeros(65537, np.float32),
                 "norm": np.zeros(3, np.float32)}
    current = {name: value.copy() for name, value in reference.items()}
    current["decoder.layers.0.weight"][:65536] = 2
    current["decoder.layers.0.weight"][-1] = 1
    current["norm"][:] = 3
    blocks = build_blocks(reference)
    scores = score_blocks(current, reference, blocks)
    chosen = select(scores, .5)
    records, payload = encode_delta(current, blocks, chosen)
    shadow = {name: value.copy() for name, value in reference.items()}
    apply_delta(shadow, records, payload)
    assert np.array_equal(shadow["norm"], current["norm"])
    assert np.array_equal(shadow["decoder.layers.0.weight"][:65536], current["decoder.layers.0.weight"][:65536])
    metric = relative_l2(shadow, current)
    assert metric["global_relative_l2"] > 0
    assert metric["worst_layer"] == "decoder.layers.0"


def test_corrupt_payload_is_rejected():
    state = {"small": np.arange(4, dtype=np.float32)}
    blocks = build_blocks(state)
    records, payload = encode_delta(state, blocks, ())
    damaged = bytearray(payload); damaged[0] ^= 1
    try:
        apply_delta({"small": np.zeros(4, np.float32)}, records, damaged)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("corrupt increment was accepted")
