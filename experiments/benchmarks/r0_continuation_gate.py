#!/usr/bin/env python3
"""CPU R0 FULL+S2 replacement continuation gate.

This gate isolates checkpoint lineage from model execution: it carries model,
Adam, RNG, loss-scale, global-step, and data-cursor fields through 100 exact
replacement frames, then verifies the next ten deterministic training steps
in a fresh process.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "python")]
from s2_delta import FileS2Ring, S2DeltaOracle  # noqa: E402


def initial_state():
    return {
        "model/layer0.weight": np.linspace(-1, 1, 257, dtype=np.float16),
        "model/layer1.bias": np.zeros(19, dtype=np.float32),
        "optimizer/m/layer0.weight": np.zeros(257, dtype=np.float32),
        "optimizer/v/layer0.weight": np.zeros(257, dtype=np.float32),
        "optimizer/m/layer1.bias": np.zeros(19, dtype=np.float32),
        "optimizer/v/layer1.bias": np.zeros(19, dtype=np.float32),
        "global_step": np.array([0], dtype=np.int64),
        "loss_scale": np.array([1.0], dtype=np.float32),
        "rng": np.array([0x123456789ABCDEF0], dtype=np.uint64),
        "data_cursor": np.array([0], dtype=np.int64),
    }


def advance(state, step):
    out = {name: value.copy() for name, value in state.items()}
    weight = out["model/layer0.weight"]
    bias = out["model/layer1.bias"]
    delta = np.float16(((step % 11) - 5) * 0.0007)
    weight[step % weight.size] += delta
    bias[step % bias.size] += np.float32(step * 0.0003)
    out["optimizer/m/layer0.weight"] += np.float32(delta)
    out["optimizer/v/layer0.weight"] += np.float32(delta * delta)
    out["optimizer/m/layer1.bias"] += np.float32(step * 0.0003)
    out["optimizer/v/layer1.bias"] += np.float32((step * 0.0003) ** 2)
    out["global_step"][0] = step
    out["rng"][0] = (int(out["rng"][0]) * 6364136223846793005 + step) & ((1 << 64) - 1)
    out["data_cursor"][0] += 17 + (step % 5)
    loss = float(np.mean(weight.astype(np.float64)) +
                 np.mean(bias.astype(np.float64)) +
                 out["data_cursor"][0] * 1e-9)
    return out, loss


def digest(state):
    import hashlib
    value = hashlib.sha256()
    for name in sorted(state):
        array = np.asarray(state[name])
        value.update(name.encode())
        value.update(array.dtype.str.encode())
        value.update(array.shape.__repr__().encode())
        value.update(array.tobytes())
    return value.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def verify(manifest):
    full = load_npz(manifest["full_state"])
    oracle = S2DeltaOracle(full, block_size=32, small_threshold=8)
    ring = FileS2Ring(manifest["ring_dir"], slot_count=128,
                      slot_size=4 * 1024 * 1024)
    frames = [ring.read(index) for index in range(100)]
    recovered = oracle.recover(full, frames)
    if recovered["generation"] != 100 or recovered["last_step"] != 100:
        raise AssertionError("R0 recovery generation/step mismatch")
    continuous = full
    continuous_losses = []
    for step in range(1, 101):
        continuous, _ = advance(continuous, step)
    if digest(recovered["state"]) != digest(continuous):
        raise AssertionError("recovered state differs from continuous step 100")
    replay = recovered["state"]
    for step in range(101, 111):
        continuous, continuous_loss = advance(continuous, step)
        replay, replay_loss = advance(replay, step)
        continuous_losses.append(continuous_loss)
        if continuous_loss != replay_loss or digest(continuous) != digest(replay):
            raise AssertionError(f"continuation mismatch at step {step}")
    return {"status": "pass", "generation": recovered["generation"],
            "last_step": recovered["last_step"], "frames": len(frames),
            "continuation_steps": 10, "byte_exact": True,
            "losses": continuous_losses}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("write", "verify", "orchestrate"),
                        default="orchestrate")
    parser.add_argument("--output-root", default="results/wp3-closeout-20260826/r0_cpu")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    if args.phase == "verify":
        print(json.dumps(verify(json.loads(Path(args.manifest).read_text())),
                         sort_keys=True))
        return
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    full = initial_state()
    full_path = root / "full_state.npz"
    np.savez(full_path, **full)
    ring_dir = root / "ring"
    ring = FileS2Ring(ring_dir, slot_count=128, slot_size=4 * 1024 * 1024)
    oracle = S2DeltaOracle(full, block_size=32, small_threshold=8)
    current = full
    for step in range(1, 101):
        current, _ = advance(current, step)
        oracle.set_current(current)
        frame = oracle.observe(step)
        oracle.ack(frame)
        ring.write(frame)
    manifest = {"full_state": str(full_path), "ring_dir": str(ring_dir)}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    command = [sys.executable, str(Path(__file__).resolve()), "--phase", "verify",
               "--manifest", str(manifest_path)]
    completed = subprocess.run(command, capture_output=True, text=True,
                               check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr)
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
