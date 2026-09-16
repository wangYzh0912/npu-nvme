#!/usr/bin/env python3
"""Compare the phase-one device scorer with its NumPy oracle on one NPU."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import mindspore as ms
    from npu_nvme.experiments.device_score import score_parameter
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=args.device)
    rng = np.random.default_rng(42)
    count = 65536 * 3 + 17
    reference = rng.normal(size=count).astype(np.float32)
    current = reference.copy()
    current[13:70003] += np.float32(.125)
    ms.runtime.reset_peak_memory_stats()
    before = ms.runtime.memory_allocated()
    start = time.monotonic_ns()
    result, cell = score_parameter(ms, ms.Tensor(current), ms.Tensor(reference))
    actual = result.asnumpy(); ms.hal.synchronize()
    elapsed = time.monotonic_ns() - start
    after = ms.runtime.memory_allocated(); peak = ms.runtime.max_memory_allocated()
    padded = np.pad((current.astype(np.float64) - reference.astype(np.float64)),
                    (0, (-count) % 65536)).reshape(-1, 65536)
    expected = np.sum(padded * padded, axis=1)
    report = {"status": "pass" if np.allclose(actual, expected, rtol=2e-5, atol=1e-4) else "fail",
              "device": args.device, "elements": count, "blocks": len(actual),
              "actual": actual.tolist(), "expected": expected.tolist(),
              "elapsed_ns_including_compile": elapsed, "memory_before": before,
              "memory_after": after, "memory_peak": peak,
              "extra_peak_bytes": max(0, peak - before), "cell": type(cell).__name__}
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
