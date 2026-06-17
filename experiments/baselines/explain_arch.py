#!/usr/bin/env python3
"""Explain PMU kernel time vs wall-clock step time on Ascend 910B Da Vinci."""

import json

with open("/home/user7/npu-nvme/experiments/output/phase1a_a1_pmu.json") as f:
    a1 = json.load(f)
with open("/home/user7/npu-nvme/experiments/output/phase1a_a2_50_pmu.json") as f:
    a2 = json.load(f)

total_a1 = a1["aic_time_ms"] + a1["aiv_time_ms"]
total_a2 = a2["aic_time_ms"] + a2["aiv_time_ms"]

print("=" * 72)
print("Ascend 910B Architecture and PMU Kernel Time Explained")
print("=" * 72)
print()
print("Hardware:")
print("  20 AI Cores (Da Vinci), each Core contains:")
print("    1x Cube Unit   - matrix multiply/conv (FP16/BF16/INT8)")
print("    2x Vector Unit - SIMD element-wise (Cast/Sub/Add/Reduce)")
print("    1x Scalar Unit - control flow, address calc")
print("  Total: 20 Cube + 40 Vector units running in parallel")
print()

print("PMU CSV columns:")
print("  'Task Type' = AI_CORE        => kernel runs on Cube Unit")
print("  'Task Type' = AI_VECTOR_CORE => kernel runs on Vector Unit")
print("  'Task Duration(us)'          => this kernel execution time")
print()
print("  Each CSV row = ONE kernel dispatch on ONE unit.")
print("  'Total kernel time' = SUM of all row durations.")
print("  This is NOT wall-clock - it is aggregated over all parallel units.")
print()

print("-" * 72)
print("A1 Baseline (no injection):")
print("  Cube kernels:     %8s  => total duration %8.0fms" % (f"{a1['aic_rows']:,}", a1["aic_time_ms"]))
print("  Vector kernels:   %8s  => total duration %8.0fms" % (f"{a1['aiv_rows']:,}", a1["aiv_time_ms"]))
print("  Sum of all kernel durations: %8.0fms" % total_a1)
print("  Wall-clock step time:                 ~379ms")
print("  Parallelism:        %.1fx  (kernel-time / wall-time)" % (total_a1 / 379))
print()

avg_cube_dur = a1["aic_time_ms"] * 1000 / a1["aic_rows"]
avg_vec_dur = a1["aiv_time_ms"] * 1000 / a1["aiv_rows"]
print("  Average Cube kernel duration:   %.1f us" % avg_cube_dur)
print("  Average Vector kernel duration: %.1f us" % avg_vec_dur)
print()

print("Interpretation:")
print("  The CSV records %s kernel dispatches across" % f"{a1['aic_rows'] + a1['aiv_rows']:,}")
print("  20 Cube + 40 Vector units over ~379ms of wall-clock time.")
print()
print("  Think of it as 60 lanes of a highway over 379ms.")
print("  The ~3,072ms is the sum of trip times if each car drove")
print("  the highway one after another (serial).")
print("  With 60 lanes, 3,072ms of serial work fits into 379ms.")
print()
print("  'Cube time 20.8%%'  = Cube-unit aggregates account for")
print("    20.8%% of the 3,072ms total kernel-time sum.")
print("  'Vector time 56.7%%' = Vector-unit aggregates account for")
print("    56.7%% of the 3,072ms total kernel-time sum.")
print()

print("-" * 72)
print("What 'Vector idle 66.8%%' means:")
print()
print("  For each Vector kernel dispatch, the PMU records how much of")
print("  the Vector ALU was actively computing vs idle.")
print()
print("  vec_ratio = 10.3%%  => Vector ALU (SIMD lanes) active")
print("  scalar_ratio = 22.8%% => Scalar pipe active")
print("  idle = 66.8%%  => neither Vec ALU nor Scalar active")
print()
print("  Why so much idle?")
print("  - Vector ops like Cast/Assign are bandwidth-bound, not ALU-bound")
print("  - They move data through the Vector pipe but the ALU sits idle")
print("  - Memory latency stalls account for much of the idle time")
print("  - GE may not perfectly pack kernels (scheduling gaps)")
print()
print("  This idle time REPRESENTS OPPORTUNITY:")
print("  The Vector Core has ~1,164ms of idle slots per step")
print("  that can be filled with compression compute ops.")
print("  Even though ALU util doesn't rise (our ops are also low-density),")
print("  the TIME SLOTS are available and don't compete with training.")
