# M1 Checkpoint I/O and AIC/AIV Feasibility Report

Date: 2026-11-18. Hardware: Ascend 910B, CANN 8.3.RC1, MindSpore 2.7.1.
The Qwen run used NPU4-7, TP4, full state, sequence length 128, fixed seed 42.

## Finding

The previously observed M1 failure is a native NVMe write request that did not
complete before the 60-second default wait expired. In the failed step-56 run,
rank 2 on physical NPU6 began the 4 MiB payload for chunk 5221 and had no
`native_submitted`, `native_completed`, `owner_payload_end`, or `source_safe`
event afterward. The outstanding request and borrowed DMA buffer were retained;
the other ranks then queued behind the owner lock and the collective ended with
peer EOF. Native counters were 357,752 submitted, 357,751 completed, one
outstanding request, one DMA in flight, and one retained buffer set.

This locates the failing boundary at host-to-NVMe payload submission/completion.
The logs and kernel journal available for that run did not identify why the
device command stalled. The 60-second wait is not evidence that the storage
command is safe to release, so no timeout change or source workaround was made.

A reduced Qwen3-8B continuation then restored step 52 / generation 13, trained
four steps, saved step 56 / generation 15, validated all four ranks, and closed
with zero outstanding requests, DMA operations, or retained buffers. It did
not reproduce the stall. One passing continuation does not close the cumulative
high-volume failure condition or complete M1 acceptance.

All four ranks completed step 56 with loss 0.0123657668 and no overflow. The
owner closed cleanly with 53,197/53,197 native NVMe commands and 53,200/53,200
transfer API calls completed; retry/error counts, outstanding requests,
in-flight DMA, and retained-buffer sets were all zero.

## Checkpoint Phases

The continuation stored 24,574,980,708 bytes per rank (98,299,922,832 bytes
total). All four rank operation records agree within milliseconds.

| Operation | Per-rank time | Evidence |
| --- | ---: | --- |
| Restore step 52 / generation 13 | 1004.76 s | Operation record; 4/4 restore-ready receipts and pin release |
| Save step 56 / generation 15, backend operation | 553.09 s | Operation record; committed generation receipt |
| Save step 56, blocking boundary | 576.96 s | Training boundary, includes admission, hashes, transport, and publish |
| Whole launch through safe close | 1905.57 s | Includes preflight, owner and framework startup, validation, restore, four steps, save, drain, close |

The save boundary's nine rank-local stages are recorded in
[`continuation-save-stage-step56.csv`](continuation-save-stage-step56.csv).
Their median durations across four ranks were:

| Stage | Median | Interpretation |
| --- | ---: | --- |
| Admission collectives | 0.208 s | Four-rank admission |
| Framework synchronize | 0.0001 s | Device/framework synchronization |
| Manifest read and hash | 22.964 s | 24.57 GB logical state per rank |
| Control capture | 0.001 s | Training/control state capture |
| Sidecar write | 0.015 s | Small metadata sidecar |
| Reservation | 0.031 s | One rank had a 0.455 s outlier |
| Save setup | 0.015 s | Owner request setup |
| Capture transport | 553.323 s | Dominant stage: source-safe payload transfer and owner I/O |
| Complete/publish collective | 0.399 s | Commit and rank completion |

The restore operation is 1004.76 s per rank and all ranks reported readiness,
then released the restore pin. This continuation did not instrument restore
internals as separate timed stages, so no per-stage restore attribution is
claimed. Resource admission recorded 6,223,298,562 bytes required HBM scratch
per rank and 7,751,925,760 bytes required host headroom; these are admission
estimates, not measured peaks.

For a separate save-focused E4 capture at step 28, the four-rank blocking save
boundary was 580.63-580.64 s. Its stage ledger is in
[`e4-save-stage-summary-step28.csv`](e4-save-stage-summary-step28.csv). The
largest observed interval unions were source preparation/copy 524.98 s,
bridge acknowledgement wait 543.53 s, owner lock wait 538.54 s, owner payload
531.58 s, and native API 302.75 s. These are overlapping interval unions over
25,504 chunk events and must not be added together or interpreted as device
service time. Per-chunk medians were 25.7 ms source preparation/copy, 62.0 ms
bridge acknowledgement wait, 26.9 ms owner lock wait, 22.3 ms owner payload,
and 12.7 ms native API.

The owner-side durable metadata/flush interval was 3.61 s, catalog publication
0.321 s, anchor write/flush 3.0 ms, and payload flush 1.8 ms. These are single
checkpoint events, not repeat distributions.

## AIC/AIV Activity Over Time

The E4 profiler covers the save-focused step-28-to-32 run. MindSpore task
timestamps were converted to a common 10 ms relative timeline for each rank in
[`e4-task-activity-10ms.csv`](e4-task-activity-10ms.csv). The profiler capture
spanned 585.65 s. Per NPU, the union of observed AIC task intervals was about
0.131 s (0.022% of the capture), AIV task intervals about 1.545 s (0.264%), and
same-time AIC/AIV task overlap 38-73 microseconds. This indicates little
simultaneous kernel-task activity during the long save-focused boundary.

These are task intervals, not hardware engine occupancy. They do not include
idle-engine samples and cannot distinguish a device wait from host scheduling.
The E4 MindSpore profiler explicitly had HBM/DDR profiling disabled, so it
provides no HBM bandwidth for the actual Qwen save.

The isolated and co-run controls use a separate NPU4 process with 100 Hz AIC
PMU and 50 Hz HBM PMU. Workloads were checked for finite/correct outputs. The
control summary is in [`control-window-summary.csv`](control-window-summary.csv)
and the 10 ms sampled timeline is in
[`control-timeseries-10ms.csv`](control-timeseries-10ms.csv).

| Control window | Median duration (3 repeats) | Mean AIC PMU | Mean AIV PMU | Observed card HBM read/write |
| --- | ---: | ---: | ---: | ---: |
| AIC-only, 4096x4096 BF16 matmul, 120 iterations | 119.57 ms | 49.27% | 0% | Median of repeat-window means: 356 / 1613 MiB/s |
| AIV-only, 16M-element BF16 vector add, 120 iterations | 239.47 ms | 0% | 1.43% | 195 / 814 MiB/s |
| Serial pair | 357.29 ms | 16.63% | 1.17% | 14,135 / 2500 MiB/s |
| Two-stream pair | 349.02 ms | 17.22% | 1.98% | 6317 / 1879 MiB/s |

The two-stream median was 2.3% faster than serial in this small fixed-order
control, while both AIC and AIV PMU were active within the same control windows.
Three repeats are insufficient to claim a stable co-run speedup or exact kernel
overlap. HBM values are measured device-wide totals; they are not split into
per-engine bandwidth. Each window spans only 6-18 HBM timestamps, and repeated
window means vary substantially, especially for isolated AIV and co-run read
traffic. Treat these as observed ranges for follow-up, not a bandwidth
characterization of Qwen training.

The HBM table contains four HBM stack rows per timestamp. The analysis sums
those four rows before calculating each timestamp's card total, then averages
timestamps inside the workload window. AIC/AIV percentages average sampled
core usage across the matching engine's sample rows. There was no other
workload on NPU4 during the controls.

## M1 Status and Next Check

The reduced continuation proves feasibility for restore, four training steps,
save, validation, and safe close on the pinned Qwen3-8B TP4 environment. It does
not reproduce the original cumulative request-volume stall, and the root cause
inside the non-completing NVMe command remains unknown. M1 acceptance therefore
remains open.

The useful next measurement is a small, bounded reproduction of cumulative
native request volume while capturing owner request and NVMe completion
latencies plus device-wide HBM/engine PMU in the same clock domain. Keep the
existing 60-second safety behavior and retained-buffer rule. Do not repeat the
full training matrix unless this targeted run changes the diagnosis.

## Evidence and Reproduction

Primary runtime evidence remains under:

- `/models/npu_nvme_exp/user7-stack/aic-aiv-m1-small-20261118/step52-to56-r2/`
- `/models/npu_nvme_exp/user7-stack/aic-aiv-m1-small-20261118/aic-aiv-controls-npu4-r4/`
- `/models/npu_nvme_exp/user7-stack/aic-aiv-io-closure-20261116-r11/evidence/runs/e4-ours/`

The compact source hashes and source paths are in
[`analysis-summary.json`](analysis-summary.json). The extraction tool is
[`checkpoint_io_m1_report.py`](../../../tools/checkpoint_io_m1_report.py);
the small-card workload is
[`checkpoint_io_aic_aiv_control.py`](../../../tools/checkpoint_io_aic_aiv_control.py).
