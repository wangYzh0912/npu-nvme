# WP2/WP3 closeout evidence — 2026-08-26

All hardware runs in this directory use only PCIe `0000:83:00.0`; the 84:00.0
filesystem device was not modified. The source branch is `exp/wp2-wp3-remaining`.

| Directory | Scope | Result |
|---|---|---|
| `i1_numeric/` | GPT-2/GPT-2 XL real MindSpore short trajectories, 3 seeds × 10 steps | PASS; finite loss and model/Adam state |
| `i3_a9_normal/` | GPT-2 XL real HBM slots, 1/2/4 slots, SPDK write/readback | PASS after one retained padding negative run |
| `i6_ring/` | 100-frame raw ring, 16 slots, 6.25 wraps, restart and 11 fault cases | PASS |
| `r0_cpu/` | FULL + 100 S2 frames, cross-process restore and 10 continuation steps | PASS |
| `r1_r2_cpu/` | 36 deterministic R1/R2 policy combinations | PASS as CPU candidate scan only |
| `gpt2_13b_a7_retry2/` | Native GPT-2 13B, HCCL 4-card, seq_length 2048, one train step | PASS; training smoke only |
| `i7_xl_100_s10/` | GPT-2 XL 100-step numeric long chain, full state sampled every 10 steps | PASS; every-step loss finite |

The earlier `i7_xl_100/` directory is an intentionally retained interrupted
run: sampling the complete XL model+Adam state every step made it impractical.
It is not included in any aggregate. The final run samples complete state at
steps 10, 20, ..., 100 while checking loss at every step.

The JSON files are the authoritative structured results. Samples and timelines
are retained beside each result where applicable. For n < 30, p99 is explicitly
not reported; confidence intervals use the Student-t critical value.
