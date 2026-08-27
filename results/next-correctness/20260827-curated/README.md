# Curated correctness evidence — 2026-08-27

This directory contains only small, reviewable summaries. Large local state
oracles, `.npy` tensors, profiling output, and raw runtime files remain outside
Git under the corresponding run directories.

All hardware runs target only `0000:83:00.0`.

| File | Scope | Result |
|---|---|---|
| `c1-gpt2.json` | Single-process GPT-2 full training-state restart | PASS |
| `c1-gpt2xl.json` | GPT-2 XL full training-state restart | loaded state exact; strict continuation diagnostic recorded separately |
| `c2-gpt2-2rank.json` | Two-rank single-SPDK-owner state commit/restart | PASS; stand-alone ranks, not HCCL |
| `r0-gpt2-100.json` | GPT-2 S2-R0 100-step/1-seed, periodic FULL, ring replay | PASS; 100 historical Delta, byte-exact fresh replay of latest 23-frame window, 10-step continuation |
| `r0-gpt2xl-open.json` | GPT-2 XL S2-R0 feasibility attempt | FULL PASS; R0 large-graph gate remains open, no Delta metadata committed |

The GPT-2 XL record deliberately distinguishes byte-exact restore from the
strict continuation tolerance result; it must not be read as proof of exact
cross-process Ascend numerical determinism.

The R0 record is a correctness closure for GPT-2 on the current single-rank
path. It does not close the planned injected-fault matrix or the GPT-2 XL
three-seed gate.
