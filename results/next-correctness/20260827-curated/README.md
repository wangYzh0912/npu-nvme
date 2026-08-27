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

The GPT-2 XL record deliberately distinguishes byte-exact restore from the
strict continuation tolerance result; it must not be read as proof of exact
cross-process Ascend numerical determinism.
