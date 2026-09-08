# Minimal I/O Matrix Results (2026-09-04)

All I/O runs use Ascend 910B3, `0000:83:00.0`, 4 MiB chunks, and the
single-Reactor owner.  The source process is stopped before every fresh
restore gate.  Results are under `results/io-minimal-20260904/` and record the
commit, environment, raw events, and restore evidence.

## IO-1

- GPT-2 mechanism: none/serial/frozen/live; the corrected live run passed 10
  generations and retained restores.
- GPT-2 credibility: none, frozen_async, and live_async for seeds 41/42/43,
  20 generations each: 9/9 pass.  State digests are byte exact and
  continuation loss is verified.
- Stress: live_async, seed 41, interval 1, 1 s persistence delay, two slots:
  20/20 generations and fresh restore pass.  Backpressure is explicit in
  `admission_wait_ns`; it is not hidden as an async speedup.
- Live API returns before persistence and records device wait fences.  The
  measured GPT-2 update wait is sub-millisecond on the formal runs; the stress
  run has no failed generation.

## IO-2

- GPT-2 XL none and frozen_async seed 41, plus none seeds 42/43, completed
  with byte-exact fresh restores of the approximately 9.823 GB state.
- GPT-2 XL live_async seed 41 persisted all 12 generations and its state
  checksum was byte exact, but strict fresh continuation loss differed by up
  to `0.0015335` from the source-process oracle.  A relaxed diagnostic restore
  (`rtol=2e-4`) passed, but the formal gate remains failed.  Per the stop rule,
  no further XL live performance or profiler runs were started.
- Historical serial GPT-2 XL 3/3 restore results remain the serial baseline.

## IO-3

- GPT-2, 200 steps, interval 20, ten generations, keep-last-n=3:
  2-rank seeds 41/42/43 and 4-rank seed 41 all pass.
- Every run exits all source/coordinator processes, starts fresh HCCL restore
  workers, verifies all rank shards byte-for-byte, and verifies ten-step
  continuation loss.  Four-rank generations show physical slot reuse.
- Multi-rank metadata retains only the latest global manifest because the
  on-disk metadata slot is fixed at 400 KiB; physical FULL slots still rotate
  at `keep_last_n`.  This is recorded as
  `metadata_retained_generations=1`.
- Representative fault gates pass: rank partial-data exit, coordinator exit
  before metadata commit, and immediate source exit after metadata commit.

## IO-4

Fixed 256 MiB payload, 4 MiB chunks, depth 4, local NUMA, 5 warmups and 20
formal samples per group.  All 100 formal samples (B0, B2 with 1/4 producers,
and B4 with 1/4 producers) are byte exact.

| Path | Producers | Mean throughput | P99 latency | NVMe outstanding peak |
|---|---:|---:|---:|---:|
| B0 Host -> SPDK | 1 | 4.24 GB/s | 65 ms | 4 |
| B2 Unix -> memory sink | 1 | 476 MB/s | 571 ms | n/a |
| B2 Unix -> memory sink | 4 | 921 MB/s | 1.19 s | n/a |
| B4 Unix -> SPDK | 1 | 225 MB/s | 1.21 s | 1 |
| B4 Unix -> SPDK | 4 | 178 MB/s | 6.36 s | 1 |

The B5 endpoint is the four-rank IO-3 run.  Reactor CPU is about 4% in B4,
the qpair is underfed, and no controlled multi-owner gain exists.  Therefore
the multi-Reactor decision is **do not implement**; the observed loss is in
Host/Unix staging and coordinator serialization rather than a saturated
Reactor.

## Scope decision

FULL live_async is viable as a mechanism for GPT-2, but GPT-2 XL does not meet
the strict continuation oracle and cannot be advertised as non-blocking FULL.
The next branch is reserved for INC observation only; no incremental algorithm
or Top-K implementation is enabled by these results.
