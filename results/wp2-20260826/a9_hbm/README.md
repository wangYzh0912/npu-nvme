# A9 real HBM snapshot-slot evidence (2026-08-26)

This bundle closes the real MindSpore HBM snapshot-slot scope of A9 on
`0000:83:00.0`. GPT-2 XL parameters were copied into dedicated D2D HBM shadow
slots, drained by one SPDK owner, read back through the Host path, and checked
with a SHA-256 digest against the frozen HBM slot.

The normal matrix used 25 steps, 5 checkpoint triggers, 2 warmups and 3
formal samples for each of 1/2/4 slots. All configurations passed. Each slot
uses 3,280,687,104 bytes of HBM shadow state. The controlled slow-disk matrix
used a 5,000 ms delay before each write and 15 steps; it also passed for all
slot counts and demonstrates the backpressure effect.

| condition | slots | formal samples | mean slot wait (ms) | mean end-to-end (ms) | status |
|---|---:|---:|---:|---:|---|
| normal | 1 | 3 | 2,497.63 | 8,107.89 | PASS |
| normal | 2 | 3 | 0.07 | 10,817.96 | PASS |
| normal | 4 | 3 | 0.07 | 10,880.32 | PASS |
| slow disk, +5 s | 1 | 4 | 7,038.40 | 13,074.70 | PASS |
| slow disk, +5 s | 2 | 4 | 3,114.42 | 18,284.12 | PASS |
| slow disk, +5 s | 4 | 4 | 0.05 | 26,217.82 | PASS |

The result proves the tested HBM slot lifecycle and frozen-buffer integrity;
it does not yet prove multi-rank HBM checkpointing or long-training benefit.
Raw run IDs and environment snapshots are preserved in the subdirectories.
