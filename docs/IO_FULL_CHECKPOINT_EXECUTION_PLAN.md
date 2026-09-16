# FULL Checkpoint I/O Execution Plan

This branch freezes incremental checkpoint work.  Only complete training state
is an acceptance result: model, optimizer, global step, loss scale, RNG state,
and data cursor.

## Stage order

0. **Unified baseline**: one entry point, fixed `serial`/`queue`/`async`
   modes, one result schema, commit/device/environment capture, and separate
   snapshot, DMA, NVMe, flush, and metadata timestamps.
1. **Single-card synchronous correctness**: Host/HBM round trips, metadata
   durability, GPT-2 restart/continuation, then GPT-2 XL restart/continuation.
2. **Single-card asynchronous data plane**: isolate `aclrtMemcpyAsync` and
   event polling from training; prove readback and real DMA/NVMe overlap.
3. **Training plus asynchronous I/O**: compare none/serial/queue/async with
   identical lifecycle and fresh-process restore gates.
4. **Backpressure and faults**: bounded slots, explicit BUSY/FAILED/TIMED_OUT,
   cleanup with in-flight requests, and previous-generation recovery.
5. **Multi-card single Reactor**: validate producer isolation and global
   PREPARE/PERSISTED_READY/COMMIT before HCCL training.
6. **Real HCCL training**: 2-card, 4-card, then larger models and rank fault
   recovery.
7. **Multiple Reactors only if measured**: implement only when one Reactor is
   CPU/qpair limited while the SSD is below its calibrated limit.

Every stage uses the same mandatory lifecycle:

`save -> data complete -> namespace flush -> metadata commit -> PERSISTED ->
source process exit -> fresh restore -> continue training`.

## Frozen scope

Top-K, R1/R2, incremental quantization, live parameter capture, graph
incremental optimization, and Delta-ring performance conclusions are excluded.
The raw device is `0000:83:00.0`; software and target-machine versions are
recorded by `EXECUTION_ENVIRONMENT_AND_COMMANDS.md`.

## Stage exits

Stage 0 exits only when every mode has the same machine-readable schema and
non-zero failure behavior.  Stage 1 exits only after GPT-2 and GPT-2 XL have
zero checksum mismatches and pass fresh-process continuation.  No performance
claim from later stages is valid before these exits.
