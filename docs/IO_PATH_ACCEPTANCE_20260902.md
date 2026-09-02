# FULL I/O Path Hardware Acceptance (2026-09-02)

## Scope

This run validates FULL training state only on `0000:83:00.0`. Delta, Top-K,
quantization, live incremental capture, and multi-Reactor work remain frozen.
Every training checkpoint gate requires data persistence, source-process exit,
fresh-process restore, byte-exact state verification, and continuation-loss
verification. Throughput and overlap alone are not acceptance criteria.

The tested source base was `ec798bb38fb1ed3785a680a28bb68aeb8e8f55d5` plus
the changes committed with this report. Raw outputs are intentionally local
under `results/acceptance-20260902`; the compact evidence files listed below are
versioned with the implementation.

## Environment

- Ascend 910B3, MindSpore 2.5.0, MindFormers 1.3.2, CANN 7.5.0.1.129.
- NVMe `0000:83:00.0` is bound to `uio_pci_generic`.
- Persistent host setup: 1663 2-MiB hugepages, group `wheel` allowed to use
  hugepage shared memory, and `/dev/uio*` mode `0660` for group `wheel`.
- Final idle check found no NPU worker process and all 1663 hugepages free.

## Results

| Stage | Hardware result | Recovery result |
| --- | --- | --- |
| Python/C baseline | 99 Python tests and 3 C smoke binaries passed | N/A |
| Stage 1 GPT-2 serial | 3/3 FULL generations persisted; latest generation 1284 | byte exact, controls restored, continuation loss matched |
| Stage 1 GPT-2 XL serial | 3/3 FULL generations persisted; latest generation 1099 | byte exact, controls restored, continuation loss matched |
| Stage 2 screening | 108 configs, 324/324 readback samples passed | independent readback SHA-256 passed |
| Stage 2 formal | depths 2/4/8, 30/30 samples each after 10 warmups | all readbacks passed; median overlap 0.9945/0.9940/0.9948 |
| Stage 3 training/I/O | 24/24 runs passed: none/serial/queue/async, three seeds, two intervals | all I/O modes completed their configured fresh restore gates |
| Stage 4 fault/lifecycle | all 11 fault, timeout, saturation, and crash-window cases passed | failed generations stayed unpublished; request ring returned BUSY |
| Stage 4 pressure 2/2 | 6/6 delay/interval configs plus fault matrix passed | 36/36 accepted generations restored; no training admission BUSY |
| Stage 4 pressure 4/4 | 6/6 delay/interval configs plus fault matrix passed | 36/36 accepted generations restored; no training admission BUSY |
| Stage 5 single Reactor | 2-rank generation 1206 and 4-rank generation 1207 passed | every rank fresh-restored and continued |
| Stage 6 HCCL | 2-rank generation 1208 and 4-rank generation 1209 passed | every rank fresh-restored and continuation loss matched |

Stage 2 depth 1 correctly reported zero DMA/NVMe overlap. For depths 2 and
higher, the screening and formal timelines reported positive overlap without a
stream-synchronize completion fallback.

Stage 4 distinguishes two signals. The training pressure cases did not exhaust
their snapshot admission slots because foreground snapshot creation was slower
than the injected background delay. The dedicated saturation case did fill the
request ring and observed `-EBUSY`; the summary records these as
`training_admission_busy_observed=false` and
`request_ring_busy_observed=true` instead of conflating them.

## Multi-rank Boundary

Stage 5 and Stage 6 use rank-local FULL state sent to one SPDK-owning
coordinator. Global metadata is published only after every rank has persisted.
Stage 6 source training is a real HCCL data-parallel process group.

Fresh restore currently runs one rank at a time because the SPDK shared-memory
instance has one primary owner. It verifies each rank's model, optimizer,
control state, and continuation loss in a fresh standalone process. Rejoining
HCCL after restore is not part of this result and remains the next distributed
recovery gate.

## Reusable Single-card Benchmark

The canonical single-disk, single-card benchmark is
`experiments/benchmarks/run_single_card_full.py`. It is the shared implementation
used by Stage 1, Stage 3, and Stage 4; do not create a separate training loop for
future comparisons. A formal invocation is:

```bash
python experiments/benchmarks/run_single_card_full.py \
  --model gpt2 --mode async --checkpoint-steps 10 50 100 \
  --total-steps 110 --seq-len 129 --seed 41 --npu 7 \
  --pci 0000:83:00.0 --chunk-size 4194304 --pipeline-depth 4 \
  --checkpoint-slots 4 --run-dir results/single-card-full/gpt2_async_seed41
```

Use the root/CANN environment template in `EXECUTION_ENVIRONMENT_AND_COMMANDS.md`.
The command returns nonzero unless the checkpoint is persisted, the source
process exits, and the fresh restore plus continuation gate succeeds.

## Decision

The FULL single-card data/control path, single-Reactor multi-rank commit, and
2/4-rank HCCL save path are accepted for this scope. There is no evidence that
the single Reactor prevents the SSD from reaching its current calibrated
limits, so multi-Reactor implementation remains deferred.
