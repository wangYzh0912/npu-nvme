# Unified GPT-2 baseline reproduction

This package implements the plan's common MindSpore GPT-2 runner, full-state
schema, deterministic batch fixture, durable restore gate, and independent
adapter preflight records.  It intentionally keeps upstream candidates
isolated: a missing worker environment or an unportable CUDA/x86 checkout is a
recorded blocker, never a silent fallback to `pickle` or ordinary `torch.save`.

Run from the repository root with the existing MindSpore environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=.:python
export LD_LIBRARY_PATH=$PWD/build_out/lib:$PWD/build:$LD_LIBRARY_PATH
/home/user7/miniconda3/envs/ms_2.5/bin/python -m experiments.baselines.repro.cli preflight \
  --config experiments/baselines/repro/configs/gpt2_30step.json --all
/home/user7/miniconda3/envs/ms_2.5/bin/python -m experiments.baselines.repro.cli prepare \
  --config experiments/baselines/repro/configs/gpt2_30step.json
```

`run --steps N --checkpoint-every K` is an explicit short-smoke override.  A
normal run uses 30 formal steps and ten generations.  Every checkpoint waits
for durable completion, writes an event log, exits the source context, starts
a fresh restore process, checks bytes and controls, then compares continuation
losses to the source oracle.

The current hardware record is in `results/baseline-gpt2-repro-20260906`.
The framework reference and no-checkpoint baseline passed. DataStates and
PCcheck are represented by an explicit `mechanism-only` downgrade because
their locked sources require CUDA/x86 components. ByteCheckpoint and
FastPersist have isolated CPU worker import probes (the versions are locked in
`worker_environment.lock.json`), but their planners/serializers are not yet
connected to the MindSpore state bridge; their measured runs therefore use the
common shared-memory Host bridge and durable writer. The common runner obtains
bytes with synchronous MindSpore `asnumpy()`; `_data_ptr()` is used for alias
identification, not for an ACL asynchronous DMA submission. These runs
validate shared capture, persistence, checksum and fresh-restore semantics,
not the upstream CUDA backend performance. The native raw adapter remains blocked by
SPDK being unable to attach the `uio_pci_generic`-bound `0000:83:00.0` device
(`PA IOVA` probe failure).
