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
The framework reference, native MindSpore `save_checkpoint` reference, and
no-checkpoint baseline passed. The four current
ports preserve the paper mechanisms while making platform substitutions
explicit: DataStates and PCcheck use real ACL `_data_ptr()` D2H into bounded
pinned-Host slots and durable XFS; ByteCheckpoint uses its DDP planners,
official extra-state workflow, shared three-component completion counter, and
async local writer in an isolated CPU worker; FastPersist
uses its patched legacy serializer, FastFileWriter, AIO and double buffer in a
CPU worker. They are labelled `NPU semantic port` or `host-adapted semantic
port`, not unmodified CUDA artifacts. All four completed 30-step/10-generation
G3/G4 runs with source-process exit, fresh-process byte-exact restore and
strict three-step continuation loss checks. The raw adapter remains blocked by
SPDK being unable to attach the `uio_pci_generic`-bound `0000:83:00.0` device
(`PA IOVA` probe failure).

`mindspore_native_save` is a separate framework reference that calls
`mindspore.save_checkpoint(async_save=False)` directly and persists controls in
a durable sidecar. Its 30-step result and per-generation API/flush timings are
under `results/baseline-gpt2-repro-20260906/mindspore_native_save/`; it must not
be confused with the existing `mindspore_sync` raw-byte reference.
