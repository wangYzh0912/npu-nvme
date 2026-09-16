# Qwen3 training and FULL checkpoint baseline

The supported entry is `train.py`. It runs Qwen3-8B with tensor parallelism on
four Ascend 910B3 devices using the explicitly selected MindSpore 2.7.1,
MindFormers 1.7 and CANN 8.3.RC1 environment. Input is a deterministic local
text fixture. This is a checkpoint development baseline, not a general dataset
training pipeline.

Methods: `none`, `mindspore_native_save`, `ours`, `bytecheckpoint_host`.
Checkpoints include model parameters, Adam state and training controls. Saves
are synchronous and periodic; restore runs in a fresh process. Incremental
checkpointing, Qwen live checkpointing, resharding and GPT-2 compatibility are
outside this release's acceptance scope.

## Run

```bash
python3 train.py preflight --config config/qwen_training.json
sudo python3 train.py fit --config config/qwen_training.json --output /models/qwen-source --stop-step 8
sudo python3 train.py fit --config config/qwen_training.json --output /models/qwen-resume \
  --resume --checkpoint-root /models/qwen-source/checkpoints --generation latest --stop-step 24
```

Use new output directories. `--generation` accepts a catalog generation number,
which is distinct from the optimizer step. Configuration controls the LR
horizon, checkpoint interval and retention. Preserve the LR horizon and model
identity across restore. `none` has no persistent checkpoint. Ours requires
root for SPDK; the other methods can run as a regular user.

The raw device is `0000:83:00.0`; Qwen owns only the registered interval starting
at 256 GiB with length 1 TiB. The filesystem on `0000:84:00.0` is protected.
All hardware runs share a lock. Failed runs with retained processes or DMA
resources require diagnosis and recovery before another hardware run.

## Build And Verify

```bash
python3 tools/build_training_runtime.py --manifest /models/ENVIRONMENTS.json \
  --out /models/NEW_RUNTIME --spdk /path/to/built/spdk \
  --dpdk-ring /path/to/librte_mempool_ring_fixed.a
sudo python3 train.py benchmark --config config/qwen_training.json \
  --output /models/NEW_ACCEPTANCE --profile all
```

After a backend repair, `benchmark --reuse-multicycle /models/PREVIOUS_ACCEPTANCE`
can reuse complete method groups whose environment, training implementation,
native extension and completion records still match. It reruns comparisons and
records the original evidence paths. Incomplete groups and changed backends run
again; repeated timing runs are always new.

The build emits a new explicit environment manifest. Select it in the training
configuration. SPDK and the patched DPDK ring archive are external build
dependencies; the build records their paths and the archive hash. The native
MindSpore allocation extension must be built in each checkout.

The published baseline has passed the Qwen TP4 training entry, Native restart
chain, Ours periodic save, and Ours fresh-process restore/continue/save path.
The exhaustive four-method matrix and repeated timing campaign remain available
through `benchmark`; they are explicitly deferred and are not claimed as passed.
See [the deferred validation register](docs/plans/DEFERRED_VALIDATION.md).

Current progress and evidence locations are recorded in
[the delivery plan](docs/plans/QWEN_TRAINING_BASELINE.md). Historical experiments
are preserved in remote `archive/qwen-baseline-20260916/*` branches; large
payloads and original history remain in the hash-verified external archive.
Host-specific environment paths and complete command examples are in
[the execution guide](EXECUTION_ENVIRONMENT_AND_COMMANDS.md).
