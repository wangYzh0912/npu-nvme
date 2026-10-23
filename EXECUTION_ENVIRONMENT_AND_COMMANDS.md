# Qwen3 training baseline: environment and commands

The supported workload is Qwen3-8B with TP4/DP1/PP1 and a fixed
local text fixture. FULL state includes model parameters, Adam state, training
position and random-generator controls.

## Selected environment

| Item | Selected value |
| --- | --- |
| Hardware | 8 x Ascend 910B3; latest targeted acceptance maps TP4 to physical NPU 4--7 |
| Driver | 24.1.rc3 |
| Python | `/models/npu_nvme_exp/user7-stack/conda/bin/python` (3.11.4) |
| MindSpore / MindFormers | 2.7.1 / 1.7 |
| CANN | `/models/npu_nvme_exp/user7-stack/cann-install/ascend-toolkit/8.3.RC1` |
| Model | `/models/Qwen3-8B` |
| Environment manifest | `/models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/environments.json` |
| ABI3 library | `/models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/candidate/build/libnpu_nvme.so.3.0` |
| ByteCheckpoint Python | `/home/user7/npu-nvme-baseline-envs/bytecheckpoint/bin/python` |
| ByteCheckpoint source | `/home/user7/npu-nvme-baseline-upstreams/ByteCheckpoint` |

The launcher constructs the process environment from the manifest and records
the actual libraries, package identity and model hashes. Do not infer the
runtime from the interactive shell environment.

The current ABI3 candidate passed targeted NPU4--7 FULL correctness acceptance
on 2026-10-22. The full performance campaign remains deferred.
The 2026-09-16 ABI2 installations remain unchanged historical evidence.
The current loader rejects ABI2; rerun old experiments only with their original
source/environment pair, not by mixing their hardcoded `.so.2` with this loader.

## Targeted FULL closure acceptance

Build an isolated environment with the recorded patched DPDK ring dependency:

```bash
python3 tools/build_stage_environments.py --out /models/NEW_ABI3_ENVIRONMENT \
  --spdk third_party/spdk \
  --dpdk-ring /models/npu_nvme_exp/user7-stack/release-build-dependencies/librte_mempool_ring_fixed.a
```

Select its manifest in the training config before testing. The current default
already selects the accepted 005 environment. Run from the repository root:

```bash
sudo -S env ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
  /models/npu_nvme_exp/user7-stack/conda/bin/python tools/full_checkpoint_acceptance.py \
  --config config/qwen_training.json --npu-ids 4,5,6,7 \
  --output /models/NEW_FULL_CLOSURE_RUN < .sudo_pw
```

This runs only the Ours correctness chain and publication interruption recovery
against a none oracle, not the full performance matrix. It never formats media.
Selected physical devices must be idle or the command returns 3 (`blocked`)
before any worker starts. TP ranks remain 0--3 and use the selected physical map
through `ASCEND_RT_VISIBLE_DEVICES`; mapping is part of run/checkpoint identity.
A retained process keeps its existing lease for explicit recovery. The driver
requires output-filesystem space for two converted model caches plus 8 GiB.
Completed runs reuse SHA256-verified model shards/name maps through hardlinks;
checkpoint files are not shared this way. Do not modify converted weights in place.

Current evidence: `/models/npu_nvme_exp/user7-stack/full-closure-acceptance-npu4-7-20261021-005/result.json`
has `status=pass`, seven scenarios completed as expected and five oracle comparisons
passed. The interruption worker deliberately returns 1; the driver checks the
fault proof, unpublished catalog and safely closed owner before automatic fresh
recovery. It recovered raw19/step4, continued to step8 and saved raw20. No manual
retry was needed. The earlier attempt004 failure and manual retry remain historical.
Native-owner locks now use physical IDs and open existing files without `O_CREAT`,
fixing the reproduced cross-user sticky-`/tmp` `EACCES`. See `FULL_CHECKPOINT_FLOW.md`.

The independent 2026-10-24 rerun also returned 0: seven expected scenarios and
five oracle comparisons passed at
`/models/npu_nvme_exp/user7-stack/full-closure-acceptance-npu4-7-20261024-001/result.json`.
It reconciled raw27/step4, continued and saved raw28/step8 with safe owner close.
All eight NPUs were healthy and idle afterward; raw83 remained on
`uio_pci_generic`, protected84 on `nvme`, and `/models` on `/dev/nvme1n1`.
Execution sources are frozen per run. The submission excludes inactive research
hooks and integrates upstream D2 changes; its 390 software/ABI checks are recorded
separately in `DEVELOPMENT_STATUS.md`, alongside 45 C sanitizer checks.

## Software checks

```bash
PYTHONPATH=.:python /models/npu_nvme_exp/user7-stack/conda/bin/python -m pytest \
  tests/python tests/b2 tests/c2 tests/campaign tests/d1 tests/d2 tests/e1 tests/regression \
  --ignore=tests/regression/test_c_host_safety.py --ignore=tests/c2/test_retirement.py \
  -q --junitxml=/tmp/npu-nvme-full-closure-npu-map-python.xml

PYTHONPATH=.:python /models/npu_nvme_exp/user7-stack/conda/bin/python -m pytest \
  tests/regression/test_c_host_safety.py -q --junitxml=/tmp/npu-nvme-full-closure-c.xml

/models/npu_nvme_exp/user7-stack/conda/bin/python scripts/run_user_environment.py \
  --manifest /models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/environments.json \
  --profile candidate -- python -m pytest tests/c2/test_retirement.py \
  -q --junitxml=/tmp/npu-nvme-full-closure-npu-map-abi3.xml
```

The C harness compiles production C sources with ASan/UBSan and external-call
stubs. It does not initialize hardware. The earlier broad software/C/ABI checks
passed 417, 43, and 3 cases. The latest C harness has 45 passing cases, plus the
cross-user existing-file case executed as root. Final focused Python/ABI/contract
checks passed 30 cases:

```bash
env PYTHONPATH=python:. \
  NPU_NVME_LIBRARY_PATH=/models/npu_nvme_exp/user7-stack/full-closure-abi3-20261021-005/candidate/build/libnpu_nvme.so.3.0 \
  LD_LIBRARY_PATH=/models/npu_nvme_exp/user7-stack/cann-install/ascend-toolkit/latest/lib64 \
  /models/npu_nvme_exp/user7-stack/conda/bin/python -m pytest -q \
  tests/python/test_full_closure_acceptance.py tests/c2/test_retirement.py \
  tests/python/test_full_closure_contracts.py tests/python/test_training_reconciliation.py \
  --junitxml=/tmp/npu-nvme-full-closure-final-python.xml
```

To repeat the cross-user case with an explicit binary location, first run the C
harness above with `GATE_ARTIFACT_DIR=/tmp/npu-nvme-c-host`, then:

```bash
sudo -S /tmp/npu-nvme-c-host/compiled/host.elf native_owner_existing_file < .sudo_pw
```

These software checks do not replace the separate hardware acceptance evidence.

## Training

Run from the repository root and always choose a new output directory.

```bash
python3 train.py preflight --config config/qwen_training.json

sudo python3 train.py fit --config config/qwen_training.json \
  --output /models/qwen-source --stop-step 8

sudo python3 train.py fit --config config/qwen_training.json \
  --output /models/qwen-resume --resume \
  --checkpoint-root /models/qwen-source/checkpoints \
  --generation latest --stop-step 24
```

To verify continuation against an uninterrupted oracle, use this alternative
to the resume command above while the source catalog is still at step 8:

```bash
python3 train.py fit --config config/qwen_training.json --method none \
  --output /models/qwen-oracle --stop-step 24

sudo python3 train.py verify-restart --config config/qwen_training.json \
  --output /models/qwen-verify \
  --checkpoint-root /models/qwen-source/checkpoints \
  --oracle-run /models/qwen-oracle --stop-step 24
```

Select `none`, `mindspore_native_save`, `ours`, or `bytecheckpoint_host` with
`--method`. Ours requires root for raw SPDK access. The other methods may run
without root. `--generation` is a catalog generation, not an optimizer step.
The restore target must exceed the selected checkpoint step and remain within
the frozen learning-rate horizon. Resumed saves update the same checkpoint
catalog, so a catalog already advanced to step 24 cannot resume again with
`--stop-step 24`.

`inspect` reads `checkpoint_root` from its config and reports accepted and
rejected committed generations. `latest` skips invalid or no-longer-retained
entries; loading still performs full identity, integrity and all-rank checks.

## Optional exhaustive campaign

```bash
sudo python3 train.py benchmark --config config/qwen_training.json \
  --output /models/qwen-acceptance --profile all
```

The full campaign is implemented but its unfinished matrix is deferred. Its
required counts and resume procedure are in
`docs/plans/DEFERRED_VALIDATION.md`. A `running` or `deferred` result is not a
full-campaign pass.

## Build

```bash
python3 tools/build_training_runtime.py \
  --manifest /models/npu_nvme_exp/user7-stack/qwen-release-runtime-20260916-001/environments.json \
  --out /models/npu_nvme_exp/user7-stack/NEW_RUNTIME \
  --spdk /home/user7/npu-nvme/third_party/spdk \
  --dpdk-ring /models/npu_nvme_exp/user7-stack/release-build-dependencies/librte_mempool_ring_fixed.a
```

The historical ABI2 reproducibility build is
`/models/npu_nvme_exp/user7-stack/qwen-mainline-runtime-20260916-001`. Its ABI2
library SHA256 matches the hardware-tested runtime:
`ba6dc8156be8d1f9c353d309e881afe6e0391916471be6152df390aa13c4d2b7`.
Keep the validated configuration on its original environment manifest because
the environment identity is part of checkpoint compatibility.

The default configuration now selects the new ABI3 environment above. It has
SHA256 `070f87d5bc3a2a32723274c9319e5b08356f5f36163e3ad3594b81a12daf4189`.
Do not overwrite historical manifests or relabel historical checkpoints as ABI3.

## Storage boundaries

Raw access is limited to PCI device `0000:83:00.0`. The Qwen D2 region begins
at 256 GiB and has length 1 TiB. Device `0000:84:00.0` contains the protected
`/models` filesystem and must not be rebound or formatted.

Hardware jobs use `/models/npu_nvme_exp/user7-stack/hardware-campaign.lock` and
its lease file. A retained run requires process, DMA and device-state diagnosis;
do not remove its lease merely to admit another job. Routine training does not
require an eight-NPU reset or a host reboot.

Research experiments and their results remain on `incremental-phase1-results`.
They are outside this FULL checkpoint delivery.
