# Qwen3 training baseline: environment and commands

The supported workload is Qwen3-8B with TP4/DP1/PP1 on NPU 0--3 and a fixed
local text fixture. FULL state includes model parameters, Adam state, training
position and random-generator controls.

## Selected environment

| Item | Selected value |
| --- | --- |
| Hardware | 8 x Ascend 910B3; this entry uses NPU 0--3 |
| Driver | 24.1.rc3 |
| Python | `/models/npu_nvme_exp/user7-stack/conda/bin/python` (3.11.4) |
| MindSpore / MindFormers | 2.7.1 / 1.7 |
| CANN | `/models/npu_nvme_exp/user7-stack/cann-install/ascend-toolkit/8.3.RC1` |
| Model | `/models/Qwen3-8B` |
| Environment manifest | `/models/npu_nvme_exp/user7-stack/qwen-release-runtime-20260916-001/environments.json` |
| ABI2 library | `/models/npu_nvme_exp/user7-stack/qwen-release-runtime-20260916-001/build/libnpu_nvme.so.2.0` |
| ByteCheckpoint Python | `/home/user7/npu-nvme-baseline-envs/bytecheckpoint/bin/python` |
| ByteCheckpoint source | `/home/user7/npu-nvme-baseline-upstreams/ByteCheckpoint` |

The launcher constructs the process environment from the manifest and records
the actual libraries, package identity and model hashes. Do not infer the
runtime from the interactive shell environment.

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

The final reproducibility build is
`/models/npu_nvme_exp/user7-stack/qwen-mainline-runtime-20260916-001`. Its ABI2
library SHA256 matches the hardware-tested runtime:
`ba6dc8156be8d1f9c353d309e881afe6e0391916471be6152df390aa13c4d2b7`.
Keep the validated configuration on its original environment manifest because
the environment identity is part of checkpoint compatibility.

## Storage boundaries

Raw access is limited to PCI device `0000:83:00.0`. The Qwen D2 region begins
at 256 GiB and has length 1 TiB. Device `0000:84:00.0` contains the protected
`/models` filesystem and must not be rebound or formatted.

Hardware jobs use `/models/npu_nvme_exp/user7-stack/hardware-campaign.lock` and
its lease file. A retained run requires process, DMA and device-state diagnosis;
do not remove its lease merely to admit another job. Routine training does not
require an eight-NPU reset or a host reboot.
