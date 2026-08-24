#!/usr/bin/env python3
"""
P0-U7 Clean-Room Test Suite — 10 configs, ~40min total.

Each test runs as an independent Python process to avoid SPDK state pollution.
Output: unified_clean_results.json in experiments/output/
"""
import os, sys, time, json, shlex, subprocess

REPO = "/home/user7/npu-nvme"
PY = "/home/user7/miniconda3/envs/ms_2.5/bin/python3"
ENV_PREFIX = (
    "export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest && "
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && "
    "export LD_LIBRARY_PATH=" + REPO + "/build_out/lib:$LD_LIBRARY_PATH && "
    "cd " + REPO
)

results = {}

def run(label, python_code, env_extra=""):
    """Run a single test and return parsed result."""
    env = ENV_PREFIX + " && " + env_extra
    script = f"""
import os, sys, time
os.chdir("{REPO}")
sys.path.insert(0, "{REPO}/python")
import numpy as np, mindspore as ms
from mindspore import nn, Tensor

{python_code}
"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # Run non-interactively.  The target machine must either execute this
    # suite as root or provide a narrowly scoped passwordless sudo rule.
    # Credentials must never be embedded in source code or piped via stdin.
    command = (
        f"{env} && exec {shlex.quote(PY)} "
        f"-c {shlex.quote(script)}"
    )
    proc = subprocess.run(
        ["sudo", "-n", "bash", "-lc", command],
        capture_output=True, text=True, timeout=600,
        cwd=REPO
    )

    # Parse output for RESULT: line
    for line in proc.stdout.split("\n"):
        if line.startswith("RESULT:"):
            parts = line.split(":", 1)[1].strip()
            data = {}
            for kv in parts.split(","):
                k, v = kv.split("=", 1)
                try: data[k.strip()] = float(v.strip())
                except: data[k.strip()] = v.strip()
            results[label] = data
            print(f"  {data}")
            return data

    print(f"  FAILED: {proc.stderr[-200:]}")
    results[label] = {"error": proc.stderr[-500:]}
    return None


# =====================================================================
# Phase 1: SPDK Base Overhead (R0-R4)
# =====================================================================

def phase1():
    # Common training code template
    train_code = """
ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=1)
ms.common.set_seed(42)
from mindformers import AutoModel, AutoConfig
cfg = AutoConfig.from_pretrained("gpt2_xl"); cfg.seq_length=1024; cfg.max_position_embeddings=1024
model = AutoModel.from_config(cfg)
ds = ms.dataset.MindDataset("dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
ds = ds.batch(1, drop_remainder=True).take(20)
opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
from direct_checkpoint import ProbeTrainOneStepCell
cell = ProbeTrainOneStepCell(model, opt, enable_probe=False)
{spdk_init}
times = []
class CB(ms.Callback):
    def on_train_step_begin(self,rc): self.t0=time.perf_counter()
    def on_train_step_end(self,rc): times.append(time.perf_counter()-self.t0)
ms_model = ms.Model(cell)
ms_model.train(epoch=1, train_dataset=ds, callbacks=[CB()], dataset_sink_mode=False)
arr=np.array(times[2:])
dt=[f\"{{t*1000:.0f}}\" for t in arr]
print(f\"RESULT:label={{lab}},mean={{arr.mean()*1000:.0f}},std={{arr.std()*1000:.0f}},p99={{np.percentile(arr,99)*1000:.0f}},n={{len(arr)}}\", flush=True)
{spdk_cleanup}
"""

    # R0: no SPDK
    run("R0_baseline", train_code.format(
        spdk_init="", spdk_cleanup="",
        lab="R0_baseline"))

    # R1: SPDK listener=off
    run("R1_spdk_off", train_code.format(
        spdk_init="""
from direct_checkpoint import DirectCheckpoint
ckpt = DirectCheckpoint(nvme_addr="0000:83:00.0", npu_device_id=1, pipeline_depth=8, requested_chunk_size=4*1024*1024, enable_profiling=False, keep_last_n=3, slot_size_gb=10)
""",
        spdk_cleanup="ckpt.cleanup()",
        lab="R1_spdk_off"),
        env_extra="NPU_NVME_LISTENER_MODE=off")

    # R2: SPDK listener=idle
    run("R2_spdk_idle", train_code.format(
        spdk_init="""
from direct_checkpoint import DirectCheckpoint
ckpt = DirectCheckpoint(nvme_addr="0000:83:00.0", npu_device_id=1, pipeline_depth=8, requested_chunk_size=4*1024*1024, enable_profiling=False, keep_last_n=3, slot_size_gb=10)
""",
        spdk_cleanup="ckpt.cleanup()",
        lab="R2_spdk_idle"),
        env_extra="NPU_NVME_LISTENER_MODE=idle")

    # R3: SPDK listener=qpoll
    run("R3_spdk_qpoll", train_code.format(
        spdk_init="""
from direct_checkpoint import DirectCheckpoint
ckpt = DirectCheckpoint(nvme_addr="0000:83:00.0", npu_device_id=1, pipeline_depth=8, requested_chunk_size=4*1024*1024, enable_profiling=False, keep_last_n=3, slot_size_gb=10)
""",
        spdk_cleanup="ckpt.cleanup()",
        lab="R3_spdk_qpoll"),
        env_extra="NPU_NVME_LISTENER_MODE=qpoll")

    # R4: SPDK listener=full
    run("R4_spdk_full", train_code.format(
        spdk_init="""
from direct_checkpoint import DirectCheckpoint
ckpt = DirectCheckpoint(nvme_addr="0000:83:00.0", npu_device_id=1, pipeline_depth=8, requested_chunk_size=4*1024*1024, enable_profiling=False, keep_last_n=3, slot_size_gb=10)
""",
        spdk_cleanup="ckpt.cleanup()",
        lab="R4_spdk_full"),
        env_extra="NPU_NVME_LISTENER_MODE=full")


# =====================================================================
# Phase 2: sink=TRUE Fire-and-Forget (R5-R6)
# =====================================================================

def phase2():
    sink_true_code = """
ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=1)
ms.common.set_seed(42)
from mindformers import AutoModel, AutoConfig
cfg = AutoConfig.from_pretrained("gpt2_xl"); cfg.seq_length=1024; cfg.max_position_embeddings=1024
model = AutoModel.from_config(cfg)
ds = ms.dataset.MindDataset("dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
ds = ds.batch(1, drop_remainder=True).take(20)
opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
from direct_checkpoint import ProbeTrainOneStepCell
cell = ProbeTrainOneStepCell(model, opt, enable_probe={enable_probe}, ckpt_interval=5)
{spdk_init}
epoch_times = []
class CB(ms.Callback):
    def on_train_epoch_begin(self,rc): self.t0=time.perf_counter()
    def on_train_epoch_end(self,rc): epoch_times.append(time.perf_counter()-self.t0)
ms_model = ms.Model(cell)
ms_model.train(epoch=2, train_dataset=ds, callbacks=[CB()], dataset_sink_mode=True, sink_size=10)
print(f"RESULT:label={{lab}},e1={{epoch_times[0]:.1f}},e2={{epoch_times[1]:.1f}},e2_per_step={{epoch_times[1]*1000/10:.0f}}", flush=True)
{spdk_cleanup}
"""

    # R5: sink=TRUE, no probe, no SPDK
    run("R5_sinkT_baseline", sink_true_code.format(
        enable_probe="False", spdk_init="", spdk_cleanup="",
        lab="R5_sinkT_baseline"))

    # R6: sink=TRUE, full FaF
    run("R6_sinkT_FaF", sink_true_code.format(
        enable_probe="True",
        spdk_init="""
import ctypes
from direct_checkpoint import DirectCheckpoint
ckpt = DirectCheckpoint(nvme_addr="0000:83:00.0", npu_device_id=1, pipeline_depth=8, requested_chunk_size=4*1024*1024, enable_profiling=False, keep_last_n=3, slot_size_gb=10)
dummy = Tensor(np.zeros((1, 1024), dtype=np.int32), ms.int32)
cell(dummy[0:1], dummy[0:1], dummy[0:1])
ckpt.register_tasks(model, step=0)
import direct_checkpoint
dc_lib = direct_checkpoint.lib
dev_flag = cell.flag._data_ptr()
dev_step = cell.step_counter._data_ptr()
dc_lib.npu_nvme_set_probe_flag_ptr(ckpt.ctx, ctypes.c_void_p(dev_flag))
dc_lib.npu_nvme_set_step_ptr(ckpt.ctx, ctypes.c_void_p(dev_step), 5)
if dev_flag == 0:
    dev_flag = dc_lib.npu_nvme_get_probe_flag_dev_ptr(ckpt.ctx)
""",
        spdk_cleanup="""
try:
    flag = ckpt.read_probe_flag_dev()
    print(f"RESULT_SAFETY: flag={int(flag)}, expected=4", flush=True)
except Exception as e:
    print(f"RESULT_SAFETY: error={e}", flush=True)
ckpt.cleanup()
""",
        lab="R6_sinkT_FaF"),
        env_extra="NPU_NVME_LISTENER_MODE=full")


# =====================================================================
# Phase 3: sink=FALSE Fire-and-Forget (R7-R8)
# =====================================================================

def phase3():
    sink_false_probe_code = """
ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=1)
ms.common.set_seed(42)
from mindformers import AutoModel, AutoConfig
cfg = AutoConfig.from_pretrained("gpt2_xl"); cfg.seq_length=1024; cfg.max_position_embeddings=1024
model = AutoModel.from_config(cfg)
ds = ms.dataset.MindDataset("dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord", shuffle=True)
ds = ds.batch(1, drop_remainder=True).take(20)
opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
from direct_checkpoint import ProbeTrainOneStepCell
cell = ProbeTrainOneStepCell(model, opt, enable_probe=True, ckpt_interval=5)
{spdk_init}
times = []
class CB(ms.Callback):
    def on_train_step_begin(self,rc): self.t0=time.perf_counter()
    def on_train_step_end(self,rc): times.append(time.perf_counter()-self.t0)
ms_model = ms.Model(cell)
ms_model.train(epoch=1, train_dataset=ds, callbacks=[CB()], dataset_sink_mode=False)
arr=np.array(times[2:])
dt=[f\"{{t*1000:.0f}}\" for t in arr]
print(f\"RESULT:label={{lab}},mean={{arr.mean()*1000:.0f}},std={{arr.std()*1000:.0f}},p99={{np.percentile(arr,99)*1000:.0f}},n={{len(arr)}}\", flush=True)
{spdk_cleanup}
"""

    # R7: sink=FALSE, step_counter only (no SPDK)
    run("R7_sinkF_probeOnly", sink_false_probe_code.format(
        spdk_init="", spdk_cleanup="",
        lab="R7_sinkF_probeOnly"))

    # R8: sink=FALSE, full FaF
    run("R8_sinkF_FaF", sink_false_probe_code.format(
        spdk_init="""
import ctypes
from direct_checkpoint import DirectCheckpoint
ckpt = DirectCheckpoint(nvme_addr="0000:83:00.0", npu_device_id=1, pipeline_depth=8, requested_chunk_size=4*1024*1024, enable_profiling=False, keep_last_n=3, slot_size_gb=10)
dummy = Tensor(np.zeros((1, 1024), dtype=np.int32), ms.int32)
cell(dummy[0:1], dummy[0:1], dummy[0:1])
ckpt.register_tasks(model, step=0)
import direct_checkpoint
dc_lib = direct_checkpoint.lib
dev_flag = cell.flag._data_ptr()
dev_step = cell.step_counter._data_ptr()
dc_lib.npu_nvme_set_probe_flag_ptr(ckpt.ctx, ctypes.c_void_p(dev_flag))
dc_lib.npu_nvme_set_step_ptr(ckpt.ctx, ctypes.c_void_p(dev_step), 5)
if dev_flag == 0:
    dev_flag = dc_lib.npu_nvme_get_probe_flag_dev_ptr(ckpt.ctx)
""",
        spdk_cleanup="""
try:
    flag = ckpt.read_probe_flag_dev()
    print(f"RESULT_SAFETY: flag={int(flag)}, expected=4", flush=True)
except Exception as e:
    print(f"RESULT_SAFETY: error={e}", flush=True)
ckpt.cleanup()
""",
        lab="R8_sinkF_FaF"),
        env_extra="NPU_NVME_LISTENER_MODE=full")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, default="all",
                       help="1, 2, 3, or all")
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        phase1()
    if args.phase in ("2", "all"):
        phase2()
    if args.phase in ("3", "all"):
        phase3()

    # Save results
    output_dir = os.path.join(REPO, "experiments", "output")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "unified_clean_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    for k, v in sorted(results.items()):
        if "error" in v:
            print(f"  {k}: ERROR {v['error'][:80]}")
        else:
            mean = v.get("mean", v.get("e2_per_step", "N/A"))
            print(f"  {k}: {mean}ms")
    print(f"\nSaved to {output_dir}/unified_clean_results.json")
