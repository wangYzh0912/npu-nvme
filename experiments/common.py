"""Shared test harness for NPU-NVMe experiment scripts.

Reduces boilerplate across ~70 experiment files by centralising:
  - GPT-2 XL model construction + training setup
  - DirectCheckpoint factory + FaF listener wiring
  - StepTimer / EpochTimer callback classes

Usage:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
    from common import *

    model, ds, opt = make_gpt2xl_training(total_steps=30)
    cell = ProbeTrainOneStepCell(model, opt, enable_probe=True, ckpt_interval=5)
    ckpt = make_ckpt(device_id=1, warmup_fn=...)
"""

import ctypes
import os
import time

import mindspore as ms
from mindspore import nn, context

from direct_checkpoint import (DirectCheckpoint, ProbeTrainOneStepCell, lib,
                                get_dev_ptr)


# -- Path defaults ----------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_TRAIN_MR = os.path.join(
    _REPO_ROOT, "dataset_prepare", "gpt2", "wikitext2_data",
    "gpt2_train_1025.mindrecord")


# -- Training setup factory -------------------------------------------------

def make_gpt2xl_training(total_steps=20, device_id=1, seq_len=1024,
                          train_mr=None):
    """Create a standard GPT-2 XL training setup.

    Returns (model, dataset, optimizer).  The dataset is pre-batched and
    limited to total_steps batches.
    """
    from mindformers import AutoModel, AutoConfig

    print("[Common] Building GPT-2 XL model (3.12 GB FP16)...", flush=True)
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = seq_len
    cfg.max_position_embeddings = seq_len
    model = AutoModel.from_config(cfg)

    mr_path = train_mr or _DEFAULT_TRAIN_MR
    ds = ms.dataset.MindDataset(mr_path, shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(total_steps)

    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
    return model, ds, opt


# -- SPDK environment + DirectCheckpoint factory ----------------------------

def make_ckpt(nvme_addr="0000:83:00.0", device_id=1, pipeline_depth=8,
              chunk_size=4 * 1024 * 1024, profiling=False,
              profiling_dir="./output/profiling", shm_id=None,
              keep_last_n=3, slot_size_gb=10, warmup_fn=None, **kwargs):
    """Create a DirectCheckpoint with standard defaults.

    Also sets SPDK_SHM_ID and (optionally) NPU_NVME_LISTENER_MODE env vars.
    """
    if shm_id is not None:
        os.environ.setdefault("SPDK_SHM_ID", str(shm_id))

    return DirectCheckpoint(
        nvme_addr=nvme_addr,
        npu_device_id=device_id,
        pipeline_depth=pipeline_depth,
        requested_chunk_size=chunk_size,
        enable_profiling=profiling,
        profiling_dir=profiling_dir,
        keep_last_n=keep_last_n,
        slot_size_gb=slot_size_gb,
        warmup_fn=warmup_fn,
        **kwargs,
    )


# -- FaF listener setup -----------------------------------------------------

def setup_faf_checkpointing(ckpt, model, cell, ckpt_interval=10):
    """Register tasks and wire probe flag + step_counter pointers to C layer.

    Must be called AFTER graph compilation (e.g. after a dummy forward pass
    or model.build()).

    Args:
        ckpt:           DirectCheckpoint instance
        model:          MindSpore nn.Cell (base model, not wrapper)
        cell:           ProbeTrainOneStepCell instance (with enable_probe=True)
        ckpt_interval:  trigger a write every N steps
    Returns:
        (dev_flag, dev_step) — device pointers for the probe flag and step counter
    """
    # Register parameter device pointers
    ckpt.register_tasks(model, step=0)

    dev_flag = get_dev_ptr(cell.flag)
    dev_step = get_dev_ptr(cell.step_counter)

    rc = lib.npu_nvme_set_probe_flag_ptr(
        ckpt.ctx, ctypes.c_void_p(dev_flag))
    if rc != 0:
        raise RuntimeError(f"set_probe_flag_ptr failed: {rc}")

    rc = lib.npu_nvme_set_step_ptr(
        ckpt.ctx, ctypes.c_void_p(dev_step), ckpt_interval)
    if rc != 0:
        raise RuntimeError(f"set_step_ptr failed: {rc}")

    # If MS lazy allocation left flag at 0, C layer self-allocated
    if dev_flag == 0 and hasattr(lib, "npu_nvme_get_probe_flag_dev_ptr"):
        dev_flag = lib.npu_nvme_get_probe_flag_dev_ptr(ckpt.ctx)
    ckpt.probe_flag_ptr = dev_flag

    print(f"[Common] FaF setup complete: flag={hex(dev_flag)} "
          f"step={hex(dev_step)} interval={ckpt_interval}", flush=True)
    return dev_flag, dev_step


# -- Timing callbacks -------------------------------------------------------

class StepTimer(ms.Callback):
    """Per-step wall-clock timer (works in sink=False mode)."""

    def __init__(self):
        super().__init__()
        self.times = []

    def on_train_step_begin(self, run_context):
        self._t0 = time.perf_counter()

    def on_train_step_end(self, run_context):
        self.times.append(time.perf_counter() - self._t0)


class EpochTimer(ms.Callback):
    """Per-epoch wall-clock timer (works in sink=True mode)."""

    def __init__(self):
        super().__init__()
        self.times = []

    def on_train_epoch_begin(self, run_context):
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, run_context):
        self.times.append(time.perf_counter() - self._t0)


# -- Standardised baseline environment --------------------------------------

def init_env(device_id=1, mode=None):
    """Initialise MindSpore GRAPH_MODE environment with deterministic seed.

    Args:
        device_id: Ascend NPU device ID
        mode:      ms.GRAPH_MODE (default) or ms.PYNATIVE_MODE
    """
    if mode is None:
        mode = context.GRAPH_MODE
    context.set_context(mode=mode, device_target="Ascend", device_id=device_id)
    ms.common.set_seed(42)
