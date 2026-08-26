"""Shared test harness for NPU-NVMe experiment scripts.

Reduces boilerplate across ~70 experiment files by centralising:
  - GPT-2 XL model construction + training setup
  - DirectCheckpoint factory + FaF Reactor step-poller wiring
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
import numpy as np
from mindspore import nn, context

from direct_checkpoint import (DirectCheckpoint, ProbeTrainOneStepCell, lib,
                                get_dev_ptr)


# -- Path defaults ----------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_TRAIN_MR = os.path.join(
    _REPO_ROOT, "dataset_prepare", "gpt2", "wikitext2_data",
    "gpt2_train_1025.mindrecord")


# -- Training setup factory -------------------------------------------------

def make_causal_lm_training(model_name="gpt2_xl", total_steps=20,
                            device_id=1, seq_len=1025, train_mr=None):
    """Create a causal-LM training setup for a supported MindFormers model.

    Returns (model, dataset, optimizer).  The dataset is pre-batched and
    limited to total_steps batches.
    """
    from mindformers import AutoModel, AutoConfig

    print(f"[Common] Building {model_name} model...", flush=True)
    cfg = AutoConfig.from_pretrained(model_name)
    if hasattr(cfg, "seq_length"):
        cfg.seq_length = seq_len
    if hasattr(cfg, "max_position_embeddings"):
        cfg.max_position_embeddings = max(seq_len, 1025)
    cfg.checkpoint_name_or_path = ""  # train from scratch
    model = AutoModel.from_config(cfg)

    mr_path = train_mr or _DEFAULT_TRAIN_MR
    ds = ms.dataset.MindDataset(mr_path, shuffle=True)
    # The GPT-2 corpus can contain token IDs above the LLaMA vocabulary. Keep
    # the same data source for path timing while making IDs valid for the
    # selected model; this is not a quality-training experiment.
    vocab_size = getattr(cfg, "vocab_size", None)
    if vocab_size and model_name != "gpt2_xl":
        ds = ds.map(operations=lambda *values: tuple(
            value % vocab_size for value in values),
                    input_columns=["input_ids", "labels"])
    ds = ds.batch(1, drop_remainder=True).take(total_steps)

    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
    return model, ds, opt


def make_causal_lm_checkpoint_model(model_name="gpt2_xl", seq_len=128):
    """Build a MindFormers causal-LM model without optimizer/dataset state.

    This is the model-scale checkpoint lane.  It preserves the real model
    tensor topology and dtype while avoiding Adam's additional 2--4x memory,
    which would otherwise prevent a 13B model from fitting on the intended
    NPU set and would contaminate model-only I/O measurements.
    """
    from mindformers import AutoModel, AutoConfig

    print(f"[Common] Building checkpoint-only {model_name} model...", flush=True)
    cfg = AutoConfig.from_pretrained(model_name)
    if hasattr(cfg, "seq_length"):
        cfg.seq_length = seq_len
    if hasattr(cfg, "max_position_embeddings"):
        cfg.max_position_embeddings = max(seq_len, 128)
    # MF 1.3.2 GLM4 inherits the GLM2 paged-attention path when use_past is
    # enabled.  The installed CANN build lacks its ReshapeAndCache adapter;
    # disabling KV-cache keeps the real parameter topology while making the
    # checkpoint-only allocation path executable.
    if model_name.startswith("glm") and hasattr(cfg, "use_past"):
        cfg.use_past = False
    cfg.checkpoint_name_or_path = ""
    return AutoModel.from_config(cfg), cfg


def warmup_checkpoint_model(model, cfg, seq_len=128):
    """Allocate model parameters and compile one small inference graph."""
    input_ids = ms.Tensor(np.zeros((1, seq_len), dtype=np.int32))
    position_ids = ms.Tensor(np.arange(seq_len, dtype=np.int32)[None, :])
    batch_valid_length = ms.Tensor(np.array([seq_len], dtype=np.int32))
    slot_mapping = ms.Tensor(np.arange(seq_len, dtype=np.int32))
    if str(getattr(cfg, "model_type", "")).startswith("glm"):
        output = model(input_ids, position_ids=position_ids,
                       batch_valid_length=batch_valid_length,
                       slot_mapping=slot_mapping)
    else:
        try:
            output = model(input_ids)
        except TypeError:
            output = model(input_ids, labels=input_ids)
    ms.hal.synchronize()
    print("  [Common] Checkpoint-only warmup complete — device addresses allocated.",
          flush=True)
    return output


def make_gpt2xl_training(total_steps=20, device_id=1, seq_len=1025,
                          train_mr=None):
    """Backward-compatible GPT-2 XL training factory."""
    return make_causal_lm_training("gpt2_xl", total_steps, device_id,
                                   seq_len, train_mr)


# -- SPDK environment + DirectCheckpoint factory ----------------------------

def make_ckpt(nvme_addr="0000:83:00.0", device_id=1, pipeline_depth=8,
              chunk_size=4 * 1024 * 1024, profiling=False,
              profiling_dir="./output/profiling", shm_id=None,
              keep_last_n=3, slot_size_gb=10, warmup_fn=None, **kwargs):
    """Create a DirectCheckpoint with standard defaults.

    Also sets SPDK_SHM_ID when a shared-memory ID is supplied.
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


# -- FaF Reactor step-poller setup -----------------------------------------

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


# -- Delta-checkpoint helpers -----------------------------------------------

def setup_delta_faf(ckpt, delta_cell, ckpt_interval=5):
    """Wire a DeltaTrainCell to the Reactor poller and initialise its delta area.

    Calls build_layout_for_delta + register_delta_tasks + delta_init.
    Must be called AFTER graph compilation (one dummy forward pass).

    Args:
        ckpt:           DirectCheckpoint instance.
        delta_cell:     compiled DeltaTrainCell.
        ckpt_interval:  trigger delta write every N steps.

    Returns:
        (dev_flag: int, dev_step: int) — HBM addresses.
    """
    # Build layout for delta output buffers
    ckpt.build_layout_for_delta(delta_cell)

    # Register with the C-layer Reactor step poller.
    dev_flag, dev_step = ckpt.register_delta_tasks(delta_cell, ckpt_interval)

    # Initialise delta ring area on NVMe
    ckpt.delta_init(slot_size_mb=256, slot_count=128)

    print(f"[Common] Delta FaF setup complete: flag={hex(dev_flag)} "
          f"step={hex(dev_step)} interval={ckpt_interval}", flush=True)
    return dev_flag, dev_step


def make_delta_training(total_steps=20, device_id=1, seq_len=1024,
                        block_size=524288, top_k_frac=0.10,
                        ckpt_interval=5, pipeline_depth=8,
                        profiling=False):
    """Create a full delta-checkpoint training environment.

    Returns:
        (model, dataset, optimizer, delta_cell, ckpt)
    """
    from delta_cell import DeltaTrainCell

    # Standard training setup
    model, ds, opt = make_gpt2xl_training(
        total_steps=total_steps, device_id=device_id, seq_len=seq_len)

    # Build delta cell
    print("[Common] Building DeltaTrainCell...", flush=True)
    delta_cell = DeltaTrainCell(
        model, opt,
        block_size=block_size,
        top_k_frac=top_k_frac)

    # DirectCheckpoint for epoch-boundary FULL ckpt
    ckpt = make_ckpt(
        device_id=device_id,
        pipeline_depth=pipeline_depth,
        profiling=profiling,
        keep_last_n=3,
        slot_size_gb=10)

    return model, ds, opt, delta_cell, ckpt


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
    # Required for DeltaTrainCell's loop unrolling (~772 params → deep graph)
    ms.set_recursion_limit(10000)


def warmup_model(model, opt, ds):
    """Run one dummy forward pass to trigger MS lazy memory allocation.

    CRITICAL: Must be called before any DirectCheckpoint operations
    (save, register_tasks, register_delta_tasks).  Without this,
    get_dev_ptr() returns 0 for all parameters and SPDK writes
    transfer zero bytes.

    Args:
        model: MindSpore model
        opt:   optimizer
        ds:    dataset (must have at least 1 batch)

    Returns:
        The loss from the warmup step (can be ignored).
    """
    from direct_checkpoint import ProbeTrainOneStepCell

    warmup_cell = ProbeTrainOneStepCell(
        model, opt, enable_probe=False, ckpt_interval=9999)
    it = ds.create_tuple_iterator()
    loss = warmup_cell(*next(it))
    ms.hal.synchronize()
    print("  [Common] Warmup forward pass complete — device addresses allocated.",
          flush=True)
    return loss
