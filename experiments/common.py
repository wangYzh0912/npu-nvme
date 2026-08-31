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
                            device_id=1, seq_len=1025, train_mr=None,
                            dropout_rate=None, require_dataset=True):
    """Create a causal-LM training setup for a supported MindFormers model.

    ``seq_len`` is the input-record length.  MindFormers GPT-2 shifts a
    record internally, so its compiled model sequence length is
    ``seq_len - 1`` (1025 corpus tokens -> 1024 training tokens).

    Returns (model, dataset, optimizer).  The dataset is pre-batched and
    limited to total_steps batches.
    """
    from mindformers import AutoModel, AutoConfig

    print(f"[Common] Building {model_name} model...", flush=True)
    if seq_len < 2:
        raise ValueError("training record length must be at least two tokens")
    model_seq_len = seq_len - 1
    cfg = AutoConfig.from_pretrained(model_name)
    if dropout_rate is not None:
        if not 0.0 <= float(dropout_rate) < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        for field in ("embedding_dropout_prob", "hidden_dropout_rate",
                      "attention_dropout_rate"):
            if hasattr(cfg, field):
                setattr(cfg, field, float(dropout_rate))
    if hasattr(cfg, "seq_length"):
        cfg.seq_length = model_seq_len
    if hasattr(cfg, "max_position_embeddings"):
        # Keep the attention-mask/lower-triangle shape consistent with the
        # requested experiment sequence length.  The old lower bound of 1025
        # made a short 13B scale run fail during graph inference with a
        # [1,1025,1025] vs [1,seq_len,seq_len] broadcast error.
        cfg.max_position_embeddings = model_seq_len
    cfg.checkpoint_name_or_path = ""  # train from scratch
    model = AutoModel.from_config(cfg)
    # MindFormers causal-LM cells return inference tuples unless training mode
    # is enabled.  Feeding that tuple to value_and_grad used to make the
    # experiment harness differentiate logits/tokens/masks and corrupt both
    # parameters and Adam state during the nominal warmup step.
    model.set_train(True)

    # Some MindFormers 1.3.2 GPT-2 configs reconstruct the model with the
    # checkpoint's original sequence-length constants even after the config
    # fields above are changed. Rebuild these non-parameter helpers so the
    # short scale lane has matching [batch, seq_len] masks and positions.
    if model_seq_len != 1024:
        from mindformers.modules.transformer import AttentionMask
        model.get_attention_mask = AttentionMask(
            seq_length=model_seq_len,
            parallel_config=cfg.parallel_config.dp_mp_config)
        if hasattr(model, "backbone"):
            model.backbone.position_ids = ms.Tensor(
                np.arange(model_seq_len), ms.int32)
            if hasattr(model.backbone, "seq_length"):
                model.backbone.seq_length = model_seq_len

    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
    if not require_dataset:
        return model, None, opt

    mr_path = train_mr or _DEFAULT_TRAIN_MR
    # The source MindRecord's physical column order is attention_mask,
    # input_ids, labels.  Passing that tuple directly to GPT2LMHeadModel made
    # an all-ones attention mask act as input_ids.  Select and order the two
    # arguments the model actually consumes.
    ds = ms.dataset.MindDataset(
        mr_path, columns_list=["input_ids", "attention_mask"], shuffle=True)
    # The GPT-2 corpus can contain token IDs above the LLaMA vocabulary. Keep
    # the same data source for path timing while making IDs valid for the
    # selected model; this is not a quality-training experiment. The source
    # records are length 1025, so crop both columns for shorter scale runs.
    vocab_size = getattr(cfg, "vocab_size", None)
    if vocab_size:
        ds = ds.map(operations=lambda value: value[:seq_len] % vocab_size,
                    input_columns=["input_ids"])
    ds = ds.map(operations=lambda value: value[:seq_len],
                input_columns=["attention_mask"])
    ds = ds.batch(1, drop_remainder=True).take(total_steps)

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

def _require_incremental_enabled():
    if os.environ.get("NPU_NVME_FULL_ONLY") == "1":
        raise RuntimeError(
            "incremental checkpoint helpers are disabled in FULL-only mode")

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
    _require_incremental_enabled()
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
    _require_incremental_enabled()
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

def init_env(device_id=1, mode=None, seed=42):
    """Initialise MindSpore GRAPH_MODE environment with deterministic seed.

    Args:
        device_id: Ascend NPU device ID
        mode:      ms.GRAPH_MODE (default) or ms.PYNATIVE_MODE
    """
    if mode is None:
        mode = context.GRAPH_MODE
    context.set_context(mode=mode, device_target="Ascend", device_id=device_id)
    ms.common.set_seed(int(seed))
    # Required for DeltaTrainCell's loop unrolling (~772 params → deep graph)
    ms.set_recursion_limit(10000)


def warmup_model(model, opt, ds, cell=None):
    """Run one excluded real training step to allocate lazy device memory.

    CRITICAL: Must be called before any DirectCheckpoint operations
    (save, register_tasks, register_delta_tasks).  Without this,
    get_dev_ptr() returns 0 for all parameters and SPDK writes
    transfer zero bytes.

    Args:
        model: MindSpore model
        opt:   optimizer
        ds:    dataset (must have at least 1 batch)

    Args:
        cell: Optional pre-created training Cell. Long-running HBM
            experiments should pass the exact Cell used by the formal loop,
            so graph compilation is completed before HBM/SPDK resources are
            allocated.

    Returns:
        The finite scalar loss from the excluded warmup training step.
    """
    from direct_checkpoint import ProbeTrainOneStepCell

    warmup_cell = cell or ProbeTrainOneStepCell(
        model, opt, enable_probe=False, ckpt_interval=9999)
    it = ds.create_tuple_iterator()
    loss = warmup_cell(*next(it))
    ms.hal.synchronize()
    loss_array = np.asarray(loss.asnumpy())
    if loss_array.ndim != 0 or not np.isfinite(loss_array).all():
        raise FloatingPointError(
            f"warmup must return one finite scalar loss, got "
            f"shape={loss_array.shape} value={loss_array}")
    print("  [Common] Excluded training warmup complete — finite scalar loss "
          f"{float(loss_array):.8g}; device addresses allocated.", flush=True)
    return loss


def training_numeric_health(model, optimizer, include_optimizer=True):
    """Summarize finite-value health without retaining host state copies."""
    groups = [("model", model.get_parameters())]
    if include_optimizer:
        groups.extend((("optimizer/m", optimizer.moments1),
                       ("optimizer/v", optimizer.moments2),
                       ("optimizer/global_step", (optimizer.global_step,))))
    arrays = 0
    bad = []
    for prefix, parameters in groups:
        for parameter in parameters:
            value = np.asarray(parameter.asnumpy())
            arrays += 1
            if not np.issubdtype(value.dtype, np.inexact):
                continue
            nonfinite = int(value.size - np.count_nonzero(np.isfinite(value)))
            if nonfinite:
                bad.append({"name": f"{prefix}/{parameter.name}",
                            "nonfinite": nonfinite,
                            "elements": int(value.size),
                            "dtype": value.dtype.name})
    return {"arrays": arrays, "nonfinite_arrays": len(bad),
            "nonfinite": bad}
