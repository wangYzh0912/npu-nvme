"""Current model construction, warmup and numerical health checks."""
import ctypes
import os
import time

import mindspore as ms
import numpy as np
from mindspore import nn, context

from npu_nvme.framework.cells import TrainOneStepCell


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








# -- SPDK environment + StrictCheckpoint factory ----------------------------



# -- FaF Reactor step-poller setup -----------------------------------------



# -- Delta-checkpoint helpers -----------------------------------------------






# -- Timing callbacks -------------------------------------------------------





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

    CRITICAL: Must be called before any StrictCheckpoint operations
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
    from npu_nvme.framework.cells import TrainOneStepCell

    warmup_cell = cell or TrainOneStepCell(model, opt)
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
