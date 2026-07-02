"""Public helpers for the GPT-2 XL benchmark example."""

import os

import mindspore as ms
from mindspore import context, nn

from direct_checkpoint import DirectCheckpoint, ProbeTrainOneStepCell


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN_MR = os.path.join(
    REPO_ROOT, "dataset_prepare", "gpt2", "wikitext2_data",
    "gpt2_train_1025.mindrecord")


def init_env(device_id=0, mode=None):
    """Initialise the MindSpore Ascend runtime for benchmark execution."""
    if mode is None:
        mode = context.GRAPH_MODE
    context.set_context(mode=mode, device_target="Ascend", device_id=device_id)
    ms.common.set_seed(42)
    ms.set_recursion_limit(10000)


def make_gpt2xl_training(total_steps=20, device_id=0, seq_len=1025,
                         train_mr=None):
    """Create the GPT-2 XL model, dataset, and optimizer used by bench.py."""
    from mindformers import AutoConfig, AutoModel

    print("[Bench] Building GPT-2 XL model...", flush=True)
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = seq_len
    cfg.max_position_embeddings = seq_len
    cfg.checkpoint_name_or_path = ""
    model = AutoModel.from_config(cfg)

    mr_path = train_mr or DEFAULT_TRAIN_MR
    ds = ms.dataset.MindDataset(mr_path, shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(total_steps)

    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
    return model, ds, opt


def make_ckpt(nvme_addr="0000:83:00.0", device_id=0, pipeline_depth=8,
              chunk_size=4 * 1024 * 1024, profiling=False,
              profiling_dir="./output/profiling", shm_id=None,
              keep_last_n=3, slot_size_gb=10, warmup_fn=None, **kwargs):
    """Create a DirectCheckpoint with benchmark-oriented defaults."""
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


def warmup_model(model, opt, ds):
    """Run one training step so MindSpore allocates parameter device memory."""
    warmup_cell = ProbeTrainOneStepCell(
        model, opt, enable_probe=False, ckpt_interval=9999)
    it = ds.create_tuple_iterator()
    loss = warmup_cell(*next(it))
    if hasattr(ms, "hal") and hasattr(ms.hal, "synchronize"):
        ms.hal.synchronize()
    return loss
