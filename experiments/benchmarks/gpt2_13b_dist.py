#!/usr/bin/env python3
"""Minimal native-sequence GPT-2 13B multi-card training gate."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import context, nn
from mindspore.communication import get_group_size, get_rank, init
from mindformers import AutoConfig, AutoModel
from mindformers.modules.transformer import TransformerOpParallelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--output", default="/tmp/gpt2_13b_dist")
    args = parser.parse_args()
    if args.seq_len != 2048:
        raise ValueError("13B gate must use native seq_length=2048")
    init()
    rank = get_rank()
    world = get_group_size()
    device = int(os.getenv("ASCEND_DEVICE_ID", os.getenv("DEVICE_ID", rank)))
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend",
                        device_id=device)
    context.set_auto_parallel_context(
        device_num=world, parallel_mode="semi_auto_parallel",
        gradients_mean=True, full_batch=True)
    ms.set_seed(1000 + rank)

    cfg = AutoConfig.from_pretrained("gpt2_13b")
    cfg.seq_length = args.seq_len
    cfg.max_position_embeddings = args.seq_len
    cfg.batch_size = 1
    cfg.checkpoint_name_or_path = ""
    cfg.parallel_config = TransformerOpParallelConfig(
        data_parallel=1, model_parallel=world, pipeline_stage=1,
        micro_batch_num=1, gradient_aggregation_group=4)
    model = AutoModel.from_config(cfg)
    model.set_train(True)
    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)

    # The GPT-2 training cell consumes a record one token longer than the
    # configured sequence because it creates shifted labels internally.
    record_len = args.seq_len + 1
    input_ids = ms.Tensor(np.random.randint(
        0, int(cfg.vocab_size), (1, record_len), dtype=np.int32))
    attention_mask = ms.Tensor(np.ones((1, record_len), dtype=np.int32))
    sys_path = str(REPO_ROOT / "python")
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)
    from training_cell import ProbeTrainOneStepCell
    cell = ProbeTrainOneStepCell(model, optimizer, enable_probe=False,
                                 ckpt_interval=9999)
    losses = []
    for step in range(args.steps):
        loss = cell(input_ids, attention_mask)
        ms.hal.synchronize()
        value = float(np.asarray(loss.asnumpy()).reshape(()))
        if not np.isfinite(value):
            raise FloatingPointError(f"rank {rank} non-finite loss: {value}")
        losses.append(value)
    result = {"status": "pass", "rank": rank, "world_size": world,
              "device": device, "model": "gpt2_13b", "seq_length": args.seq_len,
              "steps": args.steps, "losses": losses,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank_{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
