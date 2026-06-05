#!/usr/bin/env python3
"""
LLaMA2 single-card training with SPDK + WaitProbe checkpoint.

Usage:
  sudo su - root -c 'source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash && /root/miniconda3/envs/ms_2.5/bin/python /home/user7/npu-nvme/experiments/train_llama2_spdk.py'
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import mindspore as ms
from mindformers import Trainer, AutoModel, AutoTokenizer, AutoConfig


def main():
    # Placeholder — inherits train_llama2.py structure from python/
    # Uses direct_checkpoint for SPDK + WaitProbe integration
    pass


if __name__ == "__main__":
    main()
