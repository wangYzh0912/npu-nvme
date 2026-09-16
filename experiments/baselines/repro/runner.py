"""Compatibility alias for the shared GPT-2 workload implementation."""
import sys
from npu_nvme.workloads import gpt2
sys.modules[__name__] = gpt2
