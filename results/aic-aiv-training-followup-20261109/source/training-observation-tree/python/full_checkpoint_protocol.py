"""Compatibility alias for the canonical request vocabulary."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("npu_nvme.types")
