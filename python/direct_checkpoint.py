"""Compatibility identity alias for npu_nvme.checkpoint."""
import sys
from npu_nvme import checkpoint as _canonical
sys.modules[__name__] = _canonical
