"""Compatibility alias; canonical training cells live in npu_nvme.framework.cells."""
import sys
from npu_nvme.framework import cells as _canonical
sys.modules[__name__] = _canonical
