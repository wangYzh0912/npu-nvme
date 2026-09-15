"""Strict FULL checkpoint API. Native ABI 2; initialization is explicit."""
if __name__ != "npu_nvme":
    raise ImportError("Use npu_nvme to preserve module identity")
from .strict_checkpoint import StrictCheckpoint
from .runtime.errors import CheckpointBusyError

__all__ = ["StrictCheckpoint", "CheckpointBusyError"]
