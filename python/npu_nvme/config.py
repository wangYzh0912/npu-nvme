"""Read-only implementation limits, independent of framework and FFI."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyBudget:
    max_batch_items: int = 65536
    max_batch_aligned_bytes: int = 64 * 1024**3
    max_descriptor_bytes: int = 16 * 1024**2
    max_metadata_io_bytes: int = 1024**2
    max_checkpoint_slots: int = 16
    max_request_slots: int = 64
    max_dma_slots: int = 16
    alignment_bytes: int = 4096


DEFAULT_SAFETY_BUDGET = SafetyBudget()
