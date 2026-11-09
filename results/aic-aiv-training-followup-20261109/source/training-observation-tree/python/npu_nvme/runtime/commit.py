"""Pure metadata mount state shared by the strict catalog and storage adapter."""
from dataclasses import dataclass, field


@dataclass
class MetadataState:
    layout: object = None
    meta_dict: dict = field(default_factory=lambda: {"checkpoints": {}})
    metadata_generation: int = 0
    active_meta_slot: int = 0
    stack_start_bytes: int = 0
