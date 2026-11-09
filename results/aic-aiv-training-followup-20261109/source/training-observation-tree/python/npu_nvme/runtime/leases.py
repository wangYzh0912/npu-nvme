"""Admission leases retain buffers until their owning worker proves release safe."""
from dataclasses import dataclass


@dataclass(eq=False)
class CheckpointLease:
    sequence: int
    generation: int
    handle: object = None
    params: object = None
    live: bool = False
    started: bool = False
    released: bool = False
    quarantined: bool = False
