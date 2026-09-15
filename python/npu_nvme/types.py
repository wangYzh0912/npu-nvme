"""Current FULL receipts and evidence validation."""
from typing import Mapping

def validate_result_gate(result: Mapping) -> None:
    """Reject a result that claims success without durable restore evidence."""
    if not isinstance(result, Mapping) or result.get("status") not in (
            "pass", "fail", "blocked", "unsupported"):
        raise ValueError("result requires an explicit valid status")
    if result.get("mode") == "none":
        if result.get("restore_verified") is True:
            raise ValueError("none baseline cannot claim restore_verified")
        return
    required = ("request_id", "generation", "persisted", "restore_verified")
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError(f"result missing FULL gate fields: {missing}")
    for name in ("persisted", "restore_verified"):
        if type(result[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if result.get("status") == "pass" and not all(
            result[name] for name in ("persisted", "restore_verified")):
        raise ValueError("successful FULL result lacks persistence/restore proof")


# D1 strict runtime receipts.  These records are immutable after publication.
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

@dataclass(frozen=True)
class Reservation:
    request_id: str
    generation: int
    writer_epoch: str
    rank_id: int
    step: int
    slot: int

@dataclass(frozen=True)
class TransferReceipt:
    request_id: str
    generation: int
    bytes_written: int
    chunks: int
    durable: bool
    digest: str

@dataclass(frozen=True)
class CommitReceipt:
    request_id: str
    generation: int
    metadata_generation: int
    slot: int
    retained_generations: Tuple[int, ...]

@dataclass(frozen=True)
class RestoreReceipt:
    request_id: str
    generation: int
    step: int
    bytes_read: int
    chunks: int
    digest: str
    ready: bool
    applicable_controls: Tuple[str, ...] = ()
