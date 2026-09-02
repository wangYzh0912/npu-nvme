"""Small, deterministic global commit protocol for multi-rank FULL state.

The protocol is deliberately independent of SPDK/HCCL.  Rank workers report a
manifest at PREPARE, then a checksum after their data is durable.  Metadata is
published only after every rank reaches PERSISTED_READY.  This module is used
by hardware runners and is fully unit-testable without devices.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional


class GlobalState(str, Enum):
    CREATED = "CREATED"
    PREPARING = "PREPARING"
    PERSISTING = "PERSISTING"
    COMMITTING = "COMMITTING"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


@dataclass
class RankRecord:
    rank_id: int
    step: int
    generation: int
    manifest_digest: str
    state: str = "SNAPSHOT_READY"
    data_digest: Optional[str] = None
    error: Optional[str] = None


@dataclass
class GlobalCommit:
    world_size: int
    step: int
    generation: int
    state: GlobalState = GlobalState.CREATED
    ranks: Dict[int, RankRecord] = field(default_factory=dict)
    error: Optional[str] = None

    def _fail(self, reason: str):
        if self.state == GlobalState.COMMITTED:
            raise RuntimeError("a committed generation cannot be aborted")
        self.state = GlobalState.ABORTED
        self.error = str(reason)

    def prepare(self, rank_id: int, step: int, generation: int,
                manifest_digest: str) -> None:
        if self.state in (GlobalState.ABORTED, GlobalState.COMMITTED):
            raise RuntimeError(f"generation is terminal: {self.state.value}")
        rank_id, step, generation = int(rank_id), int(step), int(generation)
        if not 0 <= rank_id < self.world_size:
            self._fail(f"invalid rank_id={rank_id}")
            raise ValueError(self.error)
        if step != self.step or generation != self.generation:
            self._fail("step/generation mismatch during PREPARE")
            raise ValueError(self.error)
        if rank_id in self.ranks:
            self._fail(f"duplicate PREPARE from rank {rank_id}")
            raise ValueError(self.error)
        self.ranks[rank_id] = RankRecord(
            rank_id, step, generation, str(manifest_digest))
        self.state = GlobalState.PREPARING
        if len(self.ranks) == self.world_size:
            self.state = GlobalState.PERSISTING

    def persisted_ready(self, rank_id: int, data_digest: str) -> None:
        if self.state != GlobalState.PERSISTING:
            raise RuntimeError("PERSISTED_READY is only valid after all PREPARE")
        record = self.ranks.get(int(rank_id))
        if record is None:
            self._fail(f"unknown rank {rank_id}")
            raise ValueError(self.error)
        if record.state != "SNAPSHOT_READY":
            self._fail(f"duplicate PERSISTED_READY from rank {rank_id}")
            raise ValueError(self.error)
        record.data_digest = str(data_digest)
        record.state = "PERSISTED_READY"
        if all(item.state == "PERSISTED_READY" for item in self.ranks.values()):
            self.state = GlobalState.COMMITTING

    def commit(self) -> Mapping[str, object]:
        if self.state != GlobalState.COMMITTING:
            raise RuntimeError("metadata commit attempted before all data persisted")
        self.state = GlobalState.COMMITTED
        return self.metadata()

    def abort(self, reason: str) -> None:
        self._fail(reason)

    def metadata(self) -> Mapping[str, object]:
        return {
            "type": "MULTI_TRAINING_STATE_FULL",
            "schema_version": 1,
            "world_size": self.world_size,
            "state_step": self.step,
            "generation": self.generation,
            "state": self.state.value,
            "ranks": {
                str(rank): {
                    "manifest_digest": record.manifest_digest,
                    "data_digest": record.data_digest,
                    "state": record.state,
                }
                for rank, record in sorted(self.ranks.items())
            },
            "error": self.error,
        }

