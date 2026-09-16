"""Legacy V2 catalog owner; D1 publication/reader-pin contracts are not enabled yet.

The storage adapter is explicitly supplied. Full, delta and compatibility tools
retain their old V2 behavior during B migration; D1 must close those write paths.
"""
import copy
import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetadataState:
    layout: object = None
    meta_dict: dict = field(default_factory=lambda: {"checkpoints": {}})
    metadata_generation: int = 0
    active_meta_slot: int = 0
    stack_start_bytes: int = 0


@dataclass(frozen=True)
class LegacyCommitSpec:
    rank_id: int
    world_size: int
    chunk_size: int
    keep_last_n: int
    sidecar_path: str


class LegacyCommitCoordinator:
    def __init__(self, metadata_io, state=None):
        self.io = metadata_io
        self.state = MetadataState() if state is None else state

    def mount(self, total_bytes, rank_id):
        self.io.mount(self.state, total_bytes, rank_id)

    def persist(self, generation=None):
        self.io.persist(self.state, generation)

    def publish_full(self, spec, step: int, layout: List[Dict],
                     checkpoint_meta: Dict = None):
        if spec.rank_id != 0:
            return

        if self.state.layout is None:
            raise RuntimeError("cannot commit metadata before mounting layout")
        previous_meta = copy.deepcopy(self.state.meta_dict)
        next_generation = self.state.metadata_generation + 1
        ckpt_key = f"step_{step}"
        param_records = {}
        for p in layout:
            record = {
                "offset": p["offset"], "size": p["size"],
                "shape": p["shape"], "dtype": p["dtype"],
            }
            # Keep the on-disk record compact.  The namespace in the key
            # already identifies model/optimizer/control and source_name is
            # the suffix after the first slash; placement is known from the
            # target object at load time.
            for field in ("sha256", "codec"):
                if field in p:
                    record[field] = p[field]
            param_records[p["name"]] = record

        checkpoint_record = {
            "type": "FULL",
            "generation": next_generation,
            "chunk_size": spec.chunk_size,
            "rank_id": spec.rank_id,
            "world_size": spec.world_size,
            "params": param_records,
        }
        if checkpoint_meta:
            checkpoint_record.update(checkpoint_meta)

        # A FULL slot is selected by step modulo keep_last_n.  Once its new
        # payload is durable, no older record that points at the same physical
        # slot may remain visible under a different step/generation.
        target_slot = int(step) % int(spec.keep_last_n)
        for old_key, old_record in list(
                self.state.meta_dict.get("checkpoints", {}).items()):
            if not old_key.startswith("step_"):
                continue
            try:
                old_step = int(old_key.split("_", 1)[1])
            except ValueError:
                continue
            if (old_step != int(step) and
                    old_step % int(spec.keep_last_n) == target_slot and
                    int(old_record.get("rank_id", spec.rank_id)) == spec.rank_id):
                del self.state.meta_dict["checkpoints"][old_key]
        self.state.meta_dict["checkpoints"][ckpt_key] = checkpoint_record

        saved_records = []
        for k, record in self.state.meta_dict["checkpoints"].items():
            if k.startswith("step_"):
                try:
                    saved_records.append((int(record.get("generation", 0)),
                                          int(k.split('_')[1])))
                except ValueError:
                    pass
        saved_records.sort()
        saved_steps = [step_id for _generation, step_id in saved_records]

        delta_keys = []
        for k in self.state.meta_dict.get("delta_chain", {}).keys():
            if k.startswith("step_"):
                try:
                    delta_keys.append(int(k.split('_')[1]))
                except ValueError:
                    pass
        for ds in delta_keys:
            if saved_steps and ds < saved_steps[0]:
                del self.state.meta_dict["delta_chain"][f"step_{ds}"]

        while len(saved_records) > spec.keep_last_n:
            _oldest_generation, oldest_step = saved_records.pop(0)
            old_key = f"step_{oldest_step}"
            if old_key in self.state.meta_dict.get("checkpoints", {}):
                del self.state.meta_dict["checkpoints"][old_key]
            if old_key in self.state.meta_dict.get("delta_chain", {}):
                del self.state.meta_dict["delta_chain"][old_key]

        import pickle as _pickle
        os.makedirs(os.path.dirname(spec.sidecar_path), exist_ok=True)
        def _dump_meta_pkl():
            with open(spec.sidecar_path, "wb") as _f:
                _pickle.dump(self.state.meta_dict, _f)
        self._dump_meta_pkl = _dump_meta_pkl

        try:
            self.persist(next_generation)
        except BaseException:
            # Data may be durable, but without the superblock commit point the
            # generation is not visible.  Roll back the in-memory/sidecar
            # ledger so an explicit queue reset cannot publish it later.
            self.state.meta_dict = previous_meta
            with open(spec.sidecar_path, "wb") as stream:
                _pickle.dump(self.state.meta_dict, stream)
            raise
        self._dump_meta_pkl()
        print(f"[DirectCkpt] Rank 0 Meta committed safely to "
              f"Slot {'B' if self.state.active_meta_slot == 1 else 'A'} "
              "(Superblock updated).",
              flush=True)

    def select(self, step, *, total_bytes, rank_id):
        self.mount(total_bytes, rank_id)
        if step is None:
            candidates = []
            for key, record in self.state.meta_dict.get("checkpoints", {}).items():
                if not key.startswith("step_"):
                    continue
                try:
                    candidates.append((int(key.split("_", 1)[1]), record))
                except ValueError:
                    continue
            if not candidates:
                raise FileNotFoundError("No checkpoints found in metadata")
            return max(candidates, key=lambda item: item[0])
        key = f"step_{int(step)}"
        try:
            return int(step), self.state.meta_dict["checkpoints"][key]
        except KeyError as error:
            raise FileNotFoundError(f"Checkpoint for {key} not found") from error
