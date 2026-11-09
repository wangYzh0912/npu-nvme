"""Persistent ByteCheckpoint planner/store/load worker."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("BYTECHECKPOINT_ENABLE_PINNED_MEM_D2H", "0")
os.environ.setdefault("BYTECHECKPOINT_ENABLE_TREE_TOPO", "0")

import torch
import torch.distributed as dist

from bytecheckpoint.engine import _load_engine, _store_engine
from bytecheckpoint.distributed.rpc_context import setup_rpc_service
from bytecheckpoint.planner.ddp.ddp_planner import DDPLoadPlanner, DDPSavePlanner
from bytecheckpoint.storage import CKPTCounter
from bytecheckpoint.storage._storage import (_local_storage_reader,
                                             _local_storage_writer)
from bytecheckpoint.workflow.state_dict.load_state_dict import load_state_dict
from bytecheckpoint.workflow.state_dict.save_state_dict import save_state_dict
from bytecheckpoint.workflow.extra_state import load_extra_state, save_extra_state

from .common import attach_arrays, canonical_digest, durable_tree, serve


class Worker:
    def __init__(self):
        self._rendezvous = tempfile.NamedTemporaryFile(prefix="bc_gloo_", delete=False)
        self._rendezvous.close()
        dist.init_process_group(
            "gloo", init_method=f"file://{self._rendezvous.name}", rank=0,
            world_size=1)
        # ByteCheckpoint's LocalStorageWriter uses its paper-protocol gRPC
        # barrier even for world_size=1.  Start that coordinator explicitly in
        # the isolated worker rather than allowing a silent wait forever.
        setup_rpc_service()

    @staticmethod
    def _components(arrays):
        state = {name: torch.from_numpy(array)
                 for name, array in arrays.items()}
        return {
            "model": {name: value for name, value in state.items()
                      if name.startswith("model/")},
            "optimizer": {name: value for name, value in state.items()
                          if name.startswith("optimizer/")},
            "extra_state": {"controls/state": state["controls/state"]},
        }

    def execute(self, request):
        if request["operation"] == "save":
            return self.save(request)
        if request["operation"] == "restore":
            return self.restore(request)
        raise ValueError(f"unknown operation {request['operation']}")

    def save(self, request):
        checkpoint_dir = Path(request["checkpoint_dir"])
        state_paths = {
            component: checkpoint_dir / component
            for component in ("model", "optimizer", "extra_state")
        }
        for state_path in state_paths.values():
            state_path.mkdir(parents=True, exist_ok=True)
        shm, arrays = attach_arrays(request["descriptor"])
        try:
            expected = canonical_digest(arrays)
            components = self._components(arrays)
            counter = CKPTCounter(3, True)
            begin = time.monotonic_ns()
            save_started = time.time()
            for component in ("model", "optimizer"):
                save_state_dict(
                    state_dict=components[component],
                    state_path=state_paths[component],
                    root_path=str(checkpoint_dir), ckpt_name=component,
                    framework_name="ddp", suffix=None,
                    planner=DDPSavePlanner(),
                    save_ckpt_start_time=save_started,
                    ckpt_counter=counter,
                    global_steps=int(request["generation"]),
                    callback=None, no_dist=False, async_io=True)
            save_extra_state(
                file_name="extra_state_rank_0.pt",
                state_path=str(state_paths["extra_state"]),
                root_path=str(checkpoint_dir),
                state_dict=components["extra_state"],
                ckpt_name="extra_state", framework_name="ddp", suffix=None,
                save_ckpt_start_time=save_started, ckpt_counter=counter,
                global_steps=int(request["generation"]), callback=None,
                async_io=True)
            for component in ("model", "optimizer", "extra_state"):
                queue = _local_storage_writer.get_sync_queue(
                    component, "ddp", None)
                if not queue.join(timeout=float(request["timeout_seconds"])):
                    raise TimeoutError(
                        f"ByteCheckpoint {component} queue did not drain")
            data_done = time.monotonic_ns()
            durable_tree(checkpoint_dir)
            durable = time.monotonic_ns()
            return {
                "sha256": expected, "worker_begin_ns": begin,
                "data_done_ns": data_done, "durable_ns": durable,
                "upstream_hook": (
                    "model+optimizer DDPSavePlanner; extra_state workflow; "
                    "shared CKPTCounter(3); LocalStorageWriter"),
                "components": ["model", "optimizer", "extra_state"],
                "torch_version": torch.__version__,
            }
        finally:
            shm.close()

    def restore(self, request):
        checkpoint_dir = Path(request["checkpoint_dir"])
        shm, arrays = attach_arrays(request["descriptor"])
        try:
            components = self._components(arrays)
            for component in ("optimizer", "model"):
                futures = load_state_dict(
                    state_dict=components[component],
                    path=str(checkpoint_dir / component),
                    ckpt_name=component, framework_name="ddp", suffix=None,
                    planner=DDPLoadPlanner(strict=True), no_dist=False,
                    fast_loading=False)
                for future in futures:
                    future.result(timeout=float(request["timeout_seconds"]))
            extra_futures = load_extra_state(
                file_name="extra_state_rank_0.pt",
                path=str(checkpoint_dir / "extra_state"),
                ckpt_name="extra_state", framework_name="ddp", suffix=None,
                async_io=True)
            extra = extra_futures[0].result(
                timeout=float(request["timeout_seconds"]))
            components["extra_state"]["controls/state"].copy_(
                extra["controls/state"])
            return {
                "sha256": canonical_digest(arrays),
                "upstream_hook": (
                    "optimizer+model DDPLoadPlanner; extra_state workflow"),
            }
        finally:
            shm.close()

    def close(self):
        _store_engine.cleanup_resources()
        _load_engine.cleanup_resources()
        _local_storage_writer.cleanup_resources()
        _local_storage_reader.cleanup_resources()
        if dist.is_initialized():
            dist.destroy_process_group()
        try:
            os.unlink(self._rendezvous.name)
        except FileNotFoundError:
            pass
        return {}


if __name__ == "__main__":
    serve(Worker())
