"""Shared ACL capture engine for NPU semantic ports."""

from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor

from python.full_checkpoint_protocol import CheckpointState

from ..acl_capture import allocate_slot, capture_to_slot, device_state_layout
from ..protocol import AdapterError, Handle
from ..state_bridge import load_raw_snapshot, save_raw_snapshot
from .base import Adapter


class ACLSemanticAdapter(Adapter):
    kind = "npu-semantic-port"
    upstream_core_invoked = False
    mechanisms_preserved = ()
    platform_substitutions = ()
    configured_write_chunks = False

    def __init__(self, config, run_dir):
        super().__init__(config, run_dir)
        self.slot_count = int(config.get("max_inflight", 2))
        self._slots = None
        self._free_slots = queue.Queue(maxsize=self.slot_count)
        self._executor = ThreadPoolExecutor(
            max_workers=self.slot_count, thread_name_prefix=self.name)
        self._futures = []
        self._layout = None
        self._host_layout = None
        self._total_device_bytes = 0
        self._closed = False

    @classmethod
    def preflight(cls, config):
        return {
            "adapter": cls.name, "kind": cls.kind, "status": "ready",
            "port_class": cls.kind,
            "upstream_core_invoked": cls.upstream_core_invoked,
            "mechanisms_preserved": list(cls.mechanisms_preserved),
            "platform_substitutions": list(cls.platform_substitutions),
            "capture": "real _data_ptr -> aclrtMemcpy -> ACL pinned Host",
            "storage": "durable XFS file backend",
        }

    def prepare_components(self, components):
        if self._slots is not None:
            return
        (self._layout, self._host_layout,
         self._total_device_bytes) = device_state_layout(components)
        self._slots = [allocate_slot(i, self._total_device_bytes)
                       for i in range(self.slot_count)]
        for slot in self._slots:
            self._free_slots.put(slot)

    def _generation_dir(self, generation, slot_id):
        root = self.config["fs_test_dir"]
        from pathlib import Path
        return Path(root) / "repro_checkpoints" / self.run_dir.name / \
            f"generation_{int(generation):06d}"

    def _persist(self, handle, snapshot, slot):
        try:
            chunk_bytes = (int(self.config["chunk_bytes"])
                           if self.configured_write_chunks else None)
            detail = {"slot_id": slot.slot_id}
            if chunk_bytes:
                detail["chunk_bytes"] = chunk_bytes
            handle.transition(CheckpointState.NVME_WRITING, **detail)

            def phase(name):
                if name == "flush_begin":
                    handle.transition(CheckpointState.FLUSHING)
                elif name == "metadata_begin":
                    handle.transition(CheckpointState.METADATA_COMMITTING)

            metadata = save_raw_snapshot(
                snapshot, self._generation_dir(handle.generation, slot.slot_id),
                phase_callback=phase,
                extra_metadata={"generation": handle.generation,
                                "slot_id": slot.slot_id,
                                "port_class": self.kind,
                                "adapter": self.name},
                chunk_bytes=chunk_bytes)
            handle.sha256 = metadata["sha256"]
            handle.mark("input_buffer_released")
            handle.mark("data_completed", sha256=metadata["sha256"])
            handle.transition(CheckpointState.PERSISTED,
                              sha256=metadata["sha256"])
        except BaseException as error:
            if handle.state not in (CheckpointState.FAILED,
                                    CheckpointState.CANCELLED,
                                    CheckpointState.TIMED_OUT):
                handle.transition(CheckpointState.FAILED, error=repr(error))
        finally:
            self._free_slots.put(slot)

    def submit(self, generation, state_source, controls):
        if not isinstance(state_source, dict) or not state_source.get("components"):
            raise AdapterError("ACL semantic port requires live model/optimizer components")
        self.prepare_components(state_source["components"])
        request_id = f"{self.name}-{int(generation):06d}"
        handle = Handle(self.name, generation, request_id, self.events)
        handle.mark("admitted")
        handle.transition(CheckpointState.SNAPSHOTTING)
        handle.transition(CheckpointState.SNAPSHOT_READY,
                          fields=len(self._layout))
        wait_begin = time.monotonic_ns()
        try:
            slot = self._free_slots.get(
                timeout=float(self.config["timeout_seconds"]))
        except queue.Empty as error:
            handle.transition(
                CheckpointState.TIMED_OUT,
                error="timed out waiting for a free ACL pinned slot",
                slot_wait_ns=time.monotonic_ns() - wait_begin)
            raise TimeoutError(handle.error) from error
        handle.transition(CheckpointState.QUEUED, slot_id=slot.slot_id,
                          slot_wait_ns=time.monotonic_ns() - wait_begin)
        handle.transition(CheckpointState.DMA_COPYING, slot_id=slot.slot_id)
        try:
            snapshot, timing = capture_to_slot(
                self._layout, self._host_layout, slot, controls,
                self.config["npu_device"])
        except BaseException as error:
            self._free_slots.put(slot)
            handle.transition(CheckpointState.FAILED, error=repr(error))
            raise
        handle.mark("source_released", bytes=snapshot.total_bytes, **timing)
        future = self._executor.submit(self._persist, handle, snapshot, slot)
        self._futures.append(future)
        self.handles.append(handle)
        return handle

    def restore(self, generation, destination):
        candidates = [self._generation_dir(generation, slot_id)
                      for slot_id in range(self.slot_count)]
        for path in candidates:
            try:
                return load_raw_snapshot(path, expected_generation=generation)
            except (FileNotFoundError, ValueError):
                continue
        raise FileNotFoundError(
            f"generation {generation} not present in any {self.name} slot")

    def drain(self, timeout):
        deadline = time.monotonic() + float(timeout)
        for handle in self.handles:
            handle.wait_persisted(max(0.0, deadline - time.monotonic()))
        for future in self._futures:
            future.result(timeout=max(0.0, deadline - time.monotonic()))

    def close(self):
        if self._closed:
            return
        self.drain(self.config["timeout_seconds"])
        self._executor.shutdown(wait=True)
        if self._slots:
            for slot in self._slots:
                slot.close()
        self._closed = True
