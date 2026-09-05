"""Adapter for the project's real DirectCheckpoint FULL path."""

from __future__ import annotations

import time
from pathlib import Path

from .base import Adapter
from ..protocol import Handle
from ..protocol import BuildFailed, DependencyBlocked


class _DirectHandle:
    """Expose the common lifecycle while retaining the native handle fields."""

    def __init__(self, adapter, native, events):
        self.adapter = adapter
        self.native = native
        self.generation = int(native.generation)
        self.request_id = str(native.request_id)
        self.events = events
        self._source_released = False

    def wait_source_release(self, timeout):
        # DirectCheckpoint's frozen save only releases source ownership after
        # the request reaches its DMA-safe boundary.  The native API exposes
        # that boundary through its completion handle; waiting conservatively
        # to durable completion never publishes an unsafe source release.
        self.native.wait(timeout=float(timeout))
        if not self._source_released:
            self._source_released = True
            self.events.emit("source_released", self.generation,
                             self.request_id)
        return self

    def wait_persisted(self, timeout):
        self.native.wait(timeout=float(timeout))
        self.events.emit("persisted", self.generation, self.request_id,
                         sha256=self.native.snapshot_state_digest)
        return self

    def as_dict(self):
        row = dict(self.native.as_dict())
        row.update({
            "adapter": self.adapter.name,
            "request_id": self.request_id,
            "generation": self.generation,
            "source_released": self._source_released,
            "input_buffer_released": True,
            "data_completed": row.get("state") == "PERSISTED",
            "persisted": row.get("state") == "PERSISTED",
            "failed": row.get("status") == "FAILED",
            "sha256": row.get("snapshot_state_digest") or row.get("checksum"),
        })
        return row


class OursAdapter(Adapter):
    name = "ours"
    kind = "native"

    @classmethod
    def preflight(cls, config):
        if not config.get("raw_test_authorized"):
            return {"adapter": cls.name, "kind": cls.kind,
                    "status": "dependency_blocked",
                    "reason": "raw SPDK test is not authorized"}
        library = Path(config.get("project_root", ".")) / "build_out/lib/libnpu_nvme.so"
        if not library.exists():
            return {"adapter": cls.name, "kind": cls.kind,
                    "status": "build_failed",
                    "reason": f"missing DirectCheckpoint library: {library}"}
        return {"adapter": cls.name, "kind": cls.kind,
                "status": "ready", "raw_pci": config["raw_pci"],
                "implementation": "DirectCheckpoint FULL path"}

    def __init__(self, config, run_dir):
        super().__init__(config, run_dir)
        try:
            from direct_checkpoint import DirectCheckpoint
            self.ckpt = DirectCheckpoint(
                nvme_addr=config["raw_pci"],
                npu_device_id=int(config["npu_device"]),
                pipeline_depth=int(config.get("pipeline_depth", 4)),
                requested_chunk_size=int(config.get("chunk_bytes", 4 * 1024 * 1024)),
                spdk_shm_id=int(config.get("spdk_shm_id", 1)),
                keep_last_n=int(config.get("keep_last_n", 3)),
                slot_size_gb=int(config.get("slot_size_gb", 10)),
                checkpoint_slots=int(config.get("max_inflight", 2)),
                request_slots=int(config.get("max_inflight", 2)),
                admission="block")
        except PermissionError as error:
            raise DependencyBlocked(f"SPDK permission denied: {error}") from error
        except Exception as error:
            raise BuildFailed(f"DirectCheckpoint initialization failed: {error!r}") from error

    def submit(self, generation, state_source, controls):
        payload = state_source if isinstance(state_source, dict) else {}
        components = payload.get("components")
        control_state = payload.get("control_state") or controls
        if components is None:
            raise BuildFailed("native adapter requires live model/optimizer components")
        native = self.ckpt.save_state(
            components, control_state, step=int(payload["step"]),
            meta_path=str(self.run_dir / f"metadata_{int(generation):06d}.pkl"),
            io_mode="serial", timeout=float(self.config["timeout_seconds"]))
        handle = _DirectHandle(self, native, self.events)
        self.handles.append(handle)
        return handle

    def restore(self, generation, destination):
        if not destination:
            raise ValueError("native restore requires model and optimizer destination")
        controls = self.ckpt.load_state(
            {"model": destination["model"], "optimizer": destination["optimizer"]},
            step=destination.get("_step"))
        return controls

    def drain(self, timeout):
        for handle in self.handles:
            handle.wait_persisted(timeout)

    def close(self):
        if getattr(self, "ckpt", None) is not None:
            self.ckpt.close()
            self.ckpt = None
