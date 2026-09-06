"""Small adapter contract used by the common runner."""

from __future__ import annotations

import importlib.util
import subprocess
import os
from pathlib import Path

from ..host_bridge import SharedSnapshot
from ..protocol import AdapterError, DependencyBlocked, EventLog, Handle
from ..state_bridge import load_raw_snapshot, save_raw_snapshot


def adapter_status(name, config):
    return {"adapter": name, "status": "not_attempted", "reason": None}


class Adapter:
    name = "base"
    kind = "inspired"

    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = Path(run_dir)
        self.events = EventLog(self.run_dir / "events.jsonl")
        self.next_generation = 1
        self.handles = []

    @classmethod
    def preflight(cls, config):
        return {"adapter": cls.name, "kind": cls.kind,
                "status": "ready", "reason": None}

    def prepare(self, state_schema, config):
        return {"adapter": self.name, "kind": self.kind,
                "status": "ready", "state_fields": len(state_schema["fields"])}

    def submit(self, generation, state_source, controls):
        raise NotImplementedError

    def wait_source_release(self, handle, timeout):
        return handle.wait_source_release(timeout)

    def wait_persisted(self, handle, timeout):
        return handle.wait_persisted(timeout)

    def restore(self, generation, destination):
        raise NotImplementedError

    def drain(self, timeout):
        for handle in self.handles:
            self.wait_persisted(handle, timeout)

    def close(self):
        return None


class NoneAdapter(Adapter):
    """No-checkpoint training control used as the wall-clock baseline."""

    name = "none"
    kind = "training-reference"

    def submit(self, generation, state_source, controls):
        raise AdapterError("none adapter does not create checkpoint generations")

    def restore(self, generation, destination):
        raise AdapterError("none adapter has no persisted state")


class DurableFileAdapter(Adapter):
    """Reference backend for framework controls and the local implementation."""

    kind = "host-adapted"

    def _generation_dir(self, generation):
        # Payloads live on the explicitly configured test filesystem.  Keeping
        # events and manifests in the worktree while placing data on /models
        # avoids filling the nearly-full home volume and makes storage identity
        # visible in environment.json.
        root = Path(self.config.get("fs_test_dir") or self.run_dir / "checkpoints")
        return root / "repro_checkpoints" / self.run_dir.name / \
            f"generation_{int(generation):06d}"

    def submit(self, generation, state_source, controls):
        request_id = f"{self.name}-{int(generation):06d}"
        handle = Handle(self.name, generation, request_id, self.events)
        handle.admitted = True
        handle.mark("admitted")
        try:
            snapshot = state_source() if callable(state_source) else state_source["snapshot"]()
            handle.mark("source_released", bytes=snapshot.total_bytes)
            metadata = save_raw_snapshot(snapshot, self._generation_dir(generation))
            handle.mark("input_buffer_released")
            handle.mark("data_completed", sha256=metadata["sha256"])
            handle.mark("persisted", sha256=metadata["sha256"])
        except BaseException as error:
            handle.mark("failed", error=repr(error))
            raise
        self.handles.append(handle)
        return handle

    def restore(self, generation, destination):
        return load_raw_snapshot(self._generation_dir(generation))


class MechanismOnlyAdapter(DurableFileAdapter):
    """Explicit downgrade for CUDA-only baselines.

    Common NPU capture and durable restore are exercised, while the upstream
    CUDA writer/planner is not claimed to be present.
    """

    kind = "mechanism-only"
    degradation_reason = "upstream CUDA path unavailable on current platform"
    upstream_name = None

    @classmethod
    def preflight(cls, config):
        return {
            "adapter": cls.name, "kind": cls.kind, "status": "ready",
            "degradation": "mechanism-only", "upstream": cls.upstream_name,
            "reason": cls.degradation_reason,
            "capture": "common MindSpore NPU address capture",
            "writer": "common durable Host file writer",
        }

    def submit(self, generation, state_source, controls):
        snapshot = state_source() if callable(state_source) else state_source["snapshot"]()
        request_id = f"{self.name}-{int(generation):06d}"
        self.events.emit("degradation", int(generation), request_id,
                         level="mechanism-only", reason=self.degradation_reason)
        # Exercise the same explicit Host bridge expected by a CPU worker.  A
        # copy through shared memory is intentional and is accounted for as a
        # downgrade cost; it is not presented as zero-copy NPU persistence.
        from ..host_bridge import SharedSnapshot, snapshot_view_from_descriptor
        owner = None
        bridged = None
        try:
            self.events.emit("ipc_begin", int(generation), request_id,
                             bytes=snapshot.total_bytes)
            owner, descriptor = SharedSnapshot.from_snapshot(snapshot,
                                                               prefix=self.name)
            self.events.emit("ipc_ready", int(generation), request_id,
                             shm=descriptor["name"], bytes=descriptor["size"])
            # Keep the owner alive through the synchronous durable writer so
            # the bridged arrays are views, avoiding a third full-state copy.
            bridged = snapshot_view_from_descriptor(owner, descriptor)
            self.events.emit("ipc_receive", int(generation), request_id,
                             bytes=bridged.total_bytes)
            handle = super().submit(generation, {"snapshot": lambda: bridged}, controls)
            return handle
        finally:
            bridged = None
            if owner is not None:
                owner.close(unlink=True)
                self.events.emit("ipc_released", int(generation), request_id)


def require_path(path, label):
    if not path:
        raise DependencyBlocked(f"{label} is not configured")
    if not Path(path).exists():
        raise DependencyBlocked(f"{label} does not exist: {path}")


def require_python_module(python, module):
    if not python:
        raise DependencyBlocked(f"worker Python is not configured for {module}")
    probe = Path(python)
    if not probe.exists():
        raise DependencyBlocked(f"worker Python does not exist: {python}")


def probe_worker(python, code, timeout=30):
    """Run a side-effect-free import/build probe in a locked worker venv."""
    if not python:
        return {"status": "missing", "reason": "worker Python is not configured"}
    try:
        proc = subprocess.run([str(python), "-c", code], capture_output=True,
                              text=True, check=False, timeout=timeout)
    except Exception as error:
        return {"status": "error", "reason": repr(error)}
    result = {"status": "ready" if proc.returncode == 0 else "failed",
              "returncode": proc.returncode}
    if proc.stdout.strip():
        result["stdout"] = proc.stdout.strip()[-4000:]
    if proc.stderr.strip():
        result["stderr"] = proc.stderr.strip()[-4000:]
    return result
