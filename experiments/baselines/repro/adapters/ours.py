"""Adapter for the project's real StrictCheckpoint FULL path."""

from __future__ import annotations

import time
from pathlib import Path

from .base import Adapter
from ..protocol import Handle
from ..protocol import BuildFailed, DependencyBlocked


class _DirectHandle:
    """Expose the common lifecycle while retaining the native handle fields."""

    def __init__(self, adapter, native, events, expected_state_digest=None):
        self.adapter = adapter
        self.native = native
        self.generation = int(native.generation)
        self.request_id = str(native.request_id)
        self.events = events
        self.expected_state_digest = expected_state_digest
        self.expected_spec = None
        self._source_released = False

    def wait_source_release(self, timeout):
        # StrictCheckpoint's frozen save only releases source ownership after
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
        events = list(row.get("events", []))
        # Keep the native event spelling, but also provide the normalized
        # terminal timestamp consumed by the common runner and summaries.
        persisted_ns = next(
            (event.get("monotonic_ns") for event in events
             if event.get("state") == "PERSISTED"), None)
        row.update({
            "expected_spec": self.expected_spec,
            "adapter": self.adapter.name,
            "request_id": self.request_id,
            "generation": self.generation,
            "source_released": self._source_released,
            "input_buffer_released": True,
            "data_completed": row.get("state") == "PERSISTED",
            "persisted": row.get("state") == "PERSISTED",
            "failed": row.get("status") == "FAILED",
            "sha256": (self.expected_state_digest or
                       row.get("snapshot_state_digest")),
            "persisted_ns": persisted_ns,
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
                    "reason": f"missing StrictCheckpoint library: {library}"}
        from npu_nvme.storage.bindings import load_backend
        load_backend(library)
        return {"adapter": cls.name, "kind": cls.kind,
                "status": "ready", "raw_pci": config["raw_pci"],
                "implementation": "StrictCheckpoint FULL path"}

    @classmethod
    def runtime_probe(cls, config):
        """Attempt the real SPDK/ACL attach without submitting a checkpoint."""
        try:
            probe = cls(config, Path(config.get("results_root", ".")) / "runtime_probe")
            probe.close()
            return {"adapter": cls.name, "status": "ready",
                    "runtime_attach": "passed", "raw_pci": config["raw_pci"]}
        except DependencyBlocked as error:
            return {"adapter": cls.name, "status": "dependency_blocked",
                    "runtime_attach": "failed", "reason": repr(error)}
        except BuildFailed as error:
            return {"adapter": cls.name, "status": "runtime_attach_failed",
                    "runtime_attach": "failed", "reason": repr(error)}

    def __init__(self, config, run_dir):
        super().__init__(config, run_dir)
        try:
            from npu_nvme import StrictCheckpoint
            from npu_nvme.storage.bindings import load_backend
            self.ckpt = StrictCheckpoint(nvme_addr=config['raw_pci'], npu_device_id=int(config['npu_device']), pipeline_depth=int(config.get('pipeline_depth', 4)), requested_chunk_size=int(config['chunk_bytes']), spdk_shm_id=int(config.get('spdk_shm_id', 1)), slot_size_gb=int(config.get('slot_size_gb', 10)), admission='block', backend=load_backend(Path(config['project_root']) / 'build_out/lib/libnpu_nvme.so'))
        except PermissionError as error:
            raise DependencyBlocked(f"SPDK permission denied: {error}") from error
        except Exception as error:
            raise BuildFailed(f"StrictCheckpoint initialization failed: {error!r}") from error

    def submit(self, generation, state_source, controls):
        payload = state_source if isinstance(state_source, dict) else {}
        components = payload.get("components")
        control_state = payload.get("control_state") or controls
        if components is None:
            raise BuildFailed("native adapter requires live model/optimizer components")
        snapshot_factory = payload.get("snapshot")
        expected_state_digest = (snapshot_factory().digest()
                                 if snapshot_factory is not None else None)
        from npu_nvme.framework.full_state import training_spec
        from npu_nvme.storage.bindings import load_backend
        import mindspore as ms
        spec = training_spec(components, control_state, self.training_identity(self.config, ms),
            framework=ms, acl=load_backend().acl_lib, npu=int(self.config['npu_device']))
        native = self.ckpt.save_state(components, control_state, step=int(payload['step']), expected_spec=spec, timeout=float(self.config['timeout_seconds']))
        handle = _DirectHandle(
            self, native, self.events,
            expected_state_digest=expected_state_digest)
        handle.expected_spec = spec
        self.handles.append(handle)
        return handle

    @staticmethod
    def training_identity(config, ms):
        return dict(workload=config['model'], seq_len=int(config['input_tokens']),
                    dropout=float(config['dropout']), optimizer='AdamWeightDecay', loss_scale=1.0,
                    framework=ms.__version__, world_size=1, deterministic='ON')

    def restore(self, generation, destination):
        if not destination or not callable(destination.get('target_factory')) or 'expected_spec' not in destination:
            raise ValueError('strict native restore requires target_factory and expected_spec')
        step = int(destination['_step'])
        record = self.ckpt.meta_dict.get('checkpoints', {}).get(f'generation_{int(generation)}')
        if record is None or record['state_step'] != step:
            raise ValueError('requested generation/step is not retained')
        return self.ckpt.restore_full_state(destination['target_factory'], destination['expected_spec'],
            step=step, deadline=time.monotonic()+float(self.config['timeout_seconds']))

    def drain(self, timeout):
        for handle in self.handles:
            handle.wait_persisted(timeout)

    def close(self):
        if getattr(self, "ckpt", None) is not None:
            self.ckpt.close()
            self.ckpt = None
