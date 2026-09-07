"""MindSpore's native ``save_checkpoint`` FULL-state reference adapter."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np

from python.full_checkpoint_protocol import CheckpointState
from python.training_state import encode_control_value

from ..protocol import Handle
from ..state_bridge import (Snapshot, capture_snapshot, fsync_directory,
                            iter_unique_parameters, write_json)
from .base import Adapter


class MindSporeNativeSaveAdapter(Adapter):
    """Reference the actual MindSpore checkpoint serializer.

    The native API owns the device-to-host conversion and protobuf writer.  A
    small JSON sidecar carries the schema/control codec and is committed only
    after the native checkpoint has been flushed, so the restore gate remains
    comparable with the other FULL adapters.
    """

    name = "mindspore_native_save"
    kind = "framework-native-save"
    upstream_core_invoked = True
    mechanisms_preserved = (
        "MindSpore save_checkpoint parameter serialization",
        "MindSpore checkpoint protobuf file format",
        "synchronous native API completion before source release",
    )
    platform_substitutions = (
        "MindSpore native serializer writes the configured XFS filesystem",
        "controls are encoded as a uint8 tensor plus a durable sidecar",
    )

    @classmethod
    def preflight(cls, config):
        return {
            "adapter": cls.name,
            "kind": cls.kind,
            "status": "ready",
            "port_class": cls.kind,
            "upstream_core_invoked": True,
            "capture": "MindSpore save_checkpoint owns native parameter capture",
            "storage": "durable XFS file backend",
            "api": "mindspore.save_checkpoint(async_save=False)",
            "restore_api": "mindspore.load_checkpoint",
        }

    def _generation_dir(self, generation):
        return (Path(self.config["fs_test_dir"]) / "repro_checkpoints" /
                self.run_dir.name / f"generation_{int(generation):06d}")

    @staticmethod
    def _dtype(parameter):
        try:
            import mindspore as ms
            return np.dtype(ms.dtype_to_nptype(parameter.dtype))
        except Exception:
            return np.dtype(str(parameter.dtype).replace("Float", "float")
                            .replace("Int", "int").lower())

    def _save_entries(self, components, controls, ms):
        entries = []
        fields = []
        seen = set()
        for name, category, parameter, alias in iter_unique_parameters(components):
            if alias is not None or name in seen:
                continue
            seen.add(name)
            dtype = self._dtype(parameter)
            fields.append({
                "name": name,
                "category": category,
                "dtype": dtype.str,
                "shape": list(parameter.shape),
                "nbytes": int(parameter.size) * int(dtype.itemsize),
                "device_kind": "npu",
                "alias_group": name,
            })
            entries.append({"name": name, "data": parameter})
        payload, controls_metadata = encode_control_value(controls)
        entries.append({"name": "controls/payload",
                        "data": ms.Tensor(payload, dtype=ms.uint8)})
        fields.append({
            "name": "controls/payload",
            "category": "control",
            "dtype": np.dtype(np.uint8).str,
            "shape": list(payload.shape),
            "nbytes": int(payload.nbytes),
            "device_kind": "host",
            "alias_group": "controls/payload",
        })
        return entries, fields, payload, controls_metadata

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb", buffering=0) as stream:
            while True:
                block = stream.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _durable_file(path):
        with Path(path).open("rb", buffering=0) as stream:
            os.fsync(stream.fileno())

    def submit(self, generation, state_source, controls):
        import mindspore as ms

        components = state_source["components"]
        request_id = f"{self.name}-{int(generation):06d}"
        handle = Handle(self.name, generation, request_id, self.events)
        handle.admitted = True
        handle.mark("admitted")
        handle.transition(CheckpointState.SNAPSHOTTING)

        # Capture an oracle before handing parameters to the native serializer.
        # This is intentionally reported separately from save_checkpoint; it
        # verifies the native file did not silently omit optimizer/control state.
        capture_begin = time.monotonic_ns()
        expected = capture_snapshot(
            ms, components, components["optimizer"],
            int(state_source["step"]), int(self.config["seed"]))
        capture_end = time.monotonic_ns()
        entries, fields, payload, controls_metadata = self._save_entries(
            components, controls, ms)
        handle.transition(CheckpointState.SNAPSHOT_READY,
                          fields=len(fields), capture_ns=capture_end - capture_begin)
        handle.transition(CheckpointState.QUEUED)
        handle.transition(CheckpointState.DMA_COPYING,
                          backend="mindspore.save_checkpoint")
        handle.transition(CheckpointState.NVME_WRITING,
                          backend="mindspore.save_checkpoint")

        generation_dir = self._generation_dir(generation)
        generation_dir.mkdir(parents=True, exist_ok=True)
        target = generation_dir / "checkpoint.ckpt"
        api_begin = time.monotonic_ns()
        try:
            ms.save_checkpoint(entries, str(target), integrated_save=True,
                               async_save=False)
        except BaseException as error:
            handle.transition(CheckpointState.FAILED, error=repr(error))
            raise
        api_end = time.monotonic_ns()
        handle.mark("source_released", bytes=expected.total_bytes,
                    native_api_begin_ns=api_begin,
                    native_api_return_ns=api_end,
                    native_api_ns=api_end - api_begin)
        handle.mark("input_buffer_released")

        flush_begin = time.monotonic_ns()
        handle.transition(CheckpointState.FLUSHING,
                          native_api_ns=api_end - api_begin)
        self._durable_file(target)
        flush_end = time.monotonic_ns()
        handle.mark("data_completed", file_bytes=target.stat().st_size,
                    flush_ns=flush_end - flush_begin)

        handle.transition(CheckpointState.METADATA_COMMITTING)
        file_sha256 = self._file_sha256(target)
        metadata = {
            "format": "mindspore-native-save-v1",
            "adapter": self.name,
            "generation": int(generation),
            "step": int(state_source["step"]),
            "schema": {"schema_version": 1,
                        "format": "mindspore-native-save-v1",
                        "fields": sorted(fields, key=lambda item: item["name"])},
            "controls_metadata": controls_metadata,
            "sha256": expected.digest(),
            "file_sha256": file_sha256,
            "checkpoint_file": target.name,
            "native_api_ns": api_end - api_begin,
            "capture_ns": capture_end - capture_begin,
            "flush_ns": flush_end - flush_begin,
        }
        temp = generation_dir / "metadata.json.tmp"
        write_json(temp, metadata)
        with temp.open("rb", buffering=0) as stream:
            os.fsync(stream.fileno())
        os.replace(temp, generation_dir / "metadata.json")
        fsync_directory(generation_dir)
        handle.sha256 = metadata["sha256"]
        handle.transition(CheckpointState.PERSISTED,
                          sha256=metadata["sha256"],
                          file_sha256=file_sha256,
                          native_api_ns=api_end - api_begin,
                          capture_ns=capture_end - capture_begin,
                          flush_ns=flush_end - flush_begin)
        self.handles.append(handle)
        return handle

    def restore(self, generation, destination):
        import mindspore as ms

        generation_dir = self._generation_dir(generation)
        metadata = __import__("json").loads(
            (generation_dir / "metadata.json").read_text(encoding="utf-8"))
        if int(metadata.get("generation", -1)) != int(generation):
            raise ValueError("native checkpoint generation mismatch")
        target = generation_dir / metadata["checkpoint_file"]
        loaded = ms.load_checkpoint(str(target), net=None, strict_load=True)
        arrays = {}
        destination_components = {
            key: destination[key] for key in ("model", "optimizer")
            if key in destination
        }
        destinations = {
            name: parameter
            for name, _category, parameter, alias
            in iter_unique_parameters(destination_components)
            if alias is None
        }
        for field in metadata["schema"]["fields"]:
            name = field["name"]
            if name == "controls/payload":
                continue
            if name not in loaded or name not in destinations:
                raise ValueError(f"native checkpoint missing field: {name}")
            value = loaded[name].asnumpy()
            expected_dtype = np.dtype(field["dtype"])
            if value.dtype != expected_dtype or list(value.shape) != field["shape"]:
                raise ValueError(f"native checkpoint shape/dtype mismatch: {name}")
            arrays[name] = np.ascontiguousarray(value)
        if "controls/payload" not in loaded:
            raise ValueError("native checkpoint missing controls payload")
        payload = np.ascontiguousarray(loaded["controls/payload"].asnumpy(),
                                       dtype=np.uint8)
        snapshot = Snapshot(arrays, payload, metadata["controls_metadata"],
                            metadata["schema"])
        if snapshot.digest() != metadata["sha256"]:
            raise ValueError("native checkpoint state checksum mismatch")
        if self._file_sha256(target) != metadata["file_sha256"]:
            raise ValueError("native checkpoint file checksum mismatch")
        return snapshot
