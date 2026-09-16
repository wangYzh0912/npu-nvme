"""Canonical checkpoint facade. Import is inert; construction opens dependencies.

Batch B preserves the legacy media and public behavior. Strict FULL semantics
are introduced by the independently validated D1 change.
"""
from __future__ import annotations
import ctypes
import hashlib
import math
import os
from pathlib import Path
import time
import threading
import atexit
from typing import List, Dict, TYPE_CHECKING
if TYPE_CHECKING:
    import mindspore as ms
    from mindspore import Tensor
from npu_nvme.storage.bindings import (NPUNVMEContext, NPUNVMERequest,
    NPUNVMEStats, NPUNVMERetainedSlot, load_backend)
from npu_nvme.storage.layout import DiskLayout
from npu_nvme.storage.chunks import (build_chunks, build_ctypes_arrays,
    validate_descriptors)
from npu_nvme.types import CheckpointState
from npu_nvme.storage.transport import LegacyBatchTransport
from npu_nvme.storage.metadata import MetadataIO
from npu_nvme.runtime.restore import LegacyRestore
from npu_nvme.runtime.d1_restore import StrictRestoreSession
from npu_nvme.runtime.commit import LegacyCommitCoordinator, LegacyCommitSpec
from npu_nvme.runtime.scheduler import (
    CheckpointScheduler, CheckpointBusyError, CheckpointQueuePoisonedError)
from npu_nvme.runtime.lifecycle import cleanup_legacy
from npu_nvme.runtime.handle import CheckpointHandle, HandleServices
from npu_nvme.runtime.worker import WorkerServices, TransferJob, run_legacy_transfer
from npu_nvme.framework.parameters import get_dev_ptr

# Strong owners survive cyclic GC while native DMA may still reference them.
_QUARANTINED_CONTEXTS = set()


def __getattr__(name):
    # Historical reexports are resolved only when explicitly requested. They
    # are never used as an internal service registry by the components.
    import importlib
    modules = {
        'ms': ('mindspore', None), 'np': ('numpy', None),
        'lib': ('c_bindings', 'lib'), 'acl_lib': ('c_bindings', 'acl_lib'),
        'ProbeTrainOneStepCell': ('npu_nvme.framework.cells', 'ProbeTrainOneStepCell'),
        'NoOpInitializer': ('noop_init', 'NoOpInitializer'),
        'replace_with_noop_initializer': ('noop_init', 'replace_with_noop_initializer'),
    }
    if name in modules:
        module, attribute = modules[name]
        loaded = importlib.import_module(module)
        return loaded if attribute is None else getattr(loaded, attribute)
    pure_exports = {'ops': ('mindspore', 'ops'), 'nn': ('mindspore', 'nn'), 'Tensor': ('mindspore', 'Tensor'), 'SUPERBLOCK_OFFSET': ('disk_layout', 'SUPERBLOCK_OFFSET'), 'SUPERBLOCK_HEADER_BYTES': ('disk_layout', 'SUPERBLOCK_HEADER_BYTES'), 'META_SLOT_A_OFFSET': ('disk_layout', 'META_SLOT_A_OFFSET'), 'META_SLOT_B_OFFSET': ('disk_layout', 'META_SLOT_B_OFFSET'), 'META_SLOT_BYTES': ('disk_layout', 'META_SLOT_BYTES'), 'MAGIC_NUMBER': ('disk_layout', 'MAGIC_NUMBER'), 'UINT32_BYTES': ('disk_layout', 'UINT32_BYTES'), 'DELTA_MAGIC': ('disk_layout', 'DELTA_MAGIC'), 'FRAME_HEADER_SIZE': ('disk_layout', 'FRAME_HEADER_SIZE'), 'BLOCK_SIZE': ('disk_layout', 'BLOCK_SIZE'), 'DiskLayout': ('disk_layout', 'DiskLayout'), 'make_layout': ('disk_layout', 'make_layout'), 'pack_metadata': ('disk_layout', 'pack_metadata'), 'unpack_metadata': ('disk_layout', 'unpack_metadata'), 'pack_superblock': ('disk_layout', 'pack_superblock'), 'unpack_superblock': ('disk_layout', 'unpack_superblock'), 'build_chunks': ('chunk_helpers', 'build_chunks'), 'build_chunks_host': ('chunk_helpers', 'build_chunks_host'), 'build_ctypes_arrays': ('chunk_helpers', 'build_ctypes_arrays'), 'rebuild_chunks_from_meta': ('chunk_helpers', 'rebuild_chunks_from_meta'), 'validate_descriptors': ('chunk_helpers', 'validate_descriptors'), 'pack_delta_frame': ('delta_protocol', 'pack_delta_frame'), 'pack_lossless_delta_frame': ('delta_protocol', 'pack_lossless_delta_frame'), 'pack_s2_replacement_frame': ('delta_protocol', 'pack_s2_replacement_frame'), 'unpack_delta_frame': ('delta_protocol', 'unpack_delta_frame'), 'unpack_delta_frame_with_meta': ('delta_protocol', 'unpack_delta_frame_with_meta'), 'apply_delta_patches': ('delta_protocol', 'apply_delta_patches'), 'FileDeltaWriter': ('delta_protocol', 'FileDeltaWriter'), 'TRAINING_STATE_SCHEMA_VERSION': ('training_state', 'TRAINING_STATE_SCHEMA_VERSION'), 'decode_control_value': ('training_state', 'decode_control_value'), 'encode_control_value': ('training_state', 'encode_control_value'), 'validate_state_names': ('training_state', 'validate_state_names'), 'CheckpointState': ('full_checkpoint_protocol', 'CheckpointState'), 'TERMINAL_STATES': ('full_checkpoint_protocol', 'TERMINAL_STATES'), 'require_transition': ('full_checkpoint_protocol', 'require_transition')}
    if name in pure_exports:
        module, attribute = pure_exports[name]
        return getattr(importlib.import_module(module), attribute)
    raise AttributeError(name)


# -- DirectCheckpoint: NVMe-backed training checkpoint manager ----------------

class DirectCheckpoint:
    _ms_warmed_up = False

    @staticmethod
    def live_async_capability():
        acl_lib = load_backend().acl_lib
        required = ("aclrtMallocHost", "aclrtMemcpyAsync", "aclrtCreateEvent",
                    "aclrtRecordEvent", "aclrtStreamWaitEvent")
        missing = [name for name in required
                   if acl_lib is None or not hasattr(acl_lib, name)]
        return {
            "supported": not missing,
            "code": "SUPPORTED_GENERATION_PINNED_STAGING" if not missing else
                    "MISSING_ACL_LIVE_FENCE_API",
            "detail": ("generation-owned pinned staging and aggregate D2H event"
                       if not missing else f"missing ACL symbols: {missing}"),
            "required": "split forward/backward and optimizer launch",
        }

    def __init__(
        self, nvme_addr: str = "0000:83:00.0", npu_device_id: int = 0,
        pipeline_depth: int = 4, requested_chunk_size: int = 4 * 1024 * 1024,
        enable_profiling: bool = False, profiling_dir: str = "./output/profiling",
        rank_id: int = 0, world_size: int = 1,
        base_offset_bytes: int = 0, shard_span_bytes: int = None,
        spdk_shm_id: int = 1, keep_last_n: int = 3, slot_size_gb: int = 10,
        warmup_fn: callable = None, checkpoint_slots: int = 1,
        admission: str = "block", request_slots: int = None,
        backend=None, framework=None,
    ):
        if framework is None:
            import mindspore as framework
        if backend is None:
            backend = load_backend()
        self._framework = framework
        self._ops = framework.ops
        self._binding = backend.lib
        self._acl = backend.acl_lib
        from npu_nvme.framework.capture import FrozenCapture
        from npu_nvme.framework.live import LiveCapture
        from npu_nvme.framework.weights import LegacyWeightsRestore
        from npu_nvme.experimental.legacy import LegacyExperimental
        self.ctx = ctypes.POINTER(NPUNVMEContext)()
        self._metadata_io = MetadataIO(self._binding, self.ctx)
        self._commit = LegacyCommitCoordinator(self._metadata_io)
        self.npu_device_id = npu_device_id
        self._capture = FrozenCapture(rank_id=rank_id, device_id=npu_device_id,
            framework=self._framework, acl=self._acl, pointer_of=get_dev_ptr)
        self.enable_profiling = enable_profiling
        self.profiling_dir = profiling_dir
        self.rank_id = rank_id
        self.world_size = world_size
        self.base_offset_bytes = base_offset_bytes
        self.shard_span_bytes = shard_span_bytes
        os.environ.setdefault("SPDK_SHM_ID", str(spdk_shm_id))

        self.keep_last_n = keep_last_n
        if not 0 < int(checkpoint_slots) <= 16:
            raise ValueError("checkpoint_slots must be in [1, 16]")
        if request_slots is None:
            request_slots = checkpoint_slots
        if not 0 < int(request_slots) <= 64:
            raise ValueError("request_slots must be in [1, 64]")
        if admission not in ("block", "try"):
            raise ValueError("admission must be block or try")
        self.checkpoint_slots = int(checkpoint_slots)
        self.request_slots = int(request_slots)
        self.admission = admission
        self._scheduler = CheckpointScheduler(self.checkpoint_slots, self.request_slots)
        self._metadata_lock = threading.Lock()
        self._io_mutex = threading.Lock()
        self._sequence_lock = threading.Lock()
        self.slot_bytes = slot_size_gb * 1024**3
        self.active_meta_slot = 0
        self.metadata_generation = 0
        self.layout = None
        self.stack_start_bytes = 0
        self.meta_dict = {"checkpoints": {}}
        self.last_layout = []
        self._meta_pkl = os.path.join(
            str(Path(__file__).resolve().parents[2]),
            "experiments", "output", "checkpoint_meta.pkl")

        if self.enable_profiling:
            os.makedirs(self.profiling_dir, exist_ok=True)

        if warmup_fn is not None and not DirectCheckpoint._ms_warmed_up:
            print("[DirectCheckpoint] Running MS runtime warmup before SPDK init...",
                  flush=True)
            warmup_fn()
            DirectCheckpoint._ms_warmed_up = True
            print("[DirectCheckpoint] MS runtime warmup complete.", flush=True)

        print(f"[DirectCheckpoint] loading so from {backend.library_path}")

        rc = self._binding.npu_nvme_init(
            ctypes.byref(self.ctx),
            nvme_addr.encode(),
            npu_device_id,
            pipeline_depth,
            requested_chunk_size,
            enable_profiling,
            self.profiling_dir.encode("utf-8"),
        )
        if rc != 0:
            raise RuntimeError("npu_nvme_init failed")

        self.total_bytes = self._binding.npu_nvme_get_total_blocks(self.ctx)
        if self.total_bytes == 0:
            raise RuntimeError("Failed to get NVMe total bytes from hardware.")

        self.chunk_size = requested_chunk_size
        effective = self._binding.npu_nvme_get_max_transfer(self.ctx)
        print(f"[DirectCheckpoint] init ok. chunk={self.chunk_size/1024/1024:.2f}MB "
              f"(effective={effective/1024/1024:.2f}MB), rank={self.rank_id}/{self.world_size}")

        self._spdk_initialized = True
        self._closed = False
        self.io_thread = None
        self._io_error = None
        self._active_handle = None
        self._handle_services = HandleServices()
        self._scheduler.request_counter = 0
        self._snapshot_generation = 0
        self._last_chunk_count = 0
        self._last_save_stats = {}
        self._live = LiveCapture(acl=self._acl, device_id=npu_device_id,
            chunk_size=requested_chunk_size,
            is_quarantined=lambda h: any(l.handle is h and l.quarantined for l in self._scheduler.leases),
            retain=lambda resource: _QUARANTINED_CONTEXTS.add(self))
        self._live_resource_lock = self._live.lock
        self._live_handles = self._live.handles
        self._quarantined_live_resources = self._live.quarantined
        atexit.register(self.close)

        self._mount_filesystem()
        self._scheduler.accepted_generation = self.metadata_generation
        self._weights = LegacyWeightsRestore(binding=self._binding, context=self.ctx,
            commit=self._commit, total_bytes=self.total_bytes, rank_id=self.rank_id,
            chunk_size=self.chunk_size, pointer_of=get_dev_ptr)
        self._experimental = LegacyExperimental(binding=self._binding, context=self.ctx,
            acl=self._acl, pointer_of=get_dev_ptr, commit=self._commit, capture=self._capture,
            rank_id=self.rank_id, device_id=npu_device_id, chunk_size=self.chunk_size,
            slot_bytes=self.slot_bytes, total_bytes=self.total_bytes,
            slot_offset=self._get_current_slot_base_offset, drain=self.wait_for_io_completion,
            load_weights=self._weights.load, diagnostic_path=self._meta_pkl)


    def _admit_checkpoint(self, timeout=None, admission=None, commit_meta=True):
        if self._closed:
            raise RuntimeError("checkpoint admission is closed")
        return self._scheduler.admit(timeout, self.admission if admission is None else admission,
                                     commit_meta)


    def _release_checkpoint_slot(self, lease):
        self._scheduler.release(lease)


    def _advance_io_sequence(self, sequence):
        self._scheduler.advance(sequence)


    def reset_checkpoint_queue(self):
        """Reopen admission after all handles are terminal."""
        acknowledged_error = self._io_error
        try:
            self.wait_for_io_completion()
        except RuntimeError:
            # Reset is the explicit acknowledgement point for the recorded
            # fail-stop error after all workers have reached a terminal state.
            pass
        def acknowledge():
            self._io_error = None
        self._scheduler.reset(self.metadata_generation, acknowledge=acknowledge)
        return acknowledged_error

    def _poison_checkpoint_queue(self, failed_handle):
        pending = self._scheduler.poison(failed_handle)
        for handle in pending:
            if handle.state == CheckpointState.QUEUED:
                handle.status = CheckpointHandle.CANCELLED
                try:
                    handle.transition(CheckpointState.CANCELLED)
                except ValueError:
                    pass
                handle._done.set()
        with self._scheduler.order:
            self._scheduler.order.notify_all()

    @property
    def layout(self):
        return self._commit.state.layout

    @layout.setter
    def layout(self, value):
        self._commit.state.layout = value

    @property
    def meta_dict(self):
        return self._commit.state.meta_dict

    @meta_dict.setter
    def meta_dict(self, value):
        self._commit.state.meta_dict = value

    @property
    def metadata_generation(self):
        return self._commit.state.metadata_generation

    @metadata_generation.setter
    def metadata_generation(self, value):
        self._commit.state.metadata_generation = value

    @property
    def active_meta_slot(self):
        return self._commit.state.active_meta_slot

    @active_meta_slot.setter
    def active_meta_slot(self, value):
        self._commit.state.active_meta_slot = value

    @property
    def stack_start_bytes(self):
        return self._commit.state.stack_start_bytes

    @stack_start_bytes.setter
    def stack_start_bytes(self, value):
        self._commit.state.stack_start_bytes = value

    # -- Filesystem mount ----------------------------------------------------

    def _mount_filesystem(self):
        self._commit.mount(self.total_bytes, self.rank_id)


    # -- Slot layout ---------------------------------------------------------

    def _get_current_slot_base_offset(self, step: int):
        if self.layout is None:
            raise RuntimeError("disk layout is not mounted")
        if self.slot_bytes != self.layout.full_slot_bytes:
            raise RuntimeError(
                "configured full slot size differs from formatted disk")
        return self.layout.full_slot_offset(
            self.rank_id, step, self.keep_last_n)

    # -- Metadata commit -----------------------------------------------------

    def _commit_metadata(self, step: int, layout: List[Dict],
                         checkpoint_meta: Dict = None):
        spec = LegacyCommitSpec(self.rank_id, self.world_size, self.chunk_size,
                                self.keep_last_n, self._meta_pkl)
        return self._commit.publish_full(spec, step, layout, checkpoint_meta)


    def _persist_metadata(self, generation=None):
        return self._commit.persist(generation)


    def flush_nvme(self):
        return self._metadata_io.flush_nvme()


    # -- Probe flag helpers --------------------------------------------------

    def set_probe_flag_ptr(self, flag_tensor: Tensor = None):
        return self._experimental.set_probe_flag_ptr(flag_tensor=flag_tensor)

    def read_probe_flag_dev(self) -> int:
        return self._experimental.read_probe_flag_dev()

    def write_probe_flag_dev(self, value: int):
        return self._experimental.write_probe_flag_dev(value=value)

    def probe_flag_selftest(self):
        return self._experimental.probe_flag_selftest()

    def set_probe_flag_value(self, value: int):
        return self._experimental.set_probe_flag_value(value=value)

    # -- Cleanup / lifecycle -------------------------------------------------

    def cleanup(self, timeout=120.0):
        closed = cleanup_legacy(scheduler=self._scheduler, lib=self._binding, ctx=self.ctx,
            initialized=getattr(self, '_spdk_initialized', False),
            drain=self.wait_for_io_completion,
            live_quarantine=getattr(self, '_quarantined_live_resources', ()),
            live_handles=getattr(self, '_live_handles', ()),
            release_live=self._release_live_resources,
            retain=lambda: _QUARANTINED_CONTEXTS.add(self), timeout=timeout)
        if closed:
            self.ctx = None
            self._metadata_io.ctx = None
            self._spdk_initialized = False

    def retained_resource_report(self):
        """Diagnostics only: a returned record is never a DMA-stop proof."""
        with self._scheduler.changed:
            leases = list(self._scheduler.leases)
        records = []
        for lease in leases:
            if not lease.quarantined:
                continue
            records.append({
                "request_id": lease.handle.request_id if lease.handle else f"sequence-{lease.sequence}",
                "generation": lease.generation, "owner": "DirectCheckpoint",
                "bytes": sum(int(p["size"]) for p in lease.params or []),
                "reason": str(lease.handle.error) if lease.handle else "preparation outcome unknown",
                "release_condition": "proven native and live DMA stop, then no remaining borrowers",
            })
        for resource in getattr(self, "_quarantined_live_resources", []):
            records.append({"request_id": resource.get("request_id"), "owner": "DirectCheckpoint.live",
                "bytes": sum(int(p["size"]) for p in resource["params"]),
                "reason": "live staging failed after an accepted copy",
                "release_condition": "proven live stream stop"})
        slots = []
        if self.ctx and hasattr(self._binding, "npu_nvme_get_retained_slots"):
            raw = (NPUNVMERetainedSlot * 16)()
            count = ctypes.c_uint32()
            rc = self._binding.npu_nvme_get_retained_slots(self.ctx, raw, 16, ctypes.byref(count))
            if rc != 0:
                raise RuntimeError(f"retained-slot query failed: {rc}")
            for value in raw[:count.value]:
                slots.append({name: getattr(value, name) for name, _ in NPUNVMERetainedSlot._fields_})
        return {"snapshots": records, "native_slots": slots,
                "native_release_condition": "request completion or proven native quiescence",
                "is_stop_proof": False}

    def get_last_io_us(self, is_read: bool = False) -> int:
        """C-layer I/O latency in microseconds (DMA + SPDK only, no Python overhead).
        Returns 0 if no I/O has been performed."""
        if not self.ctx:
            return 0
        return self._binding.npu_nvme_get_last_io_us(self.ctx, 1 if is_read else 0)

    def get_runtime_stats(self):
        """Return C-layer counters with explicit NVMe outstanding fields."""
        if not self.ctx or not hasattr(self._binding, "npu_nvme_get_stats"):
            return {}
        stats = NPUNVMEStats()
        rc = self._binding.npu_nvme_get_stats(self.ctx, ctypes.byref(stats))
        if rc != 0:
            raise RuntimeError(f"npu_nvme_get_stats failed: {rc}")
        return {name: getattr(stats, name) for name, _ctype in stats._fields_}

    def close(self):
        if not getattr(self, '_closed', False) and hasattr(self, 'ctx') and self.ctx:
            print(f"[DirectCkpt] Rank {self.rank_id} safely tearing down "
                  f"NPUNVME context...", flush=True)
            self.cleanup()
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _build_local_param_registry(self, models):
        return self._capture.build_registry(models)


    def _prepare_params(self, models):
        return self._capture.prepare(models)


    def _snapshot_params(self, params, generation):
        return self._capture.snapshot(params, generation)


    def _release_snapshot(self, params):
        return self._capture.release(params)


    def _release_live_resources(self, handle):
        return self._live.release(handle)

    def _maybe_release_live(self, handle):
        return self._live.maybe_release(handle)

    def _stage_live_params(self, params, request_id=None):
        return self._live.stage(params, request_id)

    def _record_io_error(self, error):
        self._io_error = error

    def _record_layout(self, layout):
        self.last_layout = layout

    def _retain_lease(self, lease):
        _QUARANTINED_CONTEXTS.add(self)

    # -- I/O synchronisation -------------------------------------------------

    def wait_for_io_completion(self, timeout=None):
        self._scheduler.drain(timeout)
        if self._io_error is not None:
            raise RuntimeError("Background checkpoint persistence failed") from self._io_error

    # DEPRECATED: kept as no-op for backward compatibility.
    def wait_async_io(self):
        pass

    # -- Layout --------------------------------------------------------------

    def build_layout(self, models, step: int = 0):
        return self._experimental.build_layout(models=models, step=step)

    def register_tasks(self, model: ms.nn.Cell, step: int = 0):
        return self._experimental.register_tasks(model=model, step=step)

    # -- Delta-buffer layout (for DeltaTrainCell) --------------------------------

    def build_layout_for_delta(self, delta_cell):
        return self._experimental.build_layout_for_delta(delta_cell=delta_cell)

    def register_delta_tasks(self, delta_cell, ckpt_interval: int = 5):
        return self._experimental.register_delta_tasks(delta_cell=delta_cell, ckpt_interval=ckpt_interval)

    # -- Save / Load ---------------------------------------------------------

    def save(self, model: ms.nn.Cell, step: int,
             meta_path: str = "checkpoint_meta.pkl", commit_meta: bool = True,
             _prepared_params=None, _checkpoint_meta=None, io_mode="queue",
             timeout=None, _api_enter_ns=None, _live_staging=None, _admission=None):
        if _api_enter_ns is None:
            _api_enter_ns = time.monotonic_ns()
        lease = self._admit_checkpoint(timeout, _admission, commit_meta)
        try:
            return self._save_admitted(model, step, meta_path, commit_meta,
                _prepared_params, _checkpoint_meta, io_mode, timeout,
                _api_enter_ns, _live_staging, lease)
        except BaseException as error:
            if not lease.started:
                self._io_error = error
                if lease.handle is not None:
                    lease.handle._fail(error)
                self._poison_checkpoint_queue(lease.handle)
                try:
                    if lease.params is not None:
                        if lease.live:
                            lease.quarantined = True
                            _QUARANTINED_CONTEXTS.add(self)
                        else:
                            self._release_snapshot(lease.params)
                finally:
                    if not lease.quarantined:
                        self._advance_io_sequence(lease.sequence)
                        self._release_checkpoint_slot(lease)
            raise

    def _save_admitted(self, model: ms.nn.Cell, step: int,
             meta_path: str = "checkpoint_meta.pkl", commit_meta: bool = True,
             _prepared_params=None, _checkpoint_meta=None, io_mode="queue",
             timeout=None, _api_enter_ns=None, _live_staging=None, lease=None):
        if io_mode not in ("queue", "async", "serial", "frozen_async",
                           "live_async"):
            raise ValueError(
                "io_mode must be serial, queue, async, frozen_async, or live_async")
        requested_mode = io_mode
        if requested_mode == "live_async" and _live_staging is None:
            raise ValueError("live_async requires generation-owned staging")
        io_mode = ("async" if requested_mode == "frozen_async" else
                   "live_host" if requested_mode == "live_async" else requested_mode)
        api_enter_ns = (_api_enter_ns if _api_enter_ns is not None
                        else time.monotonic_ns())
        if self._scheduler.poisoned:
            raise CheckpointQueuePoisonedError(
                "checkpoint queue is poisoned; reopen the context")
        admission_wait_ns = time.monotonic_ns() - api_enter_ns
        t_start = time.perf_counter()

        # -- T_Prep --
        t_prep_start = time.perf_counter()
        try:
            params = ((_prepared_params() if callable(_prepared_params) else _prepared_params) if _prepared_params is not None
                      else self._prepare_params(model))
        except BaseException:
            raise
        self._capture.validate(params, self.chunk_size)
        t_prep_end = time.perf_counter()
        T_Prep = t_prep_end - t_prep_start

        io_sequence = lease.sequence
        snapshot_generation = lease.sequence
        generation = lease.generation
        request_id = f"rank{self.rank_id}-pid{os.getpid()}-request{io_sequence}"
        # Freeze the graph before copying any parameter address.  A D2D copy
        # submitted while the optimizer is still running would otherwise
        # produce a mixed-step checkpoint.
        if _live_staging is None:
            self._capture.synchronize()
        snapshot_slot = int(step) % int(self.keep_last_n)
        handle = CheckpointHandle(
            self._handle_services, request_id, generation, step, rank_id=self.rank_id,
            snapshot_slot=snapshot_slot,
            snapshot_generation=snapshot_generation, timeout=timeout)
        if _live_staging is not None:
            handle.services.install_fence = self._live.install_update_fence
            handle.services.collect_fence = self._live.collect_update_wait
        lease.handle = handle
        handle.api_enter_ns = api_enter_ns
        handle.admission_wait_ns = admission_wait_ns
        handle.transition(CheckpointState.SNAPSHOTTING)
        if _live_staging is None:
            try:
                params = self._snapshot_params(params, generation)
                lease.params = params
            except BaseException as error:
                raise
        else:
            try:
                params, live_resource = self._stage_live_params(params, request_id=handle.request_id)
            except BaseException as error:
                raise
            lease.params = params
            lease.live = True
            handle._live_event = live_resource["event"]
            handle._live_buffers = [live_resource]
            handle.dma_submit_ns = live_resource["dma_submit_ns"]
            with self._live_resource_lock:
                self._live_handles.add(handle)
            _live_staging = live_resource
            handle.transition(CheckpointState.SNAPSHOT_READY)
            handle.transition(CheckpointState.QUEUED)
            handle.transition(CheckpointState.DMA_COPYING)
        if _live_staging is None:
            handle.transition(CheckpointState.SNAPSHOT_READY)
        if _live_staging is None:
            checksum = hashlib.sha256()
            for item in sorted(params, key=lambda value: value["name"]):
                checksum.update(item["name"].encode("utf-8"))
                checksum.update(str(item.get("sha256") or "").encode("ascii"))
            handle.checksum = checksum.hexdigest()

        lease.params = params
        lease.live = _live_staging is not None

        # -- T_Layout --
        try:
            base_offset_bytes = self._get_current_slot_base_offset(step)
            print(f"[DirectCkpt] Rank {self.rank_id} saving step {step} to offset "
                  f"{base_offset_bytes / 1024**3:.2f} GB ...", flush=True)

            current_offset = base_offset_bytes
            layout, dev_params, host_params = [], [], []

            for p in params:
                aligned_bytes = int(math.ceil(p["size"] / 4096.0)) * 4096

                if current_offset + aligned_bytes > self.total_bytes:
                    raise MemoryError(
                        f"Rank {self.rank_id} CRITICAL: DMA Write will exceed disk "
                        f"physical capacity! Offset: "
                        f"{(current_offset + aligned_bytes)/1024**3:.2f}GB > "
                        f"Total: {self.total_bytes/1024**3:.2f}GB")

                if (current_offset - base_offset_bytes) + aligned_bytes > self.slot_bytes:
                    raise MemoryError(
                        f"Rank {self.rank_id} OOM! Tensor {p['name']} exceeds slot size.")

                p_record = {**p, "offset": current_offset}
                layout.append(p_record)

                if p.get("np_arr") is not None:
                    host_params.append(p_record)
                else:
                    dev_params.append(p_record)

                current_offset += aligned_bytes
        except BaseException as error:
            raise

        total_written = 0

        dev_chunks, dev_sz = build_chunks(dev_params, self.chunk_size)
        if dev_chunks:
            with open(f"task_mapping_rank_{self.rank_id}.txt", "w") as f:
                for i, chunk in enumerate(dev_chunks):
                    f.write(f"TaskIdx: {i} | Name: {chunk[3]} | "
                            f"Size: {chunk[2].value}\n")
            c_ptrs_dev, c_offs_dev, c_sizes_dev = build_ctypes_arrays(dev_chunks)
        else:
            c_ptrs_dev = c_offs_dev = c_sizes_dev = None

        host_chunks, host_sz = build_chunks(host_params, self.chunk_size)
        if host_chunks:
            c_ptrs_host, c_offs_host, c_sizes_host = build_ctypes_arrays(host_chunks)
        else:
            c_ptrs_host = c_offs_host = c_sizes_host = None

        t_layout_end = time.perf_counter()
        T_Layout = t_layout_end - t_prep_end

        services = WorkerServices(
            scheduler=self._scheduler, io_mutex=self._io_mutex,
            transport=LegacyBatchTransport(self._binding, self.ctx), commit=self._commit,
            flush=self.flush_nvme, publish=self._commit_metadata,
            complete_live=self._live.complete if _live_staging is not None else None,
            release_snapshot=self._release_snapshot, release_live=self._maybe_release_live,
            poison=self._poison_checkpoint_queue, record_error=self._record_io_error,
            record_layout=self._record_layout, retain=self._retain_lease)
        job = TransferJob(io_mode, io_sequence, handle, lease, _live_staging, params, layout, commit_meta, step, _checkpoint_meta, meta_path, t_start, T_Prep, T_Layout, self.rank_id)

        num_dev_val = len(dev_chunks) if dev_chunks else 0
        dev_sz_val = dev_sz if dev_chunks else 0
        num_host_val = len(host_chunks) if host_chunks else 0
        host_sz_val = host_sz if host_chunks else 0

        self._active_handle = handle
        with self._scheduler.handles_lock:
            self._scheduler.active_handles.add(handle)
        self._last_chunk_count = num_dev_val + num_host_val
        self._last_save_stats = {"prep_time": T_Prep,
                                 "layout_time": T_Layout,
                                 "request_id": request_id,
                                 "generation": generation}

        handle.services.chunk_count = self._last_chunk_count
        handle.services.stats = self._last_save_stats

        worker = threading.Thread(
            target=run_legacy_transfer,
            args=(services, job, c_ptrs_dev, c_offs_dev, c_sizes_dev, num_dev_val, dev_sz_val,
                  c_ptrs_host, c_offs_host, c_sizes_host, num_host_val, host_sz_val))
        with self._scheduler.handles_lock:
            self._scheduler.handle_threads[request_id] = worker
            self.io_thread = worker
        if _live_staging is None:
            handle.transition(CheckpointState.QUEUED)
        # If start fails the wrapper still owns the lease and all snapshots.
        worker.start()
        lease.started = True

        if requested_mode == "frozen_async":
            freeze_started_ns = time.monotonic_ns()
            handle.wait(timeout=handle.timeout)
            handle.freeze_wait_ns = time.monotonic_ns() - freeze_started_ns

        t_return = time.perf_counter()
        handle.api_return_ns = time.monotonic_ns()
        print(f"[Timeline][Rank {self.rank_id}] Step {step} | Python save() "
              f"dispatched to background thread. "
              f"Layout cost: {T_Layout*1000:.2f}ms", flush=True)

        return handle

    def _ordered_components(self, components):
        return self._capture.ordered_components(components)


    def _prepare_state_components(self, components, with_checksums=True):
        return self._capture.prepare_state_components(components, with_checksums)


    def save_state(self, components, control_state, step: int,
                   meta_path: str = "checkpoint_meta.pkl",
                   commit_meta: bool = True, verify_checksums: bool = True,
                   io_mode: str = "queue", admission: str = None,
                   timeout: float = None):
        """Freeze and persist a versioned complete training state.

        ``components`` maps namespaces such as ``model`` and ``optimizer`` to
        MindSpore objects exposing ``parameters_and_names``.  ``control_state``
        contains JSON-tagged Python/NumPy state and is returned by
        :meth:`load_state` for the caller to re-apply.
        """
        from training_state import (validate_state_names, encode_control_value,
            TRAINING_STATE_SCHEMA_VERSION)
        api_enter_ns = time.monotonic_ns()
        requested_admission = self.admission if admission is None else admission
        if requested_admission not in ("block", "try"):
            raise ValueError("admission must be block or try")
        if self._scheduler.poisoned:
            raise CheckpointQueuePoisonedError(
                "checkpoint queue is poisoned; reopen the context")
        def prepare():
            validate_state_names(components, control_state)
            if io_mode != "live_async" and hasattr(self._framework.hal, "synchronize"):
                self._framework.hal.synchronize()
            params = self._prepare_state_components(
                components, with_checksums=(verify_checksums and io_mode != "live_async"))
            for name in sorted(control_state):
                payload, control_meta = encode_control_value(control_state[name])
                params.append({
                    "name": f"control/{name}",
                    "source_name": name,
                    "component": "control",
                    "category": "control",
                    "placement": "host",
                    "ptr": 0,
                    "size": int(payload.nbytes),
                    "shape": [int(payload.nbytes)],
                    "dtype": "uint8",
                    "np_arr": payload,
                    "param_ref": None,
                    "codec": control_meta["codec"],
                    "sha256": control_meta["sha256"],
                })
            return params
        checkpoint_meta = {
            "type": "TRAINING_STATE_FULL",
            "schema_version": TRAINING_STATE_SCHEMA_VERSION,
            "state_step": int(step),
            "checksum": "sha256" if verify_checksums else "none",
            "components": self._ordered_components(components),
            "control_names": sorted(control_state),
        }
        return self.save(
            None, step=step, meta_path=meta_path, commit_meta=commit_meta,
            _prepared_params=prepare, _checkpoint_meta=checkpoint_meta,
            io_mode=io_mode, timeout=timeout, _api_enter_ns=api_enter_ns,
            _live_staging=(True if io_mode == "live_async" else None),
            _admission=requested_admission)

    def try_save_state(self, components, control_state, step: int, **kwargs):
        """Submit a FULL generation or raise an explicit BUSY/poison error."""
        kwargs["admission"] = "try"
        return self.save_state(components, control_state, step, **kwargs)

    def _select_checkpoint_record(self, step):
        return self._commit.select(step, total_bytes=self.total_bytes, rank_id=self.rank_id)


    def load_state(self, components, step: int = None,
                   verify_checksums: bool = True):
        """Legacy in-place restore; strict unready targets are not yet supported."""
        from training_state import validate_state_names, TRAINING_STATE_SCHEMA_VERSION
        from npu_nvme.framework.restore import LegacyStateTarget
        validate_state_names(components, {})
        target = LegacyStateTarget(components, self._capture, self._framework, self._ops)
        runtime = LegacyRestore(select_record=self._select_checkpoint_record,
            transport=LegacyBatchTransport(self._binding, self.ctx), rank_id=self.rank_id,
            chunk_size=self.chunk_size, schema_version=TRAINING_STATE_SCHEMA_VERSION)
        return runtime.load_state(target, step, verify_checksums)


    def restore_full_state(self, target_factory, expected_spec, step, *, reader=None,
                           deadline=None, request_id="d1-restore"):
        """Strict D1 restore: factory owns a fresh unready target until success."""
        if reader is None:
            if self.ctx is None:
                raise RuntimeError("checkpoint backend is not open")
            def reader(offset, size):
                buf = ctypes.create_string_buffer(size)
                chunks = (ctypes.c_void_p * 1)(ctypes.addressof(buf))
                offsets = (ctypes.c_uint64 * 1)(offset)
                sizes = (ctypes.c_size_t * 1)(size)
                rc = self._binding.npu_nvme_read_batch_host(self.ctx, chunks, offsets, sizes, 1)
                if rc != 0:
                    raise RuntimeError(f"strict host read failed: {rc}")
                return bytes(buf.raw)
        session = StrictRestoreSession(select_record=lambda selected: self._select_checkpoint_record(selected),
            reader=reader, request_id=request_id)
        return session.restore_full_state(target_factory, expected_spec, step, deadline=deadline)

    def load(self, model: ms.nn.Cell, step: int = None,
             meta_path: str = "checkpoint_meta.pkl"):
        return self._weights.load(model=model, step=step, meta_path=meta_path)

    # -- Delta frame I/O --------------------------------------------------------

    @staticmethod
    def _require_incremental_enabled():
        from npu_nvme.experimental.legacy import LegacyExperimental
        return LegacyExperimental._require_incremental_enabled()

    def delta_init(self, slot_size_mb: int = 256, slot_count: int = 128):
        return self._experimental.delta_init(slot_size_mb=slot_size_mb, slot_count=slot_count)

    def delta_save(self, step: int, block_patches: list, small_patches: list,
                   lossless: bool = False, base_generation: int = None):
        return self._experimental.delta_save(step=step, block_patches=block_patches, small_patches=small_patches, lossless=lossless, base_generation=base_generation)

    def delta_save_lossless(self, step: int, block_patches: list,
                            small_patches: list, base_generation: int = None):
        return self._experimental.delta_save_lossless(step=step, block_patches=block_patches, small_patches=small_patches, base_generation=base_generation)

    def write_host_frame(self, frame: bytes, byte_offset: int):
        return self._experimental.write_host_frame(frame=frame, byte_offset=byte_offset)

    def read_host_frame(self, byte_offset: int, frame_size: int):
        return self._experimental.read_host_frame(byte_offset=byte_offset, frame_size=frame_size)

    def delta_load_slot(self, slot_idx: int, return_meta: bool = False):
        return self._experimental.delta_load_slot(slot_idx=slot_idx, return_meta=return_meta)

    def delta_load_chain(self, from_step: int, to_step: int):
        return self._experimental.delta_load_chain(from_step=from_step, to_step=to_step)

    def _find_nearest_full(self, target_step: int):
        return self._experimental._find_nearest_full(target_step=target_step)

    def recover(self, model: "ms.nn.Cell", target_step: int):
        return self._experimental.recover(model=model, target_step=target_step)

    @property
    def chunks(self):
        return self._experimental.chunks

    @chunks.setter
    def chunks(self, value):
        self._experimental.chunks = value

    @property
    def probe_flag_ptr(self):
        return self._experimental.probe_flag_ptr

    @probe_flag_ptr.setter
    def probe_flag_ptr(self, value):
        self._experimental.probe_flag_ptr = value

    @property
    def delta_block_size(self):
        return self._experimental.delta_block_size

    @delta_block_size.setter
    def delta_block_size(self, value):
        self._experimental.delta_block_size = value

    @property
    def _delta_slot_size(self):
        return self._experimental._delta_slot_size

    @_delta_slot_size.setter
    def _delta_slot_size(self, value):
        self._experimental._delta_slot_size = value

    @property
    def _delta_slot_count(self):
        return self._experimental._delta_slot_count

    @_delta_slot_count.setter
    def _delta_slot_count(self, value):
        self._experimental._delta_slot_count = value

    @property
    def _delta_next_slot(self):
        return self._experimental._delta_next_slot

    @_delta_next_slot.setter
    def _delta_next_slot(self, value):
        self._experimental._delta_next_slot = value

    @property
    def _delta_step_map(self):
        return self._experimental._delta_step_map

    @_delta_step_map.setter
    def _delta_step_map(self, value):
        self._experimental._delta_step_map = value

    @property
    def _delta_frame_sizes(self):
        return self._experimental._delta_frame_sizes

    @_delta_frame_sizes.setter
    def _delta_frame_sizes(self, value):
        self._experimental._delta_frame_sizes = value

    @property
    def _dump_meta_pkl(self):
        return self._experimental._dump_meta_pkl

    @_dump_meta_pkl.setter
    def _dump_meta_pkl(self, value):
        self._experimental._dump_meta_pkl = value
