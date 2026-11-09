"""Strict FULL facade wiring; library/framework initialization remains lazy."""
from copy import deepcopy
import ctypes
import atexit
import math
import os
import time
from npu_nvme.runtime.commit import MetadataState
from npu_nvme.runtime.d1_commit import D1CommitCoordinator
from npu_nvme.runtime.d1_runtime import FullRuntime
from npu_nvme.runtime.d1_schema import CHUNK_BYTES, MAX_CHUNK_BYTES
from npu_nvme.storage.bindings import load_backend
from npu_nvme.storage.full_transport import FullTransport

_RETAINED_STORES = set()


class MigrationRequired(RuntimeError): pass


class StrictCheckpoint:
    def __init__(self, nvme_addr='0000:83:00.0', npu_device_id=7, pipeline_depth=4,
                 requested_chunk_size=CHUNK_BYTES, enable_profiling=False, profiling_dir='./output/profiling',
                 rank_id=0, world_size=1, base_offset_bytes=0, shard_span_bytes=None,
                 spdk_shm_id=1, keep_last_n=2, slot_size_gb=10, warmup_fn=None,
                 checkpoint_slots=1, admission='block', request_slots=1, backend=None, framework=None,
                 strict=True):
        if not strict or rank_id != 0 or world_size != 1 or keep_last_n != 2 or checkpoint_slots != 1 or request_slots not in (None,1):
            raise MigrationRequired('D1 requires strict FULL, rank 0/world_size 1, retention 2 and one pending request')
        if base_offset_bytes or shard_span_bytes is not None:
            raise MigrationRequired('D1 uses one namespace; sharded offsets require D2-rank')
        if admission not in ('block','try'): raise ValueError('admission must be block or try')
        if type(requested_chunk_size) is not int or not 0 < requested_chunk_size <= MAX_CHUNK_BYTES or requested_chunk_size % 4096:
            raise ValueError('D1 chunk size must be 4 KiB aligned and at most 16 MiB')
        self.admission = admission
        if framework is None:
            import mindspore as framework
        backend = backend or load_backend()
        self._framework, self._backend = framework, backend
        if warmup_fn is not None: warmup_fn()
        os.environ['SPDK_SHM_ID'] = str(spdk_shm_id)
        self.transport = FullTransport(backend, pci=nvme_addr, npu=npu_device_id, depth=pipeline_depth,
            chunk_size=requested_chunk_size, profiling_dir=profiling_dir, profiling=enable_profiling)
        _RETAINED_STORES.add(self)
        self._runtime = None
        try:
            state = MetadataState()
            self.transport.metadata.mount(state, self.transport.total_bytes, rank_id)
            if state.layout.full_slot_bytes != slot_size_gb * 1024**3:
                raise ValueError('configured slot size differs from media')
            commit = D1CommitCoordinator(metadata_io=self.transport.metadata, state=state)
            from npu_nvme.framework.capture import FrozenCapture
            from npu_nvme.framework.full_state import prepare_full
            from npu_nvme.framework.parameters import get_dev_ptr
            capture = FrozenCapture(rank_id=0, device_id=npu_device_id, framework=framework,
                                    acl=backend.acl_lib, pointer_of=get_dev_ptr)
            self._runtime = FullRuntime(commit, self.transport,
                prepare=lambda components, controls, spec:prepare_full(capture, components, controls, spec),
                freeze=capture.snapshot, release=capture.release, chunk_size=requested_chunk_size)
            self.npu_device_id = npu_device_id
            atexit.register(self._exit_close)
        except BaseException:
            self.transport.close(5)
            _RETAINED_STORES.discard(self)
            raise

    @property
    def ctx(self): return self.transport.ctx
    @property
    def meta_dict(self): return deepcopy(self._runtime.commit.state.meta_dict)
    @property
    def metadata_generation(self): return self._runtime.commit.state.metadata_generation

    def save_state(self, components, control_state, step, *, expected_spec=None, timeout=None,
                   admission=None, verify_checksums=True, io_mode='frozen_async', commit_meta=True, meta_path=None):
        if expected_spec is None: raise MigrationRequired('save_state requires an explicit expected_spec')
        if verify_checksums is not True or commit_meta is not True or io_mode not in ('frozen_async','queue'):
            raise MigrationRequired('D1 requires frozen FULL with mandatory checksums and catalog publication')
        return self._runtime.save(components, control_state, step, expected_spec,
                                  admission=admission or self.admission, timeout=timeout)

    def try_save_state(self, components, control_state, step, **kwargs):
        return self.save_state(components, control_state, step, admission='try', **kwargs)

    def begin_restore_full_state(self, target_factory, expected_spec, step=None):
        return self._runtime.begin_restore(target_factory, expected_spec, step)

    def restore_full_state(self, target_factory, expected_spec, step=None, *, deadline=None):
        return self.begin_restore_full_state(target_factory, expected_spec, step).result(deadline)

    def resolve(self, request_id): return self._runtime.commit.resolve(request_id)
    def wait_for_io_completion(self, timeout=120): return self._runtime.drain(timeout)

    def close(self, timeout=120):
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout < 0: raise ValueError('invalid close timeout')
        if not self.transport.ctx: return
        self._runtime.close(timeout)
        _RETAINED_STORES.discard(self)
        atexit.unregister(self._exit_close)

    cleanup = close

    def _exit_close(self):
        try: self.close(5)
        except Exception: pass  # Strong owner remains retained; never force DMA cleanup.

    @staticmethod
    def live_async_capability():
        return dict(supported=False, code='RETIRED_NONSTRICT_FULL',
                    required='Use strict frozen FULL; live/FaF requires a later validated contract')

    def get_runtime_stats(self):
        from npu_nvme.storage.bindings import NPUNVMEStats
        stats = NPUNVMEStats()
        rc = self._backend.lib.npu_nvme_get_stats(self.ctx, ctypes.byref(stats))
        if rc != 0: raise RuntimeError(f'npu_nvme_get_stats failed: {rc}')
        return {name:getattr(stats,name) for name,_ in stats._fields_}

    def _migration(self, *args, **kwargs):
        raise MigrationRequired('legacy operation retired; use save_state(expected_spec=...) and restore_full_state(target_factory, expected_spec)')

    save = load = load_state = recover = delta_init = delta_save = delta_save_lossless = _migration
    register_tasks = register_delta_tasks = _persist_metadata = _migration

    write_host_frame = read_host_frame = delta_load_slot = delta_load_chain = _migration
    flush_nvme = _commit_metadata = set_probe_flag_ptr = _migration
