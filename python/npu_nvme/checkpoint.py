"""Canonical strict FULL API. Legacy implementations live only in the archive tag."""
from npu_nvme.strict_checkpoint import StrictCheckpoint, MigrationRequired
from npu_nvme.runtime.handle import CheckpointHandle
from npu_nvme.runtime.scheduler import CheckpointBusyError
from npu_nvme.framework.parameters import get_dev_ptr

DirectCheckpoint = StrictCheckpoint

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
