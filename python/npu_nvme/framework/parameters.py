"""Allocation-backed tensor addresses for frozen FULL capture and restore."""
from __future__ import annotations
from dataclasses import dataclass
import math


class UnknownTensorAddress(RuntimeError):
    """The allocation's location or extent cannot be proved before DMA."""


@dataclass(frozen=True)
class TensorAddress:
    placement: str
    pointer: int
    device_id: int | None
    allocation_bytes: int
    provenance: str


def validate_allocation(row, *, required_bytes, expected_device=None):
    """Validate native metadata; errors never select a Host fallback."""
    if type(required_bytes) is not int or required_bytes <= 0:
        raise ValueError('tensor extent must be a positive integer')
    if row.get('provenance') != 'mindspore.DeviceAddress':
        raise UnknownTensorAddress('missing allocation provenance')
    placement = row.get('placement')
    if placement == 'host':
        return TensorAddress('host', 0, None, required_bytes, row['provenance'])
    if placement != 'device':
        raise UnknownTensorAddress('unsupported tensor allocation location')
    pointer, size, device = (row.get(key) for key in ('pointer', 'allocation_bytes', 'device_id'))
    if any(type(value) is not int for value in (pointer, size, device)):
        raise UnknownTensorAddress('invalid device allocation metadata')
    if pointer <= 0 or size < required_bytes or pointer + required_bytes > 1 << 64 or device < 0:
        raise UnknownTensorAddress('invalid or undersized device allocation')
    if expected_device is not None and device != expected_device:
        raise UnknownTensorAddress(f'tensor belongs to NPU {device}, expected {expected_device}')
    return TensorAddress('device', pointer, device, size, row['provenance'])


def describe_tensor(tensor, *, expected_device=None, required_bytes=None):
    import mindspore as ms
    try:
        from . import _address_native
    except ImportError as error:
        raise UnknownTensorAddress('build the allocation adapter with tools/build_mindspore_address.py '
                                   'using the selected MindSpore environment') from error
    if _address_native.framework_version != ms.__version__:
        raise UnknownTensorAddress('allocation adapter was built for another MindSpore version')
    if isinstance(tensor, ms.Parameter) and hasattr(tensor, 'is_inited') and not tensor.is_inited:
        tensor.init_data()
    data = tensor.data if hasattr(tensor, 'data') else tensor
    if required_bytes is None:
        import numpy as np
        required_bytes = math.prod(data.shape) * np.dtype(ms.dtype_to_nptype(data.dtype)).itemsize
    try:
        row = _address_native.allocation(data)
    except Exception as error:
        raise UnknownTensorAddress('cannot inspect tensor allocation') from error
    if row.get('placement') == 'device':
        # A Tensor view can begin inside its owning allocation. The raw pointer
        # is useful only after location/extent have been obtained independently.
        exported = int(data._data_ptr())
        offset = exported - row['pointer']
        if offset < 0 or offset + required_bytes > row['allocation_bytes']:
            raise UnknownTensorAddress('exported pointer escapes its device allocation')
        if not data.is_contiguous():
            raise UnknownTensorAddress('non-contiguous device tensor cannot use linear DMA')
        row = dict(row, pointer=exported, allocation_bytes=row['allocation_bytes']-offset)
    return validate_allocation(row, required_bytes=required_bytes, expected_device=expected_device)


def get_dev_ptr(tensor, *, expected_device=None):
    """Return a verified NPU pointer, or zero for a verified Host allocation.

    Unknown placement is an error. `_data_ptr()` can return an ordinary Host
    heap address; Tensor.device is a preference, not allocation provenance.
    """
    return describe_tensor(tensor, expected_device=expected_device).pointer
