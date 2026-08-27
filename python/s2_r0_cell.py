"""MindSpore graph-side capture and ACK commit cells for S2-R0.

The cells compute a parameter-local changed bitmap against a persisted HBM
reference.  R0 is a correctness-first, synchronous capture contract: the
caller must not advance training until the ACK commit completes, so the
current HBM values remain stable while selected blocks are written.  This
avoids a second full copy of the model/optimizer state; an asynchronous
double-buffered snapshot is deliberately a later performance work package.
"""

from __future__ import annotations

from dataclasses import dataclass

import mindspore as ms
from mindspore import Parameter, ParameterTuple, Tensor, nn, ops
from mindspore.common.initializer import Zero, initializer
import numpy as np

from incremental_manifest import build_training_state_manifest


def _zero_parameter(shape, dtype, name):
    return Parameter(initializer(Zero(), shape, dtype), requires_grad=False,
                     name=name)


class _CaptureShard(nn.Cell):
    def __init__(self, current, persisted, blocks):
        super().__init__(auto_prefix=False)
        self.current = ParameterTuple(tuple(current))
        self.persisted = ParameterTuple(tuple(persisted))
        self.blocks = tuple(blocks)

    def construct(self):
        flags = []
        for current, persisted, block_list in zip(
                self.current, self.persisted, self.blocks):
            current_flat = ops.reshape(current, (-1,))
            persisted_flat = ops.reshape(persisted, (-1,))
            for block in block_list:
                start = block.element_offset
                end = start + block.element_count
                changed = ops.ReduceAny(keep_dims=False)(
                    ops.NotEqual()(current_flat[start:end],
                                   persisted_flat[start:end]))
                flags.append(changed)
        return ops.Stack()(tuple(flags))


class _CommitShard(nn.Cell):
    def __init__(self, current, persisted):
        super().__init__(auto_prefix=False)
        self.persisted = ParameterTuple(tuple(persisted))
        self.current = ParameterTuple(tuple(current))
        self.assign = ops.Assign()
        self.depend = ops.Depend()
        self.zero = Tensor(0, ms.int32)

    def construct(self):
        token = self.zero
        for current, persisted in zip(self.current, self.persisted):
            token = self.depend(token, self.assign(persisted, current))
        return token


@dataclass(frozen=True)
class R0BlockBuffer:
    state_index: int
    block_index: int
    name: str
    element_offset: int
    element_count: int
    dtype: str
    pointer: int


class R0NpuState:
    """Complete R0 state capture with HBM reference and ACK commit cells."""

    def __init__(self, components, block_size=524288, small_threshold=10000,
                 shard_fields=16, capture_blocks=128):
        self.manifest = build_training_state_manifest(
            components, block_size=block_size, small_threshold=small_threshold)
        self.fields = self.manifest.fields
        self.current = self._collect_current(components)
        self.persisted = []
        for field in self.fields:
            parameter = self.current[field.canonical_name]
            self.persisted.append(_zero_parameter(
                parameter.shape, parameter.dtype,
                f"r0_persisted_{field.state_index}"))

        self.capture_cells = []
        self.commit_cells = []
        capture_group = []
        capture_count = 0
        for field in self.fields:
            for begin in range(0, len(field.blocks), int(capture_blocks)):
                blocks = field.blocks[begin:begin + int(capture_blocks)]
                if capture_group and capture_count + len(blocks) > int(capture_blocks):
                    self.capture_cells.append(self._make_capture_cell(capture_group))
                    capture_group = []
                    capture_count = 0
                capture_group.append((field, blocks))
                capture_count += len(blocks)
        if capture_group:
            self.capture_cells.append(self._make_capture_cell(capture_group))

        for begin in range(0, len(self.fields), int(shard_fields)):
            end = min(begin + int(shard_fields), len(self.fields))
            fields = self.fields[begin:end]
            self.commit_cells.append(_CommitShard(
                [self.current[field.canonical_name] for field in fields],
                self.persisted[begin:end]))
        self.initialized = False

    def _make_capture_cell(self, group):
        return _CaptureShard(
            [self.current[field.canonical_name] for field, _ in group],
            [self.persisted[field.state_index] for field, _ in group],
            [blocks for _, blocks in group])

    @staticmethod
    def _collect_current(components):
        result = {}
        seen = set()
        preferred = [name for name in ("model", "optimizer") if name in components]
        preferred.extend(sorted(name for name in components
                                if name not in {"model", "optimizer"}))
        for namespace in preferred:
            for source_name, parameter in components[namespace].parameters_and_names():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                result[f"{namespace}/{source_name}"] = parameter
        return result

    def initialize(self):
        """Capture and commit the FULL state as persisted reference."""
        for cell in self.commit_cells:
            cell()
        if hasattr(ms.hal, "synchronize"):
            ms.hal.synchronize()
        self.initialized = True

    def capture(self):
        if not self.initialized:
            raise RuntimeError("R0 state must be initialized from a FULL first")
        flags = []
        for cell in self.capture_cells:
            flags.extend(np.asarray(cell().asnumpy(), dtype=np.bool_).reshape(-1))
        if hasattr(ms.hal, "synchronize"):
            ms.hal.synchronize()
        return np.asarray(flags, dtype=np.bool_)

    def commit_ack(self):
        for cell in self.commit_cells:
            cell()
        if hasattr(ms.hal, "synchronize"):
            ms.hal.synchronize()

    def changed_buffers(self, flags):
        """Return HBM block descriptors in manifest/flag order."""
        flags = np.asarray(flags, dtype=np.bool_).reshape(-1)
        all_blocks = [block for field in self.fields for block in field.blocks]
        if flags.size != len(all_blocks):
            raise ValueError("changed bitmap length does not match manifest")
        from direct_checkpoint import get_dev_ptr
        buffers = []
        base_ptrs = {}
        total_bytes = {}
        for field in self.fields:
            name = field.canonical_name
            base = int(get_dev_ptr(self.current[name]) or 0)
            if base <= 0:
                raise RuntimeError(f"current parameter has no HBM pointer: {name}")
            itemsize = np.dtype(field.dtype).itemsize
            elements = int(np.prod(field.shape, dtype=np.int64))
            base_ptrs[name] = base
            total_bytes[name] = elements * itemsize
        for flag, block, field in zip(flags, all_blocks,
                                       [field for field in self.fields
                                        for _ in field.blocks]):
            if not flag:
                continue
            name = field.canonical_name
            itemsize = np.dtype(field.dtype).itemsize
            offset_bytes = int(block.element_offset) * itemsize
            size_bytes = int(block.element_count) * itemsize
            if (block.element_offset < 0 or block.element_count < 0 or
                    offset_bytes < 0 or offset_bytes + size_bytes > total_bytes[name]):
                raise ValueError(f"R0 block exceeds parameter bounds: {name}/"
                                 f"{block.block_index}")
            pointer = base_ptrs[name] + offset_bytes
            if pointer <= 0 or pointer + size_bytes <= pointer:
                raise ValueError(f"R0 HBM pointer overflow: {name}/"
                                 f"{block.block_index}")
            buffers.append(R0BlockBuffer(
                field.state_index, block.block_index, field.canonical_name,
                block.element_offset, block.element_count, field.dtype,
                pointer))
        return buffers


__all__ = ["R0BlockBuffer", "R0NpuState"]
