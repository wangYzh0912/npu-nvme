"""Strict D1 streaming restore into a fresh, permanently unready-on-failure target."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any, Callable, Mapping
from npu_nvme.types import RestoreReceipt

class StrictRestoreError(RuntimeError):
    pass

@dataclass(frozen=True)
class RestorePlan:
    step: int
    generation: int
    params: tuple
    controls: tuple

class StrictRestoreSession:
    """Bounded, checksummed reader; target ownership never escapes on failure."""
    def __init__(self, *, select_record: Callable[[int], Mapping], reader,
                 request_id: str = 'd1-restore', max_chunk: int = 1 << 20,
                 clock=time.monotonic):
        if max_chunk <= 0 or max_chunk > (1 << 20) or max_chunk % 4096:
            raise ValueError('max_chunk must be a positive 4 KiB-aligned value <= 1 MiB')
        self.select_record, self.reader = select_record, reader
        self.request_id, self.max_chunk, self.clock = request_id, max_chunk, clock

    def _read(self, offset, size):
        if callable(self.reader):
            data = self.reader(offset, size)
        elif hasattr(self.reader, 'read'):
            data = self.reader.read(offset, size)
        else:
            raise StrictRestoreError('reader must be callable or expose read(offset,size)')
        if not isinstance(data, (bytes, bytearray, memoryview)) or len(data) != size:
            raise StrictRestoreError('reader returned an invalid byte count')
        return bytes(data)

    def _plan(self, expected_spec, step):
        record = self.select_record(step)
        if isinstance(record, tuple) and len(record) == 2:
            selected_step, record = record
        else:
            selected_step = step
        if not isinstance(record, Mapping) or record.get('strict_contract') != 'D1':
            raise StrictRestoreError('record is not a strict D1 FULL record')
        if int(record.get('state_step', selected_step)) != int(selected_step):
            raise StrictRestoreError('record step does not match selection')
        if int(record.get('generation', 0)) <= 0:
            raise StrictRestoreError('record has no checkpoint generation')
        identity = record.get('identity', {})
        expected_identity = expected_spec.get('identity', {}) if isinstance(expected_spec, Mapping) else {}
        for key, value in expected_identity.items():
            if identity.get(key) != value:
                raise StrictRestoreError(f'checkpoint identity mismatch: {key}')
        params = record.get('params')
        if not isinstance(params, Mapping) or not params:
            raise StrictRestoreError('strict record has no tensor manifest')
        expected_names = set(expected_spec.get('parameters', params)) if isinstance(expected_spec, Mapping) else set(params)
        if set(params) != expected_names:
            raise StrictRestoreError('checkpoint tensor set does not match target specification')
        seen = []
        planned = []
        for name, info in params.items():
            if not isinstance(name, str) or not isinstance(info, Mapping):
                raise StrictRestoreError('invalid tensor descriptor')
            size = int(info.get('size', 0)); offset = int(info.get('offset', -1))
            if size <= 0 or offset < 0 or offset % 4096:
                raise StrictRestoreError(f'invalid extent for {name}')
            shape = tuple(int(x) for x in info.get('shape', ()))
            if not shape or any(x <= 0 for x in shape):
                raise StrictRestoreError(f'invalid shape for {name}')
            dtype = str(info.get('dtype', ''))
            if not dtype or len(dtype) > 64:
                raise StrictRestoreError(f'invalid dtype for {name}')
            chunks = info.get('chunks')
            if not isinstance(chunks, list) or not chunks:
                raise StrictRestoreError(f'{name} lacks per-chunk checksums')
            cursor = offset; total = 0
            for chunk in chunks:
                if not isinstance(chunk, Mapping): raise StrictRestoreError('invalid chunk descriptor')
                co, cs = int(chunk.get('offset', -1)), int(chunk.get('size', 0))
                if co != cursor or cs <= 0 or cs > self.max_chunk or co % 4096:
                    raise StrictRestoreError(f'unaligned or non-contiguous chunk for {name}')
                digest = chunk.get('sha256', '')
                if not isinstance(digest, str) or len(digest) != 64:
                    raise StrictRestoreError(f'missing chunk checksum for {name}')
                cursor += cs; total += cs
            if total != size or not isinstance(info.get('sha256'), str) or len(info['sha256']) != 64:
                raise StrictRestoreError(f'invalid whole-tensor checksum for {name}')
            seen.append((offset, offset + size, name)); planned.append((name, info, tuple(chunks)))
        seen.sort()
        for left, right in zip(seen, seen[1:]):
            if right[0] < left[1]: raise StrictRestoreError('tensor extents overlap')
        controls = record.get('controls', record.get('control_names', []))
        if isinstance(controls, Mapping): controls = tuple(sorted(controls))
        elif isinstance(controls, list): controls = tuple(sorted(str(x) for x in controls))
        else: controls = ()
        return RestorePlan(int(selected_step), int(record['generation']), tuple(planned), tuple(controls))

    def restore_full_state(self, target_factory, expected_spec, step, deadline=None):
        deadline_at = None if deadline is None else (self.clock() + float(deadline) if float(deadline) < 1e12 else float(deadline))
        plan = self._plan(expected_spec, step)
        target = target_factory(expected_spec)
        ready = False; bytes_read = 0; count = 0
        digest = hashlib.sha256()
        try:
            if getattr(target, 'ready', False):
                raise StrictRestoreError('target factory returned a ready target')
            if hasattr(target, 'prepare_restore'): target.prepare_restore(expected_spec)
            for name, info, chunks in plan.params:
                tensor_digest = hashlib.sha256()
                for chunk in chunks:
                    if deadline_at is not None and self.clock() > deadline_at: raise TimeoutError('strict restore deadline expired')
                    offset, size = int(chunk['offset']), int(chunk['size'])
                    data = self._read(offset, size)
                    if hashlib.sha256(data).hexdigest() != chunk['sha256']:
                        raise StrictRestoreError(f'chunk checksum mismatch for {name}@{offset}')
                    tensor_digest.update(data); digest.update(data)
                    if hasattr(target, 'apply_chunk'): target.apply_chunk(name, offset, data)
                    elif hasattr(target, 'apply'): target.apply(name, offset, data)
                    else: raise StrictRestoreError('target lacks apply_chunk/apply')
                    bytes_read += size; count += 1
                if tensor_digest.hexdigest() != info['sha256']:
                    raise StrictRestoreError(f'whole-tensor checksum mismatch for {name}')
            if hasattr(target, 'verify_controls') and not target.verify_controls(plan.controls):
                raise StrictRestoreError('control applicability verification failed')
            if hasattr(target, 'finish_restore'): target.finish_restore(plan.controls)
            if hasattr(target, 'mark_ready'): target.mark_ready()
            elif hasattr(target, 'ready'): target.ready = True
            ready = bool(getattr(target, 'ready', True))
            if not ready: raise StrictRestoreError('target did not publish ready state')
            return target, RestoreReceipt(self.request_id, plan.generation, plan.step, bytes_read, count, digest.hexdigest(), True, plan.controls)
        except BaseException:
            try:
                if hasattr(target, 'discard'): target.discard()
                elif hasattr(target, 'ready'): target.ready = False
            finally:
                raise
