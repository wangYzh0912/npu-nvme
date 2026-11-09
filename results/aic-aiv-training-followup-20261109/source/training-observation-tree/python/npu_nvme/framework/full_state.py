"""MindSpore FULL capture specification and fresh training target ownership."""
import ctypes
import hashlib
import numpy as np
from npu_nvme.runtime.d1_schema import spec_validate, canonical, CONTROL_BYTES
from .capture import FrozenCapture
from .parameters import get_dev_ptr
from training_state import (encode_control_value, decode_control_value,
                            restore_training_controls, capture_training_controls)


def training_spec(components, controls, identity, *, framework, acl, npu=7, scheduler=None):
    capture = FrozenCapture(rank_id=0, device_id=npu, framework=framework, acl=acl, pointer_of=get_dev_ptr)
    params = capture.prepare_state_components(components, with_checksums=False)
    spec = dict(identity=identity, parameters={p['name']:{k:p[k] for k in ('shape','dtype','size')} for p in params},
                control_names=sorted(controls), applicability={k:'required' for k in
                    ('model','optimizer','rng','data_cursor','loss_scale')})
    spec['applicability']['scheduler'] = 'required' if scheduler is not None else 'not_applicable:fixed learning rate'
    return spec_validate(spec)


def prepare_full(capture, components, controls, spec):
    if set(components) != {'model', 'optimizer'} or set(controls) != set(spec['control_names']):
        raise ValueError('FULL requires the declared model, optimizer and controls')
    capture.synchronize()
    params = capture.prepare_state_components(components, with_checksums=False)
    actual = {p['name']:{k:p[k] for k in ('shape','dtype','size')} for p in params}
    if canonical(actual) != canonical(spec['parameters']):
        raise ValueError('source tensor set/shape/dtype differs from specification')
    control_bytes = 0
    for name in sorted(controls):
        payload, meta = encode_control_value(controls[name])
        control_bytes += payload.nbytes
        if control_bytes > CONTROL_BYTES: raise ValueError('control byte budget exceeded')
        params.append(dict(name='control/'+name, placement='host', ptr=0, size=payload.nbytes,
                           shape=[payload.nbytes], dtype='uint8', np_arr=payload,
                           param_ref=None, codec=meta['codec']))
    capture.validate(params, 1 << 20)
    return params


class MindSporeRestoreTarget:
    """No training components are exposed until all restore checks have passed.

    The factory may privately materialize/warm up its new graph before handing
    this unready object to the session. It never receives an existing target.
    """
    def __init__(self, *, framework, acl, npu, model, optimizer, cell, identity, scheduler=None):
        self.ms, self.acl, self.npu = framework, acl, npu
        self._components = {'model':model, 'optimizer':optimizer}
        self._cell, self._scheduler = cell, scheduler
        self.identity = identity
        self._token = None
        self._verified = False
        self._controls = {}
        self._control_bytes = {}
        self._params = {}
        self.discarded = False
        self._unsafe = []
        self._copy_executor = None

    @property
    def ready(self): return self._token is not None
    @ready.setter
    def ready(self, value):
        if value is not False: raise RuntimeError('only the restore protocol may publish ready')
        self._token = None

    @property
    def transport_safe(self): return not self._unsafe

    def _require_ready(self):
        if not self.ready or self.discarded: raise RuntimeError('training target is not ready')

    @property
    def model(self): self._require_ready(); return self._components['model']
    @property
    def optimizer(self): self._require_ready(); return self._components['optimizer']
    @property
    def cell(self): self._require_ready(); return self._cell
    @property
    def controls(self): self._require_ready(); return self._controls
    def train_step(self, *args): self._require_ready(); return self._cell(*args)

    def prepare_restore(self, spec):
        if self.discarded or self.ready: raise RuntimeError('restore requires a fresh unready target')
        if canonical(self.identity) != canonical(spec['identity']): raise ValueError('target training identity differs')
        capture = FrozenCapture(rank_id=0, device_id=self.npu, framework=self.ms,
                                acl=self.acl, pointer_of=get_dev_ptr)
        params = capture.prepare_state_components(self._components, with_checksums=False)
        actual = {p['name']:{k:p[k] for k in ('shape','dtype','size')} for p in params}
        if canonical(actual) != canonical(spec['parameters']):
            raise ValueError('new target tensor geometry differs from expected specification')
        self._params = {p['name']:p for p in params}
        if self.acl.aclrtSetDevice(self.npu) != 0: raise RuntimeError('cannot bind restore device')

    def set_copy_executor(self, executor):
        if self.ready or self.discarded or not callable(executor):
            raise RuntimeError('invalid restore copy executor')
        self._copy_executor=executor

    def apply_chunk(self, name, offset, data):
        if name.startswith('control/'):
            buffer = self._control_bytes.setdefault(name[8:], bytearray())
            if offset != len(buffer): raise ValueError('non-contiguous control chunk')
            if sum(map(len, self._control_bytes.values())) + len(data) > CONTROL_BYTES:
                raise ValueError('control budget exceeded')
            buffer.extend(data)
            return
        item = self._params[name]
        if offset < 0 or offset + len(data) > item['size']: raise ValueError('chunk exceeds target')
        if item['ptr']:
            if self._copy_executor is None: raise RuntimeError('restore transport is not configured')
            try:
                self._copy_executor(item['ptr']+offset,data,owner=self)
            except BaseException as error:
                if not getattr(error,'transport_safe',False): self._unsafe.append(item)
                raise
        else:
            memoryview(item['np_arr']).cast('B')[offset:offset+len(data)] = data

    def finish_restore(self, spec):
        if self._unsafe: raise RuntimeError('restore DMA stop is not proven')
        for item in self._params.values():
            if not item['ptr']:
                parameter = item['param_ref']
                self.ms.ops.assign(parameter, self.ms.Tensor(item['np_arr'], dtype=parameter.dtype))
        self.ms.hal.synchronize()
        if set(self._control_bytes) != set(spec['control_names']): raise ValueError('control set differs')
        for name, raw in self._control_bytes.items():
            self._controls[name] = decode_control_value(np.frombuffer(raw, dtype=np.uint8),
                dict(codec='json-tagged-v1', sha256=hashlib.sha256(raw).hexdigest()))
        optimizer = self._components['optimizer']
        if not np.array_equal(optimizer.global_step.asnumpy(), self._controls['global_step']):
            raise ValueError('optimizer step differs from control step')
        restore_training_controls(self.ms, optimizer, self._controls, self._scheduler)
        self._cursor = self._controls['data_cursor']
        self._loss_scale = self._controls['loss_scale']
        if hasattr(self._cell, 'loss_scale_sense'):
            actual = np.asarray(self._cell.loss_scale_sense.asnumpy()).reshape(())
            if not np.array_equal(actual, np.asarray(self._loss_scale).reshape(())):
                raise ValueError('training cell loss scale differs from restored controls')
        elif hasattr(self._cell, 'checkpoint_loss_scale'):
            if not np.array_equal(np.asarray(self._cell.checkpoint_loss_scale), np.asarray(self._loss_scale)):
                raise ValueError('training cell loss scale differs from restored controls')
        else:
            raise ValueError('training cell must expose its actual loss scale for strict restore')

    def verify_controls(self, spec):
        if self.ms.get_seed() != self._controls['mindspore_seed']:
            return False
        actual = capture_training_controls(self.ms, self._components['optimizer'], self._cursor,
            self._loss_scale, self._controls['mindspore_seed'], self._scheduler)
        self._verified = set(actual) == set(spec['control_names']) and all(
            encode_control_value(actual[k])[0].tobytes() == encode_control_value(self._controls[k])[0].tobytes()
            for k in actual)
        if not self._verified:
            mismatches = [k for k in actual if encode_control_value(actual[k])[0].tobytes() !=
                          encode_control_value(self._controls[k])[0].tobytes()]
            raise ValueError('control readback mismatch: ' + ', '.join(mismatches))
        return self._verified

    def mark_ready(self):
        if not self._verified or self._unsafe or self.discarded: raise RuntimeError('restore is not verified')
        self._token = object()

    def discard(self):
        self.ready = False
        self.discarded = True
        if self._unsafe: raise RuntimeError('target retains unproven H2D buffers')
        self._components = {}; self._cell = None
        self._params.clear(); self._control_bytes.clear(); self._controls.clear()
