from types import SimpleNamespace

import numpy as np
import pytest

from npu_nvme.experiments.initial_state import capture, restore


class Parameter:
    def __init__(self, name, array):
        self.name = name
        self.array = np.array(array)
        self.dtype = self.array.dtype

    @property
    def shape(self):
        return self.array.shape

    def asnumpy(self):
        return self.array

    def set_data(self, value):
        self.array = np.array(value)


@pytest.fixture
def state(monkeypatch):
    import qwen_native_state
    monkeypatch.setattr(qwen_native_state, 'capture_control', lambda *a, **k: {'step': 0})
    monkeypatch.setattr(qwen_native_state, 'restore_control', lambda *a, **k: None)
    params = [Parameter('w', np.arange(5, dtype=np.float32)),
              Parameter('adam_m.w', np.zeros(5, np.float32)),
              Parameter('adam_v.w', np.zeros(5, np.float32)),
              Parameter('global_step', np.array(0, np.int64))]
    network = SimpleNamespace(parameters_and_names=lambda: [(p.name, p) for p in params])
    ms = SimpleNamespace(runtime=SimpleNamespace(synchronize=lambda: None),
                         Tensor=lambda value, dtype: np.array(value, dtype=dtype))
    return ms, network, params


def test_full_reset_restores_adam_and_scalar_shape(state, tmp_path):
    ms, network, params = state
    folder = tmp_path / 'full'
    before = [p.array.copy() for p in params]
    manifest = capture(ms, network, folder, identity='same', data_sha256='data', horizon=20)
    assert manifest['tensors']['global_step']['shape'] == []
    for parameter in params:
        parameter.array += 1
    result = restore(ms, network, folder, identity='same')
    assert result['verified'] and result['tensors'] == 4
    for parameter, original in zip(params, before):
        np.testing.assert_array_equal(parameter.array, original)
        assert parameter.shape == original.shape


def test_restore_rejects_corrupt_tensor(state, tmp_path):
    ms, network, _ = state
    folder = tmp_path / 'full'
    manifest = capture(ms, network, folder, identity='same', data_sha256='data', horizon=20)
    entry = manifest['tensors']['w']
    np.save(folder / entry['file'], np.ones(5, np.float32))
    with pytest.raises(ValueError, match='integrity'):
        restore(ms, network, folder, identity='same')
