import json

import pytest

from npu_nvme.runtime import training_catalog as catalog


def checkpoint(root, step):
    catalog.initialize(root, identity={'model': 'test'}, method='mindspore_native_save')
    path, reservation = catalog.reserve(root, run_id='source', step=step)
    for rank in range(2):
        catalog.write_checked(path / f'rank_{rank}/complete.json', dict(
            rank=rank, generation=reservation['generation'], step=step,
            method=reservation['method'], identity=reservation['identity'],
            state={'sha256': 'test'}, controls={'step': step}))
    catalog.publish(path, world=2)
    return path


def test_generation_is_not_step_and_unpublished_is_not_selected(tmp_path):
    checkpoint(tmp_path, 4)
    catalog.reserve(tmp_path, run_id='failed', step=8)
    with catalog.selected(tmp_path) as (_, value):
        assert (value['generation'], value['step']) == (1, 4)
    with pytest.raises(FileNotFoundError):
        with catalog.selected(tmp_path, 2):
            pass


def test_latest_falls_back_but_explicit_corruption_fails(tmp_path):
    checkpoint(tmp_path, 4)
    broken = checkpoint(tmp_path, 8)
    document = json.loads((broken / 'checkpoint.json').read_text())
    document['payload']['step'] = 9
    (broken / 'checkpoint.json').write_text(json.dumps(document))
    rejected = []
    with catalog.selected(tmp_path, rejected=rejected) as (_, value):
        assert value['step'] == 4
    assert len(rejected) == 1
    with pytest.raises(ValueError, match='digest'):
        with catalog.selected(tmp_path, 2):
            pass


def test_media_validation_and_reader_pin(tmp_path):
    first = checkpoint(tmp_path, 4)
    checkpoint(tmp_path, 8)
    def media_check(path, value):
        if value['generation'] == 2:
            raise ValueError('media generation unavailable')
    with catalog.selected(tmp_path, validate=media_check) as (path, _):
        assert path == first
        assert catalog.prune(tmp_path, 1) == []
        assert first.exists()
    assert catalog.prune(tmp_path, 1) == [1]
    assert not first.exists()


def test_partial_rank_completion_cannot_publish(tmp_path):
    catalog.initialize(tmp_path, identity={}, method='ours')
    path, _ = catalog.reserve(tmp_path, run_id='partial', step=4)
    with pytest.raises(FileNotFoundError):
        catalog.publish(path, world=2)
    assert not (path / 'checkpoint.json').exists()


def test_payload_corruption_and_escape(tmp_path):
    (tmp_path / 'weights').write_bytes(b'weights')
    files = catalog.payload_files(tmp_path)
    catalog.verify_files(tmp_path, files)
    (tmp_path / 'weights').write_bytes(b'changed')
    with pytest.raises(ValueError, match='digest'):
        catalog.verify_files(tmp_path, files)
    files[0]['path'] = '../outside'
    with pytest.raises(ValueError, match='escaping'):
        catalog.verify_files(tmp_path, files)


def test_lineage_cannot_change_method_or_identity(tmp_path):
    catalog.initialize(tmp_path, identity={'seed': 42}, method='ours')
    with pytest.raises(ValueError, match='identity/method'):
        catalog.initialize(tmp_path, identity={'seed': 43}, method='ours')
