"""Independent dense-array oracle for logical blocks crossing TP column shards."""
import math
import numpy as np
import pytest
from npu_nvme.experiments.graph_geometry import geometry, summarize


def tensor(name, shape, axis=None):
    shards = []
    for rank in range(4):
        start = [0] * len(shape)
        end = list(shape)
        if axis is not None:
            start[axis] = rank * shape[axis] // 4
            end[axis] = (rank + 1) * shape[axis] // 4
        shards.append(dict(rank=rank, start=start, end=end))
    local = [b - a for a, b in zip(shards[0]['start'], shards[0]['end'])]
    return dict(name=name, role='model', global_shape=shape, local_shape=local,
                partition='replicated' if axis is None else 'sharded', shards=shards)


@pytest.mark.parametrize('fraction', [.125, .25, .5, 1.])
def test_partial_scores_match_global_oracle(fraction):
    tensors = [tensor('layer.0.fc1', [32, 16], 0), tensor('layer.1.fc2', [16, 48], 1),
               tensor('layer.2.tail', [10, 20], 1), tensor('replica', [260]),
               tensor('small', [8])]
    schema = dict(tensors=tensors)
    values = {t['name']: (np.arange(math.prod(t['global_shape']), dtype=np.float64) % 13 - 6).reshape(t['global_shape']) for t in tensors}
    global_scores = {}
    for rank in range(4):
        rows = geometry(schema, rank, fraction, block_elements=64)
        assert {r['name'] for r in rows} == set(values) - {'small'}
        assert summarize(rows)['local_input_bytes'] == sum(r['elements'] for r in rows) * 8
        for row in rows:
            t = next(t for t in tensors if t['name'] == row['name'])
            shard = t['shards'][rank]
            local = values[t['name']][tuple(slice(a, b) for a, b in zip(shard['start'], shard['end']))].reshape(-1)
            if row['tile_indices']:
                tiles = local.reshape(-1, row['unit'])[row['tile_indices']]
                partial = np.zeros(row['block_count'])
                np.add.at(partial, row['segment_ids'], np.square(tiles).sum(axis=1))
                reconstructed = np.zeros((row['block_count'], 64))
                for tile, segment, slot in zip(tiles, row['segment_ids'], row['slot_ids']):
                    reconstructed[segment, slot * row['unit']:(slot + 1) * row['unit']] = tile
                assert np.count_nonzero(reconstructed) <= np.count_nonzero(tiles)
            else:
                partial = np.zeros(row['block_count'])
            for block, score in zip(row['global_blocks'], partial):
                key = (row['name'], block)
                global_scores[key] = global_scores.get(key, 0) + score
            if fraction == 1 and row['elements']:
                assert row['tile_indices'] == list(range(local.size // row['unit']))
    for (name, block), score in global_scores.items():
        expected = np.square(values[name].reshape(-1)[block * 64:(block + 1) * 64]).sum()
        assert score == expected


def test_rejects_unplanned_scan_fraction():
    with pytest.raises(ValueError): geometry(dict(tensors=[]), 0, .3)
