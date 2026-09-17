"""Sampled, independent FP64 CPU score oracle; strictly outside timed work."""
import numpy as np


def check_scores(ms, chain, level):
    from npu_nvme.experiments.device_score import local_graph
    actual = chain.scores.asnumpy()
    if level == 1:
        with local_graph(ms):
            sample = ms.ops.reshape(chain.first, (-1,))[:16].asnumpy()
        expected = np.asarray([sample.astype(np.float64).sum()])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)
        return dict(status='pass', indices=[0], expected=expected.tolist(), actual=actual.tolist())
    indices = sorted(set([0, len(actual) // 2, len(actual) - 1] +
                         ([int(chain.indices.asnumpy()[0]), int(chain.indices.asnumpy()[-1])] if level >= 6 else [])))
    local = []
    with local_graph(ms):
        for score_index in indices:
            i, row = next((i, r) for i, r in enumerate(chain.rows)
                          if r['score_offset'] <= score_index < r['score_offset'] + r['block_count'])
            segment = score_index - row['score_offset']
            selected = [tile for tile, sid in zip(row['tile_indices'], row['segment_ids']) if sid == segment]
            if not selected:
                local.append(0.0); continue
            index_tensor = ms.Tensor(np.asarray(selected, np.int32))
            w = ms.ops.gather(ms.ops.reshape(chain.sources[i], (-1, row['unit'])), index_tensor, 0).asnumpy()
            r = ms.ops.gather(ms.ops.reshape(chain.references[i], (-1, row['unit'])), index_tensor, 0).asnumpy()
            if level == 2:
                value = w.astype(np.float64).sum() + r.astype(np.float64).sum()
            else:
                # Difference itself is FP32 in the tested algorithm; FP64 reduction oracle.
                difference = (w - r).astype(np.float64)
                value = (difference ** 2).sum() if level >= 4 else difference.sum()
            local.append(float(value))
        if level >= 5:
            gathered = ms.ops.AllGather(group='graph_topk_aux_tp4')(ms.Tensor(np.asarray(local, np.float32))).asnumpy()
            expected = gathered.reshape(4, -1).astype(np.float64).sum(axis=0)
        else:
            expected = np.asarray(local)
    observed = actual[indices].astype(np.float64)
    np.testing.assert_allclose(observed, expected, rtol=3e-5, atol=2e-4 if level < 4 else 1e-6)
    return dict(status='pass', indices=indices, expected=expected.tolist(), actual=observed.tolist(),
                max_absolute_error=float(np.max(np.abs(observed - expected))),
                oracle='CPU FP64 summation of real TP fragments; independent of device segment reduction')
