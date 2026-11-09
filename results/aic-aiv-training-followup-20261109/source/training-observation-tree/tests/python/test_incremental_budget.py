from npu_nvme.experiments.budget import estimate


def test_weight_and_optimizer_physical_budgets():
    schema=dict(topology=dict(tp=4),tensors=[
        dict(name='w',role='model',partition='sharded',local_shape=[4],global_shape=[16],logical_bytes_per_rank=16),
        dict(name='n',role='model',partition='replicated',local_shape=[2],global_shape=[2],logical_bytes_per_rank=8),
        dict(name='adam_m.w',role='adam_m',partition='sharded',logical_bytes_per_rank=16)])
    result=estimate(schema,block_elements=8)
    assert result['weight_bytes']==72
    assert result['candidate_blocks']==2
    assert result['small_bytes']==8
    assert result['score_scan_bytes']==128
    assert result['full_state_scan_bytes']==320
