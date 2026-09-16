import pytest

from npu_nvme.runtime.training_catalog import write_checked
from npu_nvme.runtime.training_validation import compare


def run(root,loss=1.0):
    for rank in range(4):
        path=root/f'rank_{rank}';path.mkdir(parents=True)
        write_checked(path/'training.json',dict(status='pass',final_step=4,
            losses=[dict(step=4,loss=loss,overflow=False)]))
        write_checked(path/'state-step-4.json',{'sha256':'state'})
        write_checked(path/'control-step-4.json',{'step':4})


def test_exact_state_and_tolerant_loss(tmp_path):
    oracle=tmp_path/'oracle';actual=tmp_path/'actual';run(oracle);run(actual,1.000001)
    assert compare(oracle,actual)['status']=='pass'
    run(tmp_path/'bad',1.1)
    with pytest.raises(ValueError,match='loss'):
        compare(oracle,tmp_path/'bad')
