import pytest
from tools.validate_qwen_baselines import validate,METHODS

def test_primary_requires_all_four_methods():
    with pytest.raises(ValueError,match='four methods'):validate({'methods':{'ours':[]}})

def test_primary_rejects_one_pilot_per_method():
    with pytest.raises(ValueError,match='three independent'):validate({'methods':{name:[{}] for name in METHODS}})
