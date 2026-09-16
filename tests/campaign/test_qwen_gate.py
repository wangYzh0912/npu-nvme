import pytest
from tools.validate_qwen_baselines import validate,METHODS

def test_primary_requires_all_four_methods():
    with pytest.raises(ValueError,match='four methods'):validate({'methods':{'ours':[]}})

def test_primary_rejects_one_pilot_per_method():
    with pytest.raises(ValueError,match='three independent'):validate({'methods':{name:[{}] for name in METHODS}})


def test_formal_campaign_has_independent_sources_and_timing(tmp_path):
    from tools.prepare_qwen_campaign import prepare
    from tools.run_campaign import validate_plan
    plan,manifest=prepare(tmp_path,tmp_path/'out',tmp_path/'env.json')
    validate_plan(plan)
    assert len(plan['stages'])==25
    for method,groups in manifest['methods'].items():
        assert len(groups)==3
        assert len({g['source'] for g in groups})==3
        if method=='none':assert not any(g['restores'] for g in groups)
        else:
            timing=manifest['timing'][method]
            assert {timing['warmup'],*timing['measured']}=={r for g in groups for r in g['restores']}
