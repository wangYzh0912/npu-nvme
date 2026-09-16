import json
import pytest
from tools.run_qwen_method import collect_restore_timing


def test_global_timing_covers_staggered_ranks(tmp_path):
    for rank in range(4):
        root=tmp_path/f'rank_{rank}';root.mkdir()
        (root/'restore-timing.json').write_text(json.dumps(dict(begin_ns=100+rank*10,
            global_ready_ns=1000+rank*100,boundary='restore_begin_to_global_ready',mandatory_integrity=True)))
    collect_restore_timing(tmp_path)
    assert json.loads((tmp_path/'timing.json').read_text())['seconds']==1200/1e9
    row=tmp_path/'rank_3/restore-timing.json';data=json.loads(row.read_text());data['mandatory_integrity']=False
    row.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='rank restore timing'):collect_restore_timing(tmp_path)
