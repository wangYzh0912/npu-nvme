import importlib.util
from pathlib import Path
import subprocess


def test_archive_preserves_dirty_index_and_externalizes_payload(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args])
    git('init','-q');git('config','user.email','test@example.invalid');git('config','user.name','Test')
    (root/'code.py').write_text('old\n');git('add','.');git('commit','-qm','base')
    (root/'code.py').write_text('new\n');git('add','code.py')
    (root/'new.py').write_text('new source\n')
    (root/'.sudo_pw').write_text('private')
    (root/'array.npy').write_bytes(b'large payload')
    before=git('status','--porcelain=v1');index=git('diff','--cached');head=git('rev-parse','HEAD')
    script=Path(__file__).resolve().parents[2]/'tools/archive_worktree.py'
    spec=importlib.util.spec_from_file_location('archive_tool',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ref='refs/heads/archive/qwen-baseline-20260916/test'
    result=module.archive(root,tmp_path/'archive',ref)
    assert git('status','--porcelain=v1')==before
    assert git('diff','--cached')==index and git('rev-parse','HEAD')==head
    files=git('ls-tree','-r','--name-only',ref).decode().splitlines()
    assert files==['archive-manifest.json','code.py','new.py']
    assert git('show',ref+':code.py')==b'new\n'
    assert result['payload_bytes']==13
