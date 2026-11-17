import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from npu_nvme.cli import qwen_training


def test_non_profile_run_keeps_supervisor_identity_and_environment(tmp_path):
    env={'ASCEND_HOME_PATH':str(tmp_path),'HOME':'/root'}
    identity,child_env=qwen_training.profiler_worker_identity(
        {'io_experiment':{'device_profile':False}},env,tmp_path)
    assert identity is None
    assert child_env is env


def test_profiler_worker_drops_to_cann_owner_and_prepares_writable_output(monkeypatch,tmp_path):
    toolkit=tmp_path/'toolkit';toolkit.mkdir()
    output=tmp_path/'run';output.mkdir(mode=0o755)
    env={'ASCEND_HOME_PATH':str(toolkit),'HOME':'/root','PATH':'/usr/bin'}
    chowns=[]
    original_stat=Path.stat
    monkeypatch.setattr(Path,'stat',lambda self:SimpleNamespace(st_uid=22633)
        if self==toolkit else original_stat(self))
    monkeypatch.setattr(qwen_training.os,'geteuid',lambda:0)
    monkeypatch.setattr(qwen_training.os,'chown',lambda *args:chowns.append(args))
    monkeypatch.setattr(qwen_training.pwd,'getpwuid',lambda uid:SimpleNamespace(
        pw_uid=uid,pw_gid=22633,pw_name='user7',pw_dir='/home/user7'))
    monkeypatch.setattr(qwen_training.os,'getgrouplist',lambda user,gid:[gid,42])
    identity,child_env=qwen_training.profiler_worker_identity(
        {'io_experiment':{'device_profile':True}},env,output)
    assert identity==dict(user=22633,group=22633,extra_groups=[22633,42])
    assert child_env==dict(env,HOME='/home/user7',USER='user7',LOGNAME='user7')
    assert chowns==[(output,22633,22633)]


def test_non_root_cannot_silently_profile_as_wrong_cann_owner(monkeypatch,tmp_path):
    toolkit=tmp_path/'toolkit';toolkit.mkdir()
    original_stat=Path.stat
    monkeypatch.setattr(Path,'stat',lambda self:SimpleNamespace(st_uid=22633)
        if self==toolkit else original_stat(self))
    monkeypatch.setattr(qwen_training.os,'geteuid',lambda:1000)
    with pytest.raises(PermissionError,match='selected CANN owner'):
        qwen_training.profiler_worker_identity(
            {'io_experiment':{'device_profile':True}},
            {'ASCEND_HOME_PATH':str(toolkit)},tmp_path)
