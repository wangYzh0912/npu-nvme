from types import SimpleNamespace
import threading
import time
import pytest
from npu_nvme.framework.update_guard import UpdateGuard

class Handle:
    request_id='request';generation=7
    def __init__(self):self.done=threading.Event();self.receipt=SimpleNamespace(request_id=self.request_id,generation=self.generation)
    def result(self,deadline):
        if not self.done.wait(max(0,deadline-time.monotonic())):raise TimeoutError('not source-safe')
        return self.receipt

def test_optimizer_dispatch_waits_until_checkpoint_terminal():
    guard=UpdateGuard(timeout_seconds=1,synchronize=lambda:None);handle=Handle();guard.capture(lambda:handle)
    called=threading.Event();thread=threading.Thread(target=lambda:guard.update(called.set));thread.start()
    assert not called.wait(.02)
    handle.done.set();thread.join(1);assert called.is_set() and guard.pending is None
    assert [e['kind'] for e in guard.events]==['capture','checkpoint_safe','optimizer_dispatch']

def test_timeout_retains_handle_and_blocks_all_later_updates():
    guard=UpdateGuard(timeout_seconds=.01,synchronize=lambda:None);handle=Handle();guard.capture(lambda:handle)
    called=[]
    with pytest.raises(TimeoutError):guard.update(lambda:called.append(True))
    assert guard.pending is handle and guard.failed and not called
    handle.done.set()
    with pytest.raises(RuntimeError):guard.update(lambda:called.append(True))

def test_foreign_receipt_blocks_dispatch():
    guard=UpdateGuard(timeout_seconds=1,synchronize=lambda:None);handle=Handle();handle.receipt.generation=8;handle.done.set();guard.capture(lambda:handle)
    with pytest.raises(ValueError):guard.update(lambda:pytest.fail('must not dispatch'))
    assert guard.failed and guard.pending is handle

def test_capture_serializes_with_optimizer_execution():
    entered=threading.Event();release=threading.Event();captured=threading.Event()
    guard=UpdateGuard(timeout_seconds=1,synchronize=lambda:None)
    def update():entered.set();release.wait(1)
    updater=threading.Thread(target=lambda:guard.update(update));updater.start();assert entered.wait(1)
    handle=Handle()
    def submit():captured.set();return handle
    capturer=threading.Thread(target=lambda:guard.capture(submit));capturer.start()
    assert not captured.wait(.02);release.set();updater.join(1);capturer.join(1);assert captured.is_set()
