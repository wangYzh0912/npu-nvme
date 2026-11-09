"""Host dispatch fence for split forward/backward and optimizer graphs.

A checkpoint can overlap the following forward/backward. The optimizer graph
cannot be dispatched until the previous checkpoint has a successful terminal
receipt. This conservative boundary includes source-safe and durable completion.
Monolithic/sink training must not use this interface.
"""
import math
import threading
import time


class UpdateGuard:
    def __init__(self,*,timeout_seconds,synchronize):
        if not math.isfinite(timeout_seconds) or timeout_seconds<=0:raise ValueError('update deadline')
        self.timeout=timeout_seconds;self.synchronize=synchronize;self.lock=threading.RLock()
        self.pending=None;self.failed=False;self.closed=False;self.events=[];self._capturing=False
    def _available(self):
        if self.failed or self.closed:raise RuntimeError('optimizer update guard unavailable')
    def _wait(self):
        self._available()
        if self.pending is None:return
        start=time.monotonic_ns()
        try:
            receipt=self.pending.result(time.monotonic()+self.timeout)
            if getattr(receipt,'request_id',None)!=self.pending.request_id or getattr(receipt,'generation',None)!=self.pending.generation:
                raise ValueError('checkpoint receipt differs from borrowed generation')
        except BaseException:
            self.failed=True
            # Keep the handle and its source owners reachable after timeout.
            raise
        self.events.append(dict(kind='checkpoint_safe',request_id=self.pending.request_id,
                                begin_ns=start,end_ns=time.monotonic_ns()))
        self.pending=None
    def capture(self,submit):
        with self.lock:
            self._wait();self.synchronize();self._capturing=True
            try:
                handle=submit()
                if not callable(getattr(handle,'result',None)) or not getattr(handle,'request_id',None):
                    raise ValueError('capture did not return an owned checkpoint handle')
                self.pending=handle
                self.events.append(dict(kind='capture',request_id=handle.request_id,end_ns=time.monotonic_ns()))
                return handle
            except BaseException:self.failed=True;raise
            finally:self._capturing=False
    def before_optimizer_update(self):
        with self.lock:self._wait()
    def update(self,optimizer_call):
        with self.lock:
            self._wait()
            self.events.append(dict(kind='optimizer_dispatch',begin_ns=time.monotonic_ns()))
            try:
                value=optimizer_call();self.synchronize()
                return value
            except BaseException:self.failed=True;raise
    def close(self):
        with self.lock:
            self._wait();self.closed=True
