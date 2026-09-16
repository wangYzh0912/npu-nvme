"""Experimental guarded live FULL using the versioned asynchronous transport.

Use only with split optimizer dispatch through UpdateGuard. The default strict
frozen entry and its advertised capabilities remain unchanged.
"""
import numpy as np
from npu_nvme.strict_checkpoint import StrictCheckpoint


class LiveCheckpoint:
    def __init__(self,guard,**options):
        self.guard=guard;self.store=StrictCheckpoint(**options)
        self.store._runtime.freeze=self._borrow
        self.store._runtime.release=lambda items:items.clear()
    def _borrow(self,params,generation):
        if not self.guard._capturing:raise RuntimeError('live capture requires update guard admission')
        items=[]
        for value in params:
            item=dict(value)
            if item.get('placement')!='host' and item.get('ptr'):
                item['borrowed_device']=True
            else:
                item['np_arr']=np.ascontiguousarray(item['np_arr']).copy()
                item['ptr']=item['np_arr'].ctypes.data
            items.append(item)
        return items
    def save_state(self,*args,**kwargs):
        return self.guard.capture(lambda:self.store.save_state(*args,**kwargs))
    def restore_full_state(self,*args,**kwargs):
        self.guard.before_optimizer_update()
        return self.store.restore_full_state(*args,**kwargs)
    def close(self,timeout=120):
        try:self.guard.close()
        except BaseException:
            # Failed training admission does not prevent a positive transport
            # drain. If drain fails, StrictCheckpoint retains native owners.
            self.store.close(timeout)
            raise
        self.store.close(timeout)
