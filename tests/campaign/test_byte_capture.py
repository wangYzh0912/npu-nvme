from types import SimpleNamespace
import ctypes
import numpy as np
from incremental_manifest import build_training_state_manifest
from npu_nvme.framework.byte_capture import ByteCapture

class Array:
    def __init__(self,value,requires_grad=False):
        self.value=np.asarray(value.value if isinstance(value,Array) else value).copy()
        self.shape=self.value.shape;self.dtype=self.value.dtype
    def __getitem__(self,key):return Array(self.value[key])
    def asnumpy(self):return self.value.copy()

class Copy:
    def __call__(self,dst,size,src,length,kind):
        assert kind==3 and size==length
        ctypes.memmove(dst,src,size);return 0

def test_bytewise_capture_and_ack_reference(monkeypatch):
    value=Array(np.asarray([0,0x7fc00001,0x3f800000],np.uint32).view(np.float32))
    obj=SimpleNamespace(parameters_and_names=lambda:[('bits',value)])
    manifest=build_training_state_manifest({'model':obj},block_size=1,small_threshold=0)
    def assign(param,other):param.value[:]=other.value
    ops=SimpleNamespace(zeros=lambda shape,dtype:Array(np.zeros(shape,dtype)),assign=assign,
        not_equal=lambda a,b:Array(a.value!=b.value),any=lambda a:Array(np.any(a.value)))
    framework=SimpleNamespace(ops=ops,Tensor=Array,Parameter=Array,uint8=np.uint8,hal=SimpleNamespace(synchronize=lambda:None))
    monkeypatch.setattr('npu_nvme.framework.byte_capture.get_dev_ptr',lambda p:p.value.ctypes.data)
    capture=ByteCapture(framework,SimpleNamespace(aclrtMemcpy=Copy()),manifest,{'model/bits':value},hbm_budget_bytes=1024)
    capture.capture();capture.commit_ack()
    value.value.view(np.uint32)[:]=[0x80000000,0x7fc00002,0x3f800000]
    assert capture.capture().tolist()==[True,True,False]
    assert capture.capture().tolist()==[True,True,False]
    capture.commit_ack();assert capture.capture().tolist()==[False,False,False]
