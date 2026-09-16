"""Explicit single-rank blocking R0 training adapter with Host budget admission."""
import base64
import numpy as np
from incremental_manifest import build_training_state_manifest
from training_state import encode_control_value,decode_control_value
from npu_nvme.d2.incremental import PersistentR0


class IncrementalState:
    def __init__(self,framework,components,region,*,host_budget_bytes,block_elements=65536):
        self.framework=framework;self.components=components;self.ready=False
        self.manifest=build_training_state_manifest(components,block_size=block_elements,small_threshold=1024)
        state_bytes=sum(f.byte_count for f in self.manifest.fields)
        # The current CPU exact-replacement oracle owns several whole-state
        # copies and decoded frames. Admit this explicitly; no Qwen claim.
        if type(host_budget_bytes) is not int or state_bytes*16>host_budget_bytes:
            raise MemoryError('R0 CPU oracle and bounded chain exceed Host budget')
        self.store=PersistentR0(region,self.manifest,chunk_bytes=1<<20,max_chain_length=2)
        self.registry={f'{component}/{name}':p for component,obj in components.items() for name,p in obj.parameters_and_names()}
        self.state_bytes=state_bytes
    def snapshot(self):
        self.framework.hal.synchronize()
        result={}
        for field in self.manifest.fields:
            array=np.ascontiguousarray(self.registry[field.canonical_name].asnumpy())
            if array.shape!=field.shape or array.dtype!=np.dtype(field.dtype):raise ValueError('R0 runtime state geometry changed')
            result[field.canonical_name]=array.copy()
        return result
    @staticmethod
    def encode_controls(controls):
        result={}
        for name,value in controls.items():
            payload,meta=encode_control_value(value)
            result[name]=dict(meta,data=base64.b64encode(payload).decode())
        return result
    @staticmethod
    def decode_controls(controls):
        return {name:decode_control_value(np.frombuffer(base64.b64decode(entry['data'],validate=True),np.uint8),entry) for name,entry in controls.items()}
    def save(self,controls,*,step):
        return self.store.save(self.snapshot(),self.encode_controls(controls),step=step)
    def restore(self,apply_controls,verify,*,generation=None):
        if self.ready:raise RuntimeError('R0 restore requires a fresh unready target')
        recovered=self.store.recover(generation)
        # Validate every target before assigning any parameter.
        for field in self.manifest.fields:
            value=recovered['state'][field.canonical_name];parameter=self.registry[field.canonical_name]
            if tuple(parameter.shape)!=tuple(value.shape) or np.dtype(self.framework.dtype_to_nptype(parameter.dtype))!=value.dtype:
                raise ValueError('R0 target schema changed')
        for field in self.manifest.fields:
            parameter=self.registry[field.canonical_name]
            parameter.set_data(self.framework.Tensor(recovered['state'][field.canonical_name],dtype=parameter.dtype))
        controls=self.decode_controls(recovered['controls']);apply_controls(controls,recovered['step'])
        self.framework.hal.synchronize();actual=self.snapshot()
        if any(actual[k].tobytes()!=v.tobytes() for k,v in recovered['state'].items()):raise ValueError('R0 device readback differs')
        verify(controls,recovered['step']);self.ready=True
        return recovered
