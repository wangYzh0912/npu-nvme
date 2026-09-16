"""Measure serial pack and D2H stream intervals without adding synchronizations."""
import ctypes as C

class CaptureTiming:
    def __init__(self, acl, stream, chunks):
        self.acl=acl;self.stream=stream;self.events=[];self.used=0
        signatures={
            'aclrtCreateEventWithFlag':[C.POINTER(C.c_void_p),C.c_uint32],
            'aclrtRecordEvent':[C.c_void_p,C.c_void_p],
            'aclrtDestroyEvent':[C.c_void_p],
            'aclrtEventElapsedTime':[C.POINTER(C.c_float),C.c_void_p,C.c_void_p],
        }
        for name,args in signatures.items():
            fn=getattr(acl,name);fn.argtypes=args;fn.restype=C.c_int
        for _ in range(chunks*3):
            event=C.c_void_p();self.check(acl.aclrtCreateEventWithFlag(C.byref(event),8));self.events.append(event)

    @staticmethod
    def check(rc):
        if rc:raise RuntimeError('capture timing ACL failure: '+str(rc))

    def reset(self):self.used=0

    def mark(self):
        if self.used>=len(self.events):raise RuntimeError('capture event capacity exceeded')
        self.check(self.acl.aclrtRecordEvent(self.events[self.used],self.stream));self.used+=1

    def result(self):
        if self.used%3:raise ValueError('incomplete capture interval')
        totals=[0.,0.]
        for i in range(0,self.used,3):
            for j in range(2):
                value=C.c_float();self.check(self.acl.aclrtEventElapsedTime(C.byref(value),self.events[i+j],self.events[i+j+1]))
                totals[j]+=value.value*1e6
        return dict(pack_stream_ns=round(totals[0]),d2h_stream_ns=round(totals[1]),
                    capture_chunks=self.used//3,
                    capture_clock='ACL event elapsed time; stream intervals may include host submission gaps')

    def close(self):
        for event in self.events:self.check(self.acl.aclrtDestroyEvent(event))
        self.events=[]
