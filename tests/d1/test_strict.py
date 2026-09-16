import hashlib, pytest
from npu_nvme.runtime.d1_restore import StrictRestoreSession, StrictRestoreError

def record(raw=b'abcd'*1024):
 d=hashlib.sha256(raw).hexdigest(); return {'strict_contract':'D1','state_step':4,'generation':7,'identity':{'model':'x'},'params':{'model/x':{'offset':4096,'size':len(raw),'shape':[len(raw)],'dtype':'uint8','chunks':[{'offset':4096,'size':len(raw),'sha256':d}],'sha256':d}},'controls':['rng']}
class T:
 def __init__(self): self.ready=False; self.discarded=False; self.data=[]
 def apply_chunk(self,n,o,d): self.data.append(d)
 def mark_ready(self): self.ready=True
 def discard(self): self.discarded=True; self.ready=False

def test_strict_restore_success_and_failure():
 raw=b'abcd'*1024; t=T(); out=StrictRestoreSession(select_record=lambda s:record(raw),reader=lambda o,n:raw,request_id='r').restore_full_state(lambda spec:t,{'identity':{'model':'x'},'parameters':['model/x']},4)
 assert out[0].ready and out[1].ready and out[1].chunks==1
 bad=record(raw); bad['params']['model/x']['chunks'][0]['sha256']='0'*64; t2=T()
 with pytest.raises(StrictRestoreError): StrictRestoreSession(select_record=lambda s:bad,reader=lambda o,n:raw).restore_full_state(lambda spec:t2,{'identity':{'model':'x'},'parameters':['model/x']},4)
 assert t2.discarded

def test_strict_restore_discards_after_late_failure():
 raw=b'abcd'*1024; t=T(); bad=record(raw); bad['params']['model/x']['sha256']='0'*64
 with pytest.raises(StrictRestoreError): StrictRestoreSession(select_record=lambda s:bad,reader=lambda o,n:raw).restore_full_state(lambda spec:t,{'identity':{'model':'x'},'parameters':['model/x']},4)
 assert t.discarded and not t.ready
