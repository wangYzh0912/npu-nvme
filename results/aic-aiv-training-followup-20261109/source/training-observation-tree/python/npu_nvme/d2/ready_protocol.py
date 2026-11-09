"""Collective restore readiness: all rank proofs precede any release message."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading

class ReadyDecision:
    def __init__(self,*,world_size,epoch,generation,manifest_sha256):
        if world_size not in (2,4) or not epoch or type(generation) is not int or generation<=0:raise ValueError('ready identity')
        self.world_size=world_size;self.identity=dict(epoch=epoch,generation=generation,manifest_sha256=manifest_sha256)
        self.proofs={};self.failed=False;self.lock=threading.RLock()
    def prepare(self,rank,proof):
        with self.lock:
            if self.failed:raise RuntimeError('ready decision failed')
            if type(rank) is not int or not 0<=rank<self.world_size or proof.get('rank')!=rank:raise ValueError('rank identity')
            if any(proof.get(k)!=v for k,v in self.identity.items()):raise ValueError('restore identity differs')
            if proof.get('transport_safe') is not True or proof.get('schema_verified') is not True or proof.get('controls_verified') is not True:raise ValueError('rank not locally verified')
            if rank in self.proofs and self.proofs[rank]!=proof:raise ValueError('conflicting readiness proof')
            self.proofs[rank]=dict(proof)
    def decision(self):
        with self.lock:
            if self.failed or len(self.proofs)!=self.world_size:raise RuntimeError('not globally ready')
            value=dict(self.identity,world_size=self.world_size,ranks=sorted(self.proofs))
            raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
            return dict(value,token=hashlib.sha256(raw).hexdigest())
    def abort(self):
        with self.lock:self.failed=True


def coordinate(connections,decision,wire,*,deadline):
    if set(connections)!=set(range(decision.world_size)):raise ValueError('connection set')
    def receive(rank,sock):
        control,payload=wire.receive(sock,deadline=deadline,max_payload=0)
        if payload or control.get('kind')!='prepared':raise ValueError('expected prepared rank')
        decision.prepare(rank,control)
    try:
        with ThreadPoolExecutor(max_workers=decision.world_size) as executor:
            futures=[executor.submit(receive,r,s) for r,s in connections.items()]
            for f in futures:f.result()
        value=decision.decision()
        # No rank receives a training release before all local proofs exist.
        for rank,sock in connections.items():
            wire.send(sock,dict(kind='release',**value),b'',deadline=deadline,max_payload=0)
        return value
    except BaseException:
        decision.abort()
        # Caller retains any unknown-DMA targets/leases. It must not use a
        # failed collective as permission to free rank or primary buffers.
        raise
