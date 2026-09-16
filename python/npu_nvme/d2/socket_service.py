"""Draft socket service glue: single storage owner, rank-local byte producers."""
from __future__ import annotations
import hashlib
import json
from . import manifest_schema as manifest_schema
import threading
import time
from concurrent.futures import ThreadPoolExecutor
# Integration replaces these injected protocols with packaged implementations.


class RankService:
    def __init__(self,collective,region,wire,*,max_payload,deadline,tensor_schema):
        self.collective=collective;self.region=region;self.wire=wire
        self.max_payload=max_payload;self.deadline=deadline;self.lock=threading.RLock()
        self.tensor_schema=json.loads(json.dumps(tensor_schema))
        self.schema={(v['rank'],v['name']):v for v in self.tensor_schema}
        expected={(rank,name):size for rank,tensors in collective.rows.items() for name,size in tensors.items()}
        if len(self.schema)!=len(self.tensor_schema) or {k:v['bytes'] for k,v in self.schema.items()}!=expected:raise ValueError('service schema differs from collective')
        self.payload_rows={};self.errors=[]

    def serve_rank(self,sock,rank):
        try:
            while True:
                control,payload=self.wire.receive(sock,deadline=self.deadline,max_payload=self.max_payload)
                if control.get('rank')!=rank:raise ValueError('connection rank differs')
                if control.get('kind')=='chunk':
                    fields={k:control[k] for k in ('epoch','request_id','rank','name','offset','length','sha256','lease')}
                    if len(payload)!=fields['length'] or hashlib.sha256(payload).hexdigest()!=fields['sha256']:
                        raise ValueError('declared payload digest/length differs')
                    # Serialize admission with the single storage owner's
                    # source-safe completion. Other rank threads block here
                    # instead of interpreting temporary credit pressure as a
                    # failed collective. Socket receive buffers remain bounded.
                    with self.lock:
                        key=self.collective.admit(**fields)
                        if isinstance(key,tuple):
                            ref=self.region.payload(payload)
                            self.collective.complete(key,payload)
                            tensor=self.schema[(rank,control['name'])]
                            self.payload_rows[key]=dict(rank=rank,name=control['name'],shape=tensor['shape'],dtype=tensor['dtype'],partition=tensor['partition'],logical_offset=control['offset'],
                                logical_bytes=len(payload),payload=ref)
                    self.wire.send(sock,dict(kind='source_safe',lease=control['lease']),b'',deadline=self.deadline,max_payload=0)
                elif control.get('kind')=='complete':
                    if payload:raise ValueError('unexpected control payload')
                    fields={k:control[k] for k in ('epoch','request_id','rank','step','controls')}
                    self.collective.rank_complete(**fields)
                    return
                else:raise ValueError('unknown rank message')
        except BaseException as error:
            with self.lock:self.errors.append(error)
            self.collective.disconnect(rank)
            raise

    def run(self,connections,*,step,topology):
        if set(connections)!=set(range(self.collective.world_size)):raise ValueError('connection set')
        with ThreadPoolExecutor(max_workers=len(connections)) as executor:
            futures=[executor.submit(self.serve_rank,sock,rank) for rank,sock in connections.items()]
            for future in futures:future.result()
        manifest=self.collective.manifest()
        if manifest['step']!=step:raise ValueError('global step differs')
        if self.errors:raise RuntimeError('rank failure prevents commit')
        rows=[self.payload_rows[k] for k in sorted(self.payload_rows)]
        manifest_schema.validate(rows,self.tensor_schema,world_size=self.collective.world_size,chunk_bytes=self.collective.chunk)
        receipt=self.region.commit(rows,step=step,topology=topology,rank_controls=[dict(rank=rank,**value) for rank,value in sorted(manifest['ranks'].items())])
        for rank,sock in connections.items():
            self.wire.send(sock,dict(kind='committed',receipt=receipt),b'',deadline=self.deadline,max_payload=0)
        return receipt
