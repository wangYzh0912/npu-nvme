"""Single-rank lossless R0 frames with durable D2 receipts and shared FULL data.

Every retained generation names its complete FULL + delta closure. Payloads are
immutable and shared; manifests are republished, but FULL bytes are not recopied
until explicit compaction. Only a validated commit receipt advances reference.
"""
import hashlib
import copy
import json
import uuid
import numpy as np
from incremental_frame import unpack_r0_frame
from r0_session import R0Session
from .format import SHARED_PAYLOAD,canonical


def sha(data):return hashlib.sha256(data).hexdigest()


class PersistentR0:
    def __init__(self,region,manifest,*,chunk_bytes=1<<20,max_chain_length=8):
        if SHARED_PAYLOAD not in region.header['required_flags']:raise ValueError('R0 requires explicit shared-payload region')
        if type(chunk_bytes) is not int or chunk_bytes<=0 or chunk_bytes%4096 or type(max_chain_length) is not int or max_chain_length<=0:
            raise ValueError('R0 storage budgets')
        self.region=region;self.manifest=copy.deepcopy(manifest);self.chunk=chunk_bytes;self.max_chain=max_chain_length
        self.ledger=None;self.rows=[];self.frames=[];self.failed=False
    def _put(self,name,data,*,dtype,shape):
        rows=[]
        for offset in range(0,len(data),self.chunk):
            piece=data[offset:offset+self.chunk];ref=self.region.payload(piece)
            rows.append(dict(rank=0,name=name,shape=shape,dtype=dtype,partition='replicated',logical_offset=offset,logical_bytes=len(piece),payload=ref))
        return rows
    def _commit(self,rows,identity,controls,step,*,inherited=()):
        expected=self.region.current['sequence']+1 if self.region.current else 1
        receipt=self.region.commit(rows,step=step,topology=dict(world_size=1),
            rank_controls=[dict(rank=0,step=step,controls=controls)],inherited_payloads=inherited)
        if receipt!=dict(request_id=identity['request_id'],identity=identity,generation=expected):
            raise ValueError('durable R0 receipt identity differs')
        return receipt
    def save_full(self,state,controls,*,step,request_id=None):
        if self.failed:raise RuntimeError('R0 owner requires fresh recovery')
        generation=self.region.current['sequence']+1 if self.region.current else 1
        ledger=R0Session(state,self.manifest,base_full_generation=generation)
        request_id=request_id or uuid.uuid4().hex
        identity=dict(request_id=request_id,kind='FULL',manifest_sha256=self.manifest.digest,root=generation,parent=None,frame_sha256=None)
        meta=dict(kind='R0',root=generation,frames=[],manifest_sha256=self.manifest.digest,state_controls=json.loads(canonical(controls)))
        try:
            self.region.begin(request_id,identity)
            rows=[]
            for field in self.manifest.fields:
                value=ledger.base_state[field.canonical_name]
                rows+=self._put('root/'+field.canonical_name,value.tobytes(),dtype=value.dtype.name,shape=list(value.shape))
            receipt=self._commit(rows,identity,meta,step)
            self.ledger=ledger;self.rows=rows;self.frames=[]
            return receipt
        except BaseException:self.failed=True;raise
    def save(self,state,controls,*,step,request_id=None):
        if self.failed:raise RuntimeError('R0 owner requires fresh recovery')
        if self.ledger is None or len(self.frames)>=self.max_chain:
            return self.save_full(state,controls,step=step,request_id=request_id)
        self.ledger.set_current(state)
        index=self.ledger.persisted_generation+1
        frame=self.ledger.observe(step,index,controls=[('state_controls','json',canonical(controls))])
        request_id=request_id or uuid.uuid4().hex
        identity=dict(request_id=request_id,kind='DELTA',manifest_sha256=self.manifest.digest,
            root=self.ledger.base_full_generation,parent=self.ledger.persisted_generation,frame_generation=index,frame_sha256=sha(frame))
        frames=[*self.frames,dict(generation=index,name='delta/'+str(index),sha256=sha(frame),bytes=len(frame))]
        meta=dict(kind='R0',root=self.ledger.base_full_generation,frames=frames,manifest_sha256=self.manifest.digest,state_controls=json.loads(canonical(controls)))
        try:
            self.region.begin(request_id,identity)
            rows=[*self.rows,*self._put('delta/'+str(index),frame,dtype='uint8',shape=[len(frame)])]
            inherited=[r['payload'] for r in self.rows]
            receipt=self._commit(rows,identity,meta,step,inherited=inherited)
            self.ledger.ack(frame)
            self.rows=rows;self.frames=frames
            return receipt
        except BaseException:self.failed=True;raise
    def recover(self,generation=None):
        with self.region.selected(generation) as selected:
            controls=self.region.read_controls(selected)
            if len(controls)!=1 or controls[0]['rank']!=0 or controls[0]['step']!=selected['step']:raise ValueError('R0 rank controls')
            meta=controls[0]['controls']
            if meta.get('kind')!='R0' or meta.get('manifest_sha256')!=self.manifest.digest or len(meta['frames'])>self.max_chain:
                raise ValueError('R0 manifest/chain differs')
            rows=list(self.region.codec.read(self.region.backend,selected['pages'],self.region.data_base,self.region.end))
            expected={'root/'+f.canonical_name:(list(f.shape),np.dtype(f.dtype).name,f.byte_count) for f in self.manifest.fields}
            for frame in meta['frames']:
                if frame['name'] in expected:raise ValueError('duplicate frame')
                expected[frame['name']]=([frame['bytes']],'uint8',frame['bytes'])
            payloads={name:bytearray() for name in expected}
            for row in rows:
                name=row['name'];ref=row['payload']
                if name not in expected or row['rank']!=0:raise ValueError('unexpected R0 row')
                shape,dtype,size=expected[name];current=payloads[name]
                if row['shape']!=shape or row['dtype']!=dtype or row['logical_offset']!=len(current) or row['logical_bytes']!=min(self.chunk,size-len(current)):
                    raise ValueError('R0 row shape/offset')
                raw=self.region.backend.read(ref['offset'],ref['length']);data=raw[:ref['logical_bytes']]
                if sha(raw)!=ref['sha256'] or sha(data)!=ref['logical_sha256'] or len(data)!=row['logical_bytes'] or any(raw[len(data):]):
                    raise ValueError('R0 payload integrity')
                current.extend(data)
            if any(len(payloads[name])!=spec[2] for name,spec in expected.items()):raise ValueError('R0 state incomplete')
            base={f.canonical_name:np.frombuffer(payloads['root/'+f.canonical_name],dtype=f.dtype).reshape(f.shape).copy() for f in self.manifest.fields}
            ledger=R0Session(base,self.manifest,base_full_generation=meta['root']);frames=[]
            for index,descriptor in enumerate(meta['frames'],1):
                frame=bytes(payloads[descriptor['name']]);info=unpack_r0_frame(frame)
                if descriptor['generation']!=index or info['generation']!=index or sha(frame)!=descriptor['sha256']:raise ValueError('R0 frame identity')
                frames.append(frame)
            recovered=ledger.recover(frames)
            if frames:
                last=unpack_r0_frame(frames[-1])
                if last['step']!=selected['step'] or len(last['controls'])!=1 or last['controls'][0]['name']!='state_controls' or json.loads(last['controls'][0]['payload'])!=meta['state_controls']:
                    raise ValueError('R0 control/frame mismatch')
            self.ledger=ledger;self.ledger.persisted={k:v.copy() for k,v in recovered['state'].items()}
            self.ledger.current={k:v.copy() for k,v in recovered['state'].items()}
            self.ledger.persisted_generation=recovered['generation'];self.rows=rows;self.frames=meta['frames'];self.failed=False
            return dict(state=recovered['state'],controls=meta['state_controls'],generation=selected['generation'],step=selected['step'])
