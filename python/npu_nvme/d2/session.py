"""Bounded socket sessions for one storage owner and independent rank clients.

The caller freezes training until save returns. Restore callers must perform a
framework collective barrier after local verification and release, before any
optimizer step; socket release alone is not an atomic distributed operation.
"""
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from . import wire,manifest_schema
from .rank import Collective
from .socket_service import RankService
from .ready_protocol import ReadyDecision,coordinate
from .region_budget import region_budget


def sha(data):return hashlib.sha256(data).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def validate_schemas(schemas,world):
    if set(schemas)!=set(range(world)):raise ValueError('rank schema set')
    expected=[];names=None
    for rank,rows in sorted(schemas.items()):
        observed=set()
        for row in rows:
            if set(row)!= {'rank','name','shape','dtype','partition','bytes'} or row['rank']!=rank:
                raise ValueError('rank schema fields')
            if row['name'] in observed:raise ValueError('duplicate tensor')
            observed.add(row['name'])
            if row['dtype'] not in manifest_schema.WIDTHS or row['partition'] not in ('sharded','replicated','per_rank_control'):
                raise ValueError('schema dtype/partition')
            if not isinstance(row['name'],str) or not 0<len(row['name'])<=1024 or type(row['shape']) is not list or any(type(v) is not int or v<0 for v in row['shape']):
                raise ValueError('schema name/shape')
            if type(row['bytes']) is not int or row['bytes']<=0 or math.prod(row['shape'])*manifest_schema.WIDTHS[row['dtype']]!=row['bytes']:
                raise ValueError('schema bytes')
            expected.append(row)
        if names is None:names=observed
        if not observed or names!=observed:raise ValueError('rank tensor names differ')
    return expected


class OwnerSession:
    def __init__(self,region,connections,*,epoch,request_id,topology,chunk_bytes,deadline,identity):
        world=topology.get('world_size')
        if world not in (2,4) or set(connections)!=set(range(world)):raise ValueError('topology')
        self.region=region;self.connections=connections;self.world=world;self.epoch=epoch
        self.request=request_id;self.topology=topology;self.chunk=chunk_bytes;self.deadline=deadline
        self.identity=json.loads(canonical(identity))
    def send(self,rank,control,data=b''):
        wire.send(self.connections[rank],control,data,deadline=self.deadline,max_payload=self.chunk)
    def schemas(self,operation):
        def receive(rank):
            control,payload=wire.receive(self.connections[rank],deadline=self.deadline,max_payload=0)
            if payload or control.get('kind')!='hello' or control.get('rank')!=rank or control.get('epoch')!=self.epoch or control.get('operation')!=operation or control.get('identity')!=self.identity:
                raise ValueError('session identity mismatch')
            return rank,control['schema']
        with ThreadPoolExecutor(max_workers=self.world) as pool:
            schemas=dict(pool.map(receive,range(self.world)))
        return validate_schemas(schemas,self.world)
    def save(self,*,step):
        expected=self.schemas('save')
        by_key={(v['rank'],v['name']):v for v in expected}
        names=sorted({v['name'] for v in expected})
        schema=[dict(name=name,partition=by_key[(0,name)]['partition'],
                     bytes_per_rank=[by_key[(rank,name)]['bytes'] for rank in range(self.world)]) for name in names]
        collective=Collective(world_size=self.world,epoch=self.epoch,schema=schema,chunk_bytes=self.chunk,
                              credits=self.world,timeout_seconds=max(1,self.deadline-__import__('time').monotonic()))
        collective.begin(self.request,step)
        budget=region_budget(state_bytes=sum(v['bytes'] for v in expected),
            descriptor_count=sum((v['bytes']+self.chunk-1)//self.chunk for v in expected),
            page_bytes=self.region.codec.page_bytes,retention=self.region.retention,active_pins=0,
            control_bytes=self.world*wire.MAX_CONTROL)
        reserve=(budget['per_generation_upper_bytes']+4*self.region.codec.page_bytes+4095)//4096*4096
        self.region.begin(self.request,self.identity,reserve_bytes=reserve)
        for rank in self.connections:self.send(rank,dict(kind='admitted',step=step,request_id=self.request))
        return RankService(collective,self.region,wire,max_payload=self.chunk,deadline=self.deadline,
                           tensor_schema=expected).run(self.connections,step=step,topology=self.topology)
    def restore(self,*,generation=None):
        expected=self.schemas('restore')
        with self.region.selected(generation) as selected:
            if selected['topology']!=self.topology:raise ValueError('restore topology differs')
            receipts=self.region.current['state']['receipts']
            stored=next((v for v in receipts if v['generation']==selected['generation']),None)
            if stored is None or stored['identity']!=self.identity:raise ValueError('stored training identity differs')
            rows=list(self.region.codec.read(self.region.backend,selected['pages'],self.region.data_base,self.region.end))
            manifest_schema.validate(rows,expected,world_size=self.world,chunk_bytes=self.chunk)
            controls=self.region.read_controls(selected)
            if len(controls)!=self.world or {v.get('rank') for v in controls}!=set(range(self.world)):
                raise ValueError('restore rank controls incomplete')
            control_by_rank={v['rank']:v for v in controls}
            if any(v['step']!=selected['step'] for v in controls):raise ValueError('controls step differs')
            decision=ReadyDecision(world_size=self.world,epoch=self.epoch,generation=selected['generation'],
                                   manifest_sha256=selected['manifest_sha256'])
            for rank in self.connections:self.send(rank,dict(kind='restore_begin',step=selected['step'],**decision.identity))
            for row in rows:
                ref=row['payload'];raw=self.region.backend.read(ref['offset'],ref['length'])
                if len(raw)!=ref['length'] or sha(raw)!=ref['sha256']:raise ValueError('stored payload digest differs')
                data=raw[:ref['logical_bytes']]
                if sha(data)!=ref['logical_sha256'] or any(raw[len(data):]):raise ValueError('stored logical bytes differ')
                rank=row['rank'];self.send(rank,dict(kind='restore_chunk',name=row['name'],offset=row['logical_offset'],
                    length=len(data),sha256=ref['logical_sha256']),data)
                ack,payload=wire.receive(self.connections[rank],deadline=self.deadline,max_payload=0)
                if payload or ack!=dict(kind='applied',name=row['name'],offset=row['logical_offset']):raise ValueError('restore ACK differs')
            for rank in self.connections:self.send(rank,dict(kind='controls',controls=control_by_rank[rank]['controls']))
            return coordinate(self.connections,decision,wire,deadline=self.deadline)


class RankSession:
    def __init__(self,sock,*,rank,epoch,identity,schema,chunk_bytes,deadline):
        self.sock=sock;self.rank=rank;self.epoch=epoch;self.identity=identity;self.schema=schema
        self.chunk=chunk_bytes;self.deadline=deadline
    def send(self,control,data=b''):wire.send(self.sock,control,data,deadline=self.deadline,max_payload=self.chunk)
    def receive(self):return wire.receive(self.sock,deadline=self.deadline,max_payload=self.chunk)
    def hello(self,operation):self.send(dict(kind='hello',rank=self.rank,epoch=self.epoch,identity=self.identity,schema=self.schema,operation=operation))
    def save(self,read_chunk,controls,*,step):
        self.hello('save');admission,payload=self.receive()
        if payload or admission.get('kind')!='admitted' or admission.get('step')!=step:raise ValueError('save admission differs')
        request=admission['request_id'];lease=0
        for tensor in sorted(self.schema,key=lambda v:v['name']):
            for offset in range(0,tensor['bytes'],self.chunk):
                length=min(self.chunk,tensor['bytes']-offset)
                with read_chunk(tensor['name'],offset,length) as data:
                    if len(data)!=length:raise ValueError('capture short read')
                    self.send(dict(kind='chunk',rank=self.rank,epoch=self.epoch,request_id=request,
                        name=tensor['name'],offset=offset,length=length,sha256=sha(data),lease=lease),data)
                    ack,payload=self.receive()
                    if payload or ack!=dict(kind='source_safe',lease=lease):raise ValueError('save ACK differs')
                lease+=1
        self.send(dict(kind='complete',rank=self.rank,epoch=self.epoch,request_id=request,step=step,controls=controls))
        result,payload=self.receive()
        if payload or result.get('kind')!='committed':raise ValueError('global commit missing')
        return result['receipt']
    def restore(self,apply_chunk,verify_controls):
        self.hello('restore');begin,payload=self.receive()
        if payload or begin.get('kind')!='restore_begin' or begin.get('epoch')!=self.epoch:raise ValueError('restore admission differs')
        expected={v['name']:v['bytes'] for v in self.schema};cursors={name:0 for name in expected}
        while True:
            control,data=self.receive()
            if control.get('kind')=='controls':
                if data or cursors!=expected:raise ValueError('incomplete restored schema')
                verify_controls(control['controls'],begin['step'])
                break
            if control.get('kind')!='restore_chunk':raise ValueError('unexpected restore message')
            name=control['name'];offset=control['offset']
            if name not in expected or offset!=cursors[name] or len(data)!=min(self.chunk,expected[name]-offset) or not data or control['length']!=len(data) or sha(data)!=control['sha256']:
                raise ValueError('restore chunk invalid')
            apply_chunk(name,offset,data,control['sha256'])
            cursors[name]+=len(data);self.send(dict(kind='applied',name=name,offset=offset))
        self.send(dict(kind='prepared',rank=self.rank,epoch=self.epoch,generation=begin['generation'],
            manifest_sha256=begin['manifest_sha256'],transport_safe=True,schema_verified=True,controls_verified=True))
        release,data=self.receive()
        if data or release.get('kind')!='release' or any(release.get(k)!=begin[k] for k in ('epoch','generation','manifest_sha256')):
            raise ValueError('restore release differs')
        value={k:v for k,v in release.items() if k not in ('kind','token')}
        if sha(canonical(value))!=release.get('token'):raise ValueError('invalid ready token')
        return {k:v for k,v in release.items() if k!='kind'}
