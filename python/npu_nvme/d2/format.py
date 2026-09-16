"""D2 immutable paged manifests, two anchors, and extent-union retention.

Draft for the separately registered D2 stage. Not installed or hardware accepted.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import struct
import threading
import uuid
from . import semantics as semantics
from . import control_pages as control_codec

BLOCK=4096
MAGIC=b'NPUNVM3\0'
HEADER=struct.Struct('<8sIIQ32s')
KINDS={'super':1,'anchor':2,'manifest':3,'catalog':4}
MAX_JSON=64*1024**2
SHARED_PAYLOAD='immutable_shared_payload_v1'


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(data):return hashlib.sha256(data).hexdigest()
def align(n):return (n+BLOCK-1)//BLOCK*BLOCK


def pack(kind,value,capacity):
    raw=canonical(value)
    if kind not in KINDS or capacity%BLOCK or len(raw)>capacity-HEADER.size:raise ValueError('page capacity exceeded')
    protected=HEADER.pack(MAGIC,3,KINDS[kind],len(raw),bytes(32))
    header=HEADER.pack(MAGIC,3,KINDS[kind],len(raw),hashlib.sha256(protected+raw).digest())
    return (header+raw).ljust(capacity,b'\0')


def unpack(kind,raw):
    if len(raw)<HEADER.size:raise ValueError('short page')
    magic,version,observed,size,digest=HEADER.unpack_from(raw)
    if magic!=MAGIC or version!=3 or observed!=KINDS[kind] or size>len(raw)-HEADER.size:raise ValueError('invalid page header')
    payload=raw[HEADER.size:HEADER.size+size]
    protected=HEADER.pack(magic,version,observed,size,bytes(32))
    if hashlib.sha256(protected+payload).digest()!=digest or any(raw[HEADER.size+size:]):raise ValueError('damaged page')
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate key')
            result[k]=v
        return result
    def constant(value):raise ValueError('nonfinite JSON constant')
    return json.loads(payload,object_pairs_hook=pairs,parse_constant=constant)


@dataclass(frozen=True,order=True)
class Extent:
    offset:int
    length:int

    def check(self,base,end):
        if type(self.offset) is not int or type(self.length) is not int or self.offset%BLOCK or self.length<=0 or self.length%BLOCK or self.offset<base or self.offset>end-self.length:raise ValueError('extent escapes region')
        return self


def union(extents):
    result=[]
    for e in sorted(extents):
        if result and e.offset<=result[-1].offset+result[-1].length:
            p=result.pop();result.append(Extent(p.offset,max(p.offset+p.length,e.offset+e.length)-p.offset))
        else:result.append(e)
    return result


def allocate(base,end,length,protected):
    length=align(length);cursor=base
    if length<=0:raise ValueError('invalid allocation')
    for e in union(protected):
        if e.offset-cursor>=length:return Extent(cursor,length)
        cursor=max(cursor,e.offset+e.length)
    if end-cursor>=length:return Extent(cursor,length)
    raise BufferError('retention/pins/pending generations consume the region')


class PageCodec:
    def __init__(self,page_bytes=65536,max_bytes=MAX_JSON):
        if type(page_bytes) is not int or type(max_bytes) is not int or page_bytes<BLOCK or page_bytes%BLOCK or max_bytes<page_bytes or max_bytes>4*1024**3:raise ValueError('invalid page budget')
        self.page_bytes=page_bytes;self.max_bytes=max_bytes

    def pages(self,rows):
        group=[];used=0;total=0
        for row in rows:
            size=len(canonical(row))+1
            if size>self.page_bytes-HEADER.size-64:raise ValueError('one manifest row exceeds page capacity')
            if group and used+size>self.page_bytes-HEADER.size-64:
                page=pack('manifest',{'rows':group},self.page_bytes);total+=len(page)
                if total>self.max_bytes:raise ValueError('manifest aggregate budget exceeded')
                yield page;group=[];used=0
            group.append(row);used+=size
        if group:
            page=pack('manifest',{'rows':group},self.page_bytes);total+=len(page)
            if total>self.max_bytes:raise ValueError('manifest aggregate budget exceeded')
            yield page

    def read(self,backend,refs,base,end):
        total=0
        for ref in refs:
            e=Extent(ref['offset'],ref['length']).check(base,end)
            total+=e.length
            if e.length!=self.page_bytes or total>self.max_bytes:raise ValueError('manifest decode budget exceeded')
            raw=backend.read(e.offset,e.length)
            if len(raw)!=e.length or sha(raw)!=ref['sha256']:raise ValueError('manifest page checksum differs')
            page=unpack('manifest',raw)
            if set(page)!={'rows'} or not isinstance(page['rows'],list):raise ValueError('invalid page rows')
            yield from page['rows']


class Region:
    """Single owner transaction engine; backend provides read/write/flush.

    The current and fallback catalog graphs, reader pins, and pending transaction
    all participate in allocation. Publication is payload flush -> immutable
    manifest/catalog flush -> alternate anchor flush. Failures poison this owner;
    an explicit fresh mount resolves uncertain anchor durability.
    """
    def __init__(self,backend,*,offset,length,retention=2,page_bytes=65536,metadata_budget_bytes=MAX_JSON):
        if type(offset) is not int or type(length) is not int or offset<0 or offset>2**64-length or offset%BLOCK or length%BLOCK or length<16*BLOCK or type(retention) is not int or retention not in (2,3):raise ValueError('region geometry')
        self.backend=backend;self.base=offset;self.end=offset+length;self.data_base=offset+3*BLOCK
        self.local_metadata_budget=metadata_budget_bytes
        self.retention=retention;self.codec=PageCodec(page_bytes,metadata_budget_bytes);self.lock=threading.RLock()
        self.roots=[None,None];self.current=None;self.pins={};self.pending=None;self.poisoned=False

    def format(self,*,region_id,features=()):
        # Caller must authorize this independently registered extent. Never
        # autoformat on mount failure, and never reuse a nonblank header here.
        if type(region_id) is not str or not 0<len(region_id)<=128 or any(self.backend.read(self.base,3*BLOCK)):raise ValueError('new region header is not blank')
        if set(features)-{SHARED_PAYLOAD}:raise ValueError('unknown region feature')
        header=dict(format='D2',region_id=region_id,offset=self.base,length=self.end-self.base,
                    page_bytes=self.codec.page_bytes,retention=self.retention,required_flags=sorted(set(features)),metadata_budget_bytes=self.codec.max_bytes)
        self.backend.write(self.base,pack('super',header,BLOCK));self.backend.flush()
        self.header=header

    def _extent_refs(self,root):
        if root is None:return []
        result=[Extent(root['catalog']['offset'],root['catalog']['length'])]
        for generation in root['state']['generations']:
            result += [Extent(p['offset'],p['length']) for p in generation['pages']+generation['extent_pages']+generation['control_pages']]
            result += [Extent(e['offset'],e['length']) for e in generation['extents']]
        return result

    def _graph_refs(self,root):
        yield root['catalog'],'catalog',root['sequence']
        for g in root['state']['generations']:
            for key in ('pages','extent_pages','control_pages','extents'):
                for ref in g[key]:yield ref,key,g['generation']

    def _check_root_union(self):
        if not all(self.roots):return
        refs=[]
        for slot,root in enumerate(self.roots):
            for ref,kind,generation in self._graph_refs(root):
                refs.append((ref['offset'],ref['length'],slot,kind,generation,canonical(ref)))
        previous=None
        for value in sorted(refs):
            if previous and value[0]<previous[0]+previous[1]:
                same=value[:2]==previous[:2] and value[3:]==previous[3:]
                shared=(SHARED_PAYLOAD in self.header['required_flags'] and value[3]==previous[3]=='extents' and
                        value[:2]==previous[:2] and value[5]==previous[5])
                if not (same or shared):raise ValueError('conflicting anchor graphs')
            previous=value

    def protected(self):
        result=[]
        for root in self.roots:result+=self._extent_refs(root)
        for root,count in self.pins.values():
            if count:result+=self._extent_refs(root)
        if self.pending:result+=self.pending['allocated']
        return result

    def _reserve(self,length):
        length=align(length)
        if length<=0:raise ValueError('invalid allocation')
        if self.pending.get('remaining_budget') is not None:
            if length>self.pending['remaining_budget']:raise BufferError('transaction exceeds admitted budget')
        for index,gap in enumerate(self.pending['free']):
            if gap.length>=length:
                e=Extent(gap.offset,length)
                if gap.length==length:self.pending['free'].pop(index)
                else:self.pending['free'][index]=Extent(gap.offset+length,gap.length-length)
                self.pending['allocated'].append(e)
                if self.pending.get('remaining_budget') is not None:self.pending['remaining_budget']-=length
                return e
        raise BufferError('retention/pins/pending generations consume the region')

    def _put(self,raw):
        e=self._reserve(len(raw))
        self.backend.write(e.offset,raw)
        return dict(offset=e.offset,length=e.length,sha256=sha(raw))

    def mount(self):
        raw=self.backend.read(self.base,BLOCK);header=unpack('super',raw)
        semantics.obj(header,('format','region_id','offset','length','page_bytes','retention','required_flags','metadata_budget_bytes'))
        if type(header['region_id']) is not str or not 0<len(header['region_id'])<=128:raise ValueError('region identity')
        if type(header['offset']) is not int or type(header['length']) is not int or type(header['retention']) is not int:raise ValueError('region geometry types')
        if header.get('format')!='D2' or header.get('offset')!=self.base or header.get('length')!=self.end-self.base or type(header.get('required_flags')) is not list or set(header.get('required_flags',[]))-{SHARED_PAYLOAD}:raise ValueError('unsupported region identity/flags')
        if header.get('retention') not in (2,3):raise ValueError('invalid retention')
        if self.pending or self.pins:raise RuntimeError('mount requires a fresh owner')
        self.roots=[None,None];self.current=None
        if type(header['metadata_budget_bytes']) is not int or header['metadata_budget_bytes']>self.local_metadata_budget:raise ValueError('region metadata exceeds local admission budget')
        self.header=header;self.retention=header['retention'];self.codec=PageCodec(header['page_bytes'],header['metadata_budget_bytes'])
        errors=[]
        for slot in (0,1):
            try:
                anchor_raw=self.backend.read(self.base+(slot+1)*BLOCK,BLOCK)
                if not any(anchor_raw):continue
                anchor=unpack('anchor',anchor_raw)
                if anchor['region_id']!=header['region_id']:raise ValueError('foreign anchor')
                ref=anchor['catalog'];e=Extent(ref['offset'],ref['length']).check(self.data_base,self.end)
                if e.length>self.codec.max_bytes:raise ValueError('catalog decode budget')
                catalog_raw=self.backend.read(e.offset,e.length)
                if sha(catalog_raw)!=ref['sha256']:raise ValueError('catalog checksum differs')
                state=unpack('catalog',catalog_raw)
                for generation in state['generations']:
                    if 'extents' in generation:raise ValueError('inline extent index unsupported')
                    generation['extents']=list(self.codec.read(self.backend,generation['extent_pages'],self.data_base,self.end))
                semantics.catalog(anchor,state,self.retention,allow_shared=SHARED_PAYLOAD in header['required_flags'])
                if len(state.get('receipts',[]))>128:raise ValueError('receipt window exceeded')
                if type(anchor['sequence']) is not int or anchor['sequence']<=0:raise ValueError('invalid anchor sequence')
                if state['sequence']!=anchor['sequence'] or len(state['generations'])>self.retention:raise ValueError('catalog sequence/retention')
                root=dict(slot=slot,sequence=anchor['sequence'],catalog=ref,state=state)
                for extent in self._extent_refs(root):extent.check(self.data_base,self.end)
                # Verify all manifest pages before accepting an anchor, including
                # retained generations needed for promised fallback.
                for generation in state['generations']:
                    self.read_controls(generation)
                    rows=list(self.codec.read(self.backend,generation['pages'],self.data_base,self.end))
                    if sha(canonical(rows))!=generation['manifest_sha256']:raise ValueError('manifest digest differs')
                    owned={canonical(ref) for ref in generation['extents']}
                    for row in rows:
                        ref=row.get('payload')
                        if not isinstance(ref,dict) or canonical(ref) not in owned:
                            raise ValueError('manifest references unowned payload')
                self.roots[slot]=root
            except (ValueError,KeyError,TypeError,RecursionError) as error:errors.append(dict(slot=slot,error=str(error)))
        self._check_root_union()
        valid=[r for r in self.roots if r]
        if not valid:
            if errors:raise ValueError('no valid committed anchor')
            self.current=None
        else:
            if len(valid)==2 and valid[0]['sequence']==valid[1]['sequence']:raise ValueError('ambiguous anchor sequence')
            self.current=max(valid,key=lambda r:r['sequence'])
        self.poisoned=False;return errors

    def read_controls(self,generation):
        return control_codec.decode(self.codec.read(self.backend,generation['control_pages'],self.data_base,self.end),budget=self.codec.max_bytes)

    def verify_payloads(self,generation):
        for ref in generation['extents']:
            extent=Extent(ref['offset'],ref['length']).check(self.data_base,self.end)
            logical=ref['logical_bytes']
            if type(logical) is not int or not 0<logical<=extent.length:raise ValueError('logical payload bounds')
            # Bound reads by transport page size, not by full tensor extent.
            digest=hashlib.sha256();remaining=logical;cursor=0
            while cursor<extent.length:
                count=min(self.codec.page_bytes,extent.length-cursor)
                raw=self.backend.read(extent.offset+cursor,count)
                if len(raw)!=count:raise ValueError('short payload read')
                take=min(remaining,count);digest.update(raw[:take]);remaining-=take;cursor+=count
                if any(raw[take:]):raise ValueError('nonzero payload padding')
            if digest.hexdigest()!=ref['logical_sha256']:raise ValueError('payload corruption')
        return True

    def begin(self,request_id,identity,*,retry=False,reserve_bytes=None):
        with self.lock:
            if self.poisoned or self.pending:raise RuntimeError('owner unavailable')
            if not isinstance(request_id,str) or not 0<len(request_id)<=128:raise ValueError('request id')
            state=self.current['state'] if self.current else {'receipts':[]}
            for receipt in state['receipts']:
                if receipt['request_id']==request_id:
                    if receipt['identity']!=identity:raise ValueError('idempotency identity mismatch')
                    return dict(receipt,replayed=True)
            if retry:raise KeyError('retry receipt expired or unknown; no automatic new commit')
            # Capture free intervals once per transaction. Pins acquired later
            # only reference roots already protected here; releasing a pin can
            # conservatively leave space unavailable until the next begin.
            free=[];cursor=self.data_base
            for e in union(self.protected()):
                if e.offset>cursor:free.append(Extent(cursor,e.offset-cursor))
                cursor=max(cursor,e.offset+e.length)
            if cursor<self.end:free.append(Extent(cursor,self.end-cursor))
            if reserve_bytes is not None:
                if type(reserve_bytes) is not int or reserve_bytes<=0 or reserve_bytes%BLOCK:raise ValueError('transaction reservation geometry')
                # A contiguous reservation avoids admitting capacity that is
                # available only as fragments smaller than requested pages.
                reserved=next((e for e in free if e.length>=reserve_bytes),None)
                if reserved is None:raise BufferError('insufficient contiguous transaction capacity')
                free=[Extent(reserved.offset,reserve_bytes)]
            self.pending=dict(request_id=request_id,identity=json.loads(canonical(identity)),allocated=[],payload=[],free=free,remaining_budget=reserve_bytes)
            return None

    def payload(self,data):
        with self.lock:
            if not self.pending or self.poisoned:raise RuntimeError('no writable transaction')
            try:
                if not data:raise ValueError('empty payload')
                ref=self._put(bytes(data).ljust(align(len(data)),b'\0'))
                ref.update(logical_bytes=len(data),logical_sha256=sha(data))
                self.pending['payload'].append(json.loads(canonical(ref)));return ref
            except BaseException:self.poisoned=True;raise

    def commit(self,rows,*,step,topology,rank_controls=None,inherited_payloads=()):
        with self.lock:
            if not self.pending or self.poisoned:raise RuntimeError('no writable transaction')
            try:
                self.backend.flush()
                rows=json.loads(canonical(list(rows)))
                inherited=json.loads(canonical(list(inherited_payloads)))
                if inherited and SHARED_PAYLOAD not in self.header['required_flags']:
                    raise ValueError('shared payload requires explicitly formatted feature')
                existing={canonical(ref) for g in (self.current['state']['generations'] if self.current else []) for ref in g['extents']}
                if any(canonical(ref) not in existing for ref in inherited):raise ValueError('inherited payload is not committed')
                all_payloads={canonical(ref):ref for ref in self.pending['payload']+inherited}
                payloads=list(all_payloads.values())
                owned={(ref['offset'],ref['length'],ref['sha256']) for ref in payloads}
                for row in rows:
                    ref=row.get('payload')
                    if not isinstance(ref,dict) or canonical(ref) not in all_payloads:
                        raise ValueError('manifest payload is not owned by this transaction')
                page_refs=[self._put(raw) for raw in self.codec.pages(rows)]
                sequence=self.current['sequence']+1 if self.current else 1
                extent_pages=[self._put(raw) for raw in self.codec.pages(payloads)]
                control_pages=[self._put(raw) for raw in self.codec.pages(control_codec.encode(rank_controls or [],budget=self.codec.max_bytes))]
                generation=dict(generation=sequence,step=step,topology=topology,pages=page_refs,extent_pages=extent_pages,control_pages=control_pages,
                    extents=payloads,manifest_sha256=sha(canonical(rows)))
                old=self.current['state'] if self.current else dict(generations=[],receipts=[])
                receipt=dict(request_id=self.pending['request_id'],identity=self.pending['identity'],generation=sequence)
                state=dict(sequence=sequence,generations=([generation]+old['generations'])[:self.retention],
                           receipts=(old['receipts']+[receipt])[-128:])
                disk_state=dict(state,generations=[{k:v for k,v in g.items() if k!='extents'} for g in state['generations']])
                catalog_bytes=align(HEADER.size+len(canonical(disk_state)))
                if catalog_bytes>self.codec.max_bytes:raise ValueError('catalog aggregate budget exceeded')
                catalog=self._put(pack('catalog',disk_state,catalog_bytes))
                self.backend.flush()
                slot=1-self.current['slot'] if self.current else 0
                anchor=dict(sequence=sequence,region_id=self.header['region_id'],catalog=catalog)
                self.backend.write(self.base+(slot+1)*BLOCK,pack('anchor',anchor,BLOCK))
                self.backend.flush()
                root=dict(slot=slot,sequence=sequence,catalog=catalog,state=state)
                self.roots[slot]=root;self.current=root;self.pending=None
                return receipt
            except BaseException:self.poisoned=True;raise

    @contextmanager
    def selected(self,generation=None):
        with self.lock:
            if not self.current:raise ValueError('no committed generation')
            root=self.current
            available=root['state']['generations']
            chosen=available[0] if generation is None else next((g for g in available if g['generation']==generation),None)
            if chosen is None:raise ValueError('generation outside retention window')
            key=root['sequence'];previous=self.pins.get(key,(root,0));self.pins[key]=(root,previous[1]+1)
        try:yield chosen
        finally:
            with self.lock:
                root,count=self.pins[key]
                if count==1:del self.pins[key]
                else:self.pins[key]=(root,count-1)
