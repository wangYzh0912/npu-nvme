"""Independent shadow replay from bytes read back from committed raw frames."""
import json
from collections import OrderedDict
import math
from pathlib import Path
import shutil
import numpy as np
from npu_nvme.runtime.training_catalog import read_checked
from npu_nvme.experiments.tp_blocks import logical_fragments


class ShadowArrays:
    """Bound mapped files so the socket transport remains below FD_SETSIZE."""
    def __init__(self,paths,limit=16):self.paths=paths;self.limit=limit;self.cache=OrderedDict()
    def __contains__(self,name):return name in self.paths
    def __getitem__(self,name):
        if name in self.cache:
            value=self.cache.pop(name);self.cache[name]=value;return value
        if len(self.cache)>=self.limit:
            _,old=self.cache.popitem(last=False);old.flush();old._mmap.close()
        value=np.load(self.paths[name],mmap_mode='r+',allow_pickle=False)
        self.cache[name]=value;return value
    def values(self):
        for name in self.paths:yield self[name]


class MediaShadow:
    def __init__(self, initial, schema_path, output, block_elements=65536):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        schema=json.loads(Path(schema_path).read_text())
        self.schemas={r['name']:r for r in schema['tensors'] if r['role']=='model'}
        self.arrays={};self.expected={};self.indices={}
        for rank in range(4):
            source=Path(initial)/f'rank_{rank}';manifest=read_checked(source/'manifest.json')
            directory=self.output/f'rank_{rank}';directory.mkdir();index={};arrays={};expected=set()
            for i,name in enumerate(sorted(self.schemas)):
                entry=manifest['tensors'][name];filename=f'weight-{i:04d}.npy'
                shutil.copyfile(source/entry['file'],directory/filename)
                value=np.load(directory/filename,mmap_mode='r+',allow_pickle=False)
                if value.dtype!=np.float32 or value.nbytes!=entry['bytes']:raise ValueError('shadow initial geometry differs')
                arrays[name]=directory/filename;index[name]=filename
                value._mmap.close()
                for row in logical_fragments(self.schemas[name],block_elements):
                    if row['rank']==rank:expected.add((name,row['local_element_offset'],row['element_count'],row['block_index'],row['global_element_offset']))
            (directory/'index.json').write_text(json.dumps(index)+'\n')
            self.arrays[rank]=ShadowArrays(arrays);self.expected[rank]=expected;self.indices[rank]=index
        self.block_elements=block_elements;self.step=0

    def begin(self, step):
        if step!=self.step+1:raise ValueError('shadow replay is not sequential')
        self.current_step=step;self.records={};self.cursors={};self.blocks={};self.seen=set();self.expected_bytes={}

    def _descriptor(self, rank, descriptor):
        if descriptor['rank']!=rank or descriptor['logical_step']!=self.current_step:raise ValueError('shadow descriptor lineage differs')
        records=descriptor['records'];cursor=0
        for row in records:
            name=row['name'];start=row['element_offset'];count=row['element_count']
            if name not in self.arrays[rank] or start<0 or count<=0 or start+count>self.arrays[rank][name].size:
                raise ValueError('shadow destination out of bounds')
            if row['payload_offset']!=cursor or row['payload_bytes']!=count*4:raise ValueError('shadow payload layout differs')
            inner=start
            for fragment in row['fragments']:
                key=(name,fragment['local_element_offset'],fragment['element_count'],fragment['block_index'],fragment['global_element_offset'])
                if (fragment['name']!=name or fragment['rank']!=rank or key not in self.expected[rank] or
                        (rank,key) in self.seen or fragment['local_element_offset']!=inner):
                    raise ValueError('shadow fragment mapping differs')
                self.seen.add((rank,key));inner+=fragment['element_count']
                block=(name,fragment['block_index']);self.blocks[block]=self.blocks.get(block,0)+fragment['element_count']
            if inner!=start+count:raise ValueError('shadow fragments do not cover packed range')
            cursor+=row['payload_bytes']
        self.records[rank]=records;self.cursors[rank]=[0,0];self.expected_bytes[rank]=cursor

    def consume(self, rank, descriptor, offset, data):
        if rank not in self.records:self._descriptor(rank,descriptor)
        cursor,index=self.cursors[rank]
        if offset!=cursor:raise ValueError('shadow byte stream is not contiguous')
        raw=np.frombuffer(data,dtype=np.uint8);used=0;records=self.records[rank]
        while used<len(raw):
            row=records[index];within=cursor-row['payload_offset'];take=min(len(raw)-used,row['payload_bytes']-within)
            if take<=0:raise ValueError('invalid shadow decode extent')
            target=self.arrays[rank][row['name']].reshape(-1).view(np.uint8)
            begin=row['element_offset']*4+within;target[begin:begin+take]=raw[used:used+take]
            used+=take;cursor+=take
            if within+take==row['payload_bytes']:index+=1
        self.cursors[rank]=[cursor,index]

    def finish(self):
        if set(self.records)!=set(range(4)):raise ValueError('shadow replay lacks a rank')
        for rank,(cursor,index) in self.cursors.items():
            if cursor!=self.expected_bytes[rank] or index!=len(self.records[rank]):raise ValueError('shadow payload incomplete')
        for (name,index),count in self.blocks.items():
            total=math.prod(self.schemas[name]['global_shape'])
            if count!=min(self.block_elements,total-index*self.block_elements):raise ValueError('logical block was only partly saved')
        for name,row in self.schemas.items():
            total=math.prod(row['global_shape'])
            if total<self.block_elements and self.blocks.get((name,0))!=total:raise ValueError('small parameter missing')
            if row['partition']=='replicated':
                for rank in range(1,4):self.arrays[rank][name][...]=self.arrays[0][name]
        for arrays in self.arrays.values():
            for array in arrays.values():array.flush()
        self.step=self.current_step
        document=dict(step=self.step,source='actual raw media readback',blocks=len(self.blocks),payload_bytes=sum(self.expected_bytes.values()))
        target=self.output/'ready.json';temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(document)+'\n');temporary.replace(target)
        return document
