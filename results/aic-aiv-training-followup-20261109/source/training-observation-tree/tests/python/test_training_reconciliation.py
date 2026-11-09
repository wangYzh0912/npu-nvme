"""Public-index recovery against real D2 metadata with an in-memory disk."""
import hashlib

import pytest

from npu_nvme.d2.format import Region
from npu_nvme.runtime import training_catalog as catalog


class Disk:
    def __init__(self): self.data=bytearray(16<<20)
    def read(self,offset,length): return bytes(self.data[offset:offset+length])
    def write(self,offset,data): self.data[offset:offset+len(data)]=data
    def flush(self): pass


def transaction(root, *, commit=True):
    identity={'model':'publication-fixture'}
    catalog.initialize(root,identity=identity,method='ours')
    path,reservation=catalog.reserve(root,run_id='source',step=4)
    disk=Disk();region=Region(disk,offset=0,length=len(disk.data),retention=3)
    region.format(region_id='fixture')
    request='epoch:0';rows=[];controls=[]
    region.begin(request,identity)
    for rank in range(4):
        data=bytes([rank])*4
        control={'cursor':4,'rank':rank}
        state={'x':dict(shape=[4],dtype='|u1',bytes=4,sha256=hashlib.sha256(data).hexdigest())}
        prepared=dict(preparation_version=1,raw_request_id=request,rank=rank,
            generation=reservation['generation'],step=4,method='ours',identity=identity,
            state=state,controls=control,backend=dict(operation='save',step=4,sequence=0,seconds=None))
        catalog.write_checked(path/f'rank_{rank}/prepared.json',prepared)
        ref=region.payload(data)
        rows.append(dict(rank=rank,name='x',shape=[4],dtype='uint8',partition='sharded',
                         logical_offset=0,logical_bytes=4,payload=ref))
        controls.append(dict(rank=rank,step=4,controls=control))
    if commit:region.commit(rows,step=4,topology={'world_size':4},rank_controls=controls)
    fresh=Region(disk,offset=0,length=len(disk.data),retention=3)
    fresh.mount()
    return path,fresh.publication_inventory()


def test_raw_commit_without_ack_or_rank_completion_is_published_once(tmp_path):
    path,inventory=transaction(tmp_path)
    result=catalog.reconcile(tmp_path,inventory)
    assert result==dict(recovered=[1],rejected=[])
    with catalog.selected(tmp_path) as (selected,value):
        assert selected==path and value['step']==4 and value['recovered']
        for rank,row in enumerate(value['ranks']):
            assert row['recovered'] and row['save_seconds'] is None
            assert row['backend']['seconds'] is None
            assert row['backend']['receipt']==inventory[0]['receipt']
            catalog.verify_files(path/f'rank_{rank}',row['files'])
    before=(path/'checkpoint.json').read_bytes()
    assert catalog.reconcile(tmp_path,inventory)==dict(recovered=[],rejected=[])
    assert (path/'checkpoint.json').read_bytes()==before


@pytest.mark.parametrize('damage',['missing_rank','raw_absent','wrong_request','wrong_step',
                                  'wrong_identity','wrong_topology','wrong_controls','corrupt_prepare','wrong_directory'])
def test_incomplete_or_conflicting_material_never_publishes(tmp_path,damage):
    path,inventory=transaction(tmp_path,commit=damage!='raw_absent')
    journal=path/'rank_3/prepared.json'
    if damage=='missing_rank':journal.unlink()
    elif damage=='wrong_request':
        row=catalog.read_checked(journal);row['raw_request_id']='other:0';catalog.write_checked(journal,row)
    elif damage=='wrong_step':inventory[0]['step']=8
    elif damage=='wrong_identity':inventory[0]['receipt']['identity']={'model':'other'}
    elif damage=='wrong_topology':inventory[0]['topology']={'world_size':2}
    elif damage=='wrong_controls':inventory[0]['controls_sha256']='0'*64
    elif damage=='corrupt_prepare':journal.write_text('damaged')
    elif damage=='wrong_directory':
        target=path.with_name('generation-000000000002');path.rename(target);path=target
    result=catalog.reconcile(tmp_path,inventory)
    assert not result['recovered'] and len(result['rejected'])==1
    assert not (path/'checkpoint.json').exists()
    assert not list(path.glob('rank_*/complete.json'))


def test_partial_completion_is_reused_but_corruption_is_not_overwritten(tmp_path):
    path,inventory=transaction(tmp_path)
    row=catalog.read_checked(path/'rank_0/prepared.json')
    complete={k:v for k,v in row.items() if k not in ('preparation_version','raw_request_id')}
    complete.update(backend=dict(row['backend'],receipt=inventory[0]['receipt']),files=[],save_seconds=1)
    catalog.write_checked(path/'rank_0/complete.json',complete)
    saved=(path/'rank_0/complete.json').read_bytes()
    (path/'rank_3/complete.json').write_text('corrupt')
    result=catalog.reconcile(tmp_path,inventory)
    assert not result['recovered'] and len(result['rejected'])==1
    assert not (path/'rank_1/complete.json').exists()
    assert (path/'rank_0/complete.json').read_bytes()==saved
    assert (path/'rank_3/complete.json').read_text()=='corrupt'


def test_partial_completions_and_retention_filter(tmp_path):
    path,inventory=transaction(tmp_path)
    assert catalog.reconcile(tmp_path,[])['rejected']
    assert catalog.reconcile(tmp_path,inventory)['recovered']==[1]
    (path/'checkpoint.json').unlink()
    (path/'rank_2/complete.json').unlink()
    assert catalog.reconcile(tmp_path,inventory)['recovered']==[1]


def test_corrupt_publication_and_old_unjournaled_reservation_remain_evidence(tmp_path):
    path,inventory=transaction(tmp_path)
    (path/'checkpoint.json').write_text('corrupt')
    assert catalog.reconcile(tmp_path,inventory)['rejected']
    assert (path/'checkpoint.json').read_text()=='corrupt'
    old,_=catalog.reserve(tmp_path,run_id='legacy',step=8)
    result=catalog.reconcile(tmp_path,inventory)
    assert len(result['rejected'])==2 and not (old/'checkpoint.json').exists()
