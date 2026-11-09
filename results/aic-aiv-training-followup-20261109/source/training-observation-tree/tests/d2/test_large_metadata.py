"""Large logical-state CPU metadata test; allocates no tensor payload."""
from npu_nvme.d2.manifest_schema import validate
from npu_nvme.d2.format import PageCodec,HEADER,unpack


def test_70000_descriptors_and_over_64gib_have_independent_budgets():
    count=70000;chunk=1<<20;total=count*chunk
    expected=[dict(rank=0,name='large',dtype='uint8',shape=[total],partition='sharded',bytes=total)]
    def rows():
        for i in range(count):
            yield dict(rank=0,name='large',dtype='uint8',shape=[total],partition='sharded',logical_offset=i*chunk,
                logical_bytes=chunk,payload=dict(offset=(i+1)*chunk,length=chunk,logical_bytes=chunk,
                    sha256='a'*64,logical_sha256='a'*64))
    report=validate(rows(),expected,world_size=1,chunk_bytes=chunk)
    assert report['chunks']==70000 and report['bytes']>64<<30
    codec=PageCodec();observed=0;pages=0
    for page in codec.pages(rows()):
        assert len(page)==65536
        observed+=len(unpack('manifest',page)['rows']);pages+=1
    assert observed==count and pages>256
