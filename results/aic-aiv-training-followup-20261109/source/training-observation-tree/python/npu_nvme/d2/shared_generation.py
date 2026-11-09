"""Commit a rank-owned shared slot without a second Host payload buffer."""
import ctypes as C
import hashlib

def payload_from_shared(region,shared,lease,*,declared_sha256):
    with region.lock:
        if region.poisoned or not region.pending:raise RuntimeError('no writable transaction')
        view=shared.pool.view(lease);length=lease['length'];pointer=C.addressof(view)
        if type(declared_sha256) is not str or len(declared_sha256)!=64:raise ValueError('declared digest')
        actual=hashlib.sha256(memoryview(view).cast('B')).hexdigest()
        if actual!=declared_sha256:raise ValueError('rank shared payload digest differs')
        padded=(length+4095)//4096*4096
        if padded>shared.pool.plan.chunk_bytes:raise ValueError('slot padding exceeds chunk')
        try:
            extent=region._reserve(padded)
            # Region backend enforces the independent registered D2 extent.
            region.backend._check(extent.offset,extent.length)
            result=shared.transfer(read=False,disk_offset=extent.offset,lease=lease)
            if result!=actual:raise ValueError('shared payload changed during NVMe write')
            raw=(C.c_ubyte*padded).from_address(pointer)
            ref=dict(offset=extent.offset,length=padded,sha256=hashlib.sha256(memoryview(raw).cast('B')).hexdigest(),logical_bytes=length,logical_sha256=actual)
            region.pending['payload'].append(ref)
            return ref
        except BaseException:
            region.poisoned=True
            raise
