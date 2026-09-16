"""Draft region-bound backend atop the final versioned transport.

No implicit formatting; immutable extent registration is checked before any
write. Metadata and payload reads are split by actual native chunk capability.
"""
from __future__ import annotations
from pathlib import Path
import json


class RegisteredBackend:
    def __init__(self,transport,registration,*,region_id):
        if isinstance(registration,(str,Path)):registration=json.loads(Path(registration).read_text())
        required={'schema_version','pci_addr','protected_pci_addr','region_id','offset','length','write_authorized','format'}
        if set(registration)!=required:raise ValueError('registration fields differ')
        if (registration['schema_version'],registration['pci_addr'],registration['protected_pci_addr'],registration['format'])!=(1,'0000:83:00.0','0000:84:00.0','D2'):
            raise ValueError('unsupported region registration')
        if registration['region_id']!=region_id or registration['write_authorized'] is not True:raise ValueError('region authorization differs')
        base,length=registration['offset'],registration['length']
        if any(type(x) is not int or x<=0 or x%4096 for x in (base,length)):raise ValueError('unaligned region')
        if base>transport.total_bytes-length:raise ValueError('region exceeds actual namespace')
        # Keep all C1 headers, FULL slots and scratch probes below128GiB.
        if base<128*1024**3:raise ValueError('region overlaps protected legacy area')
        self.transport=transport;self.base=base;self.end=base+length
        self.chunk=transport.capabilities.chunk_size;self.region_id=region_id
        if self.chunk<=0 or self.chunk%4096:raise ValueError('invalid transport geometry')

    def _check(self,offset,length):
        if any(type(x) is not int for x in (offset,length)) or offset%4096 or length<=0 or length%4096 or offset<self.base or offset>self.end-length:raise ValueError('I/O escapes registered extent')

    def read(self,offset,length):
        self._check(offset,length)
        return b''.join(self.transport.read(offset+i,min(self.chunk,length-i)) for i in range(0,length,self.chunk))

    def write(self,offset,raw):
        self._check(offset,len(raw))
        for i in range(0,len(raw),self.chunk):self.transport.write(offset+i,raw[i:i+self.chunk])

    def flush(self):self.transport.flush()


class ReadOnlyV2:
    """Offline parser facade cannot reach a write method on its backend."""
    def __init__(self,reader,total_bytes):self.reader=reader;self.total_bytes=total_bytes
    def read(self,offset,length):
        if type(offset) is not int or type(length) is not int or offset<0 or length<=0 or offset>self.total_bytes-length:raise ValueError('V2 read extent')
        return self.reader(offset,length)
    def write(self,*args):raise PermissionError('V2 migration source is read-only')
    def flush(self):raise PermissionError('V2 migration source is read-only')
