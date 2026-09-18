"""Draft region-bound backend atop the final versioned transport.

No implicit formatting; immutable extent registration is checked before any
write. Metadata and payload reads are split by actual native chunk capability.
"""
from __future__ import annotations
from pathlib import Path
import json


_STAGE_FIELDS = {'schema_version','pci','protected_pci_addr','region_id','write_authorized',
                 'offset_bytes','length_bytes','chunk_bytes','retention','transport','owner','ranks','npu_ids'}


def stage_config_from_config(config, *, required_purpose=None):
    """Load a bounded D2 stage configuration before any native I/O.

    Validation-only callers must name the validation purpose explicitly.  This
    prevents a fault harness from silently falling back to the formal Qwen
    extent when its command line or configuration evolves.
    """
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    allowed = _STAGE_FIELDS | {'purpose'}
    if set(config) not in (_STAGE_FIELDS, allowed) or config['schema_version'] != 1:
        raise ValueError('invalid D2 stage configuration')
    if required_purpose is not None and config.get('purpose') != required_purpose:
        raise ValueError('D2 configuration purpose differs')
    if config['transport'] != 'socket_host_bridge' or config['owner'] != 'single_spdk_nvme_owner':
        raise ValueError('unsupported D2 transport/owner')
    if config['ranks'] != [0,1,2,3] or config['npu_ids'] != [0,1,2,3]:
        raise ValueError('D2 Qwen requires explicit TP4 mapping')
    if config['retention'] != 3 or config['chunk_bytes'] not in (1<<20,4<<20,16<<20):
        raise ValueError('invalid D2 retention/chunk configuration')
    return dict(config)


def registration_from_config(config, *, required_purpose=None):
    """Translate a checked stage configuration into the backend contract."""
    config = stage_config_from_config(config, required_purpose=required_purpose)
    return dict(schema_version=1,pci_addr=config['pci'],protected_pci_addr=config['protected_pci_addr'],
                region_id=config['region_id'],offset=config['offset_bytes'],length=config['length_bytes'],
                write_authorized=config['write_authorized'],format='D2')


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
        raw=bytes(raw)
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
