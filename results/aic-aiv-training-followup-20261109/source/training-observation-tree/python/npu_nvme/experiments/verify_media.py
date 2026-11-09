"""Readback verification outside the timed saving path."""
import hashlib
import json

from npu_nvme.experiments.raw_store import ALIGN, SUPER_BYTES, unpack_commit, unpack_frame_prefix


def verify_receipt(transport, receipt, *, chunk_bytes=4 << 20, consume=None):
    rows=receipt['ranks']
    if len(rows)!=4 or {row['rank'] for row in rows}!=set(range(4)):
        raise ValueError('receipt requires four distinct ranks')
    spans=[]
    verified=[]
    for row in rows:
        offset=row['frame_offset']; prefix_bytes=row['prefix_bytes']; payload_bytes=row['payload_bytes']
        end=offset+row['frame_bytes']
        commit_offset=row['commit_offset']
        if (offset<SUPER_BYTES or offset%ALIGN or prefix_bytes%ALIGN or
                prefix_bytes<ALIGN or payload_bytes<=0 or end>transport.total_bytes or
                row['payload_offset']!=offset+prefix_bytes or
                row['frame_bytes']!=prefix_bytes+((payload_bytes+ALIGN-1)//ALIGN)*ALIGN or
                commit_offset<SUPER_BYTES or commit_offset%ALIGN or commit_offset+ALIGN>transport.total_bytes):
            raise ValueError('receipt frame bounds differ')
        spans.extend([(offset,end),(commit_offset,commit_offset+ALIGN)])
    spans.sort()
    if any(left[1]>right[0] for left,right in zip(spans,spans[1:])):
        raise ValueError('receipt media ranges overlap')
    for row in sorted(rows,key=lambda item:item['rank']):
        prefix=b''.join(transport.read(row['frame_offset']+offset,min(chunk_bytes,row['prefix_bytes']-offset))
                        for offset in range(0,row['prefix_bytes'],chunk_bytes))
        frame=unpack_frame_prefix(prefix)
        if row.get('descriptor_sha256') is not None and hashlib.sha256(json.dumps(frame['descriptor'],sort_keys=True,separators=(',',':')).encode()).hexdigest()!=row['descriptor_sha256']:
            raise ValueError('media descriptor identity differs')
        commit=unpack_commit(transport.read(row['commit_offset'],ALIGN))
        if (frame['generation']!=receipt['generation'] or frame['step']!=receipt['step'] or
                frame['payload_bytes']!=row['payload_bytes'] or
                frame['payload_sha256']!=row['payload_sha256'] or
                commit['generation']!=receipt['generation'] or
                commit['frame_offset']!=row['frame_offset'] or
                commit['frame_bytes']!=row['prefix_bytes']+row['payload_bytes']):
            raise ValueError('frame/commit identity differs')
        frame_digest=hashlib.sha256(prefix);payload_digest=hashlib.sha256()
        for offset in range(0,row['payload_bytes'],chunk_bytes):
            take=min(chunk_bytes,row['payload_bytes']-offset)
            data=transport.read(row['payload_offset']+offset,take)
            if len(data)!=take:raise ValueError('short media read')
            frame_digest.update(data);payload_digest.update(data)
            if consume is not None:consume(row['rank'],frame['descriptor'],offset,data)
        if (payload_digest.hexdigest()!=frame['payload_sha256'] or
                frame_digest.hexdigest()!=commit['frame_sha256'] or
                commit['frame_sha256']!=row['frame_sha256']):
            raise ValueError('media checksum differs')
        verified.append(dict(rank=row['rank'],payload_bytes=row['payload_bytes'],verified=True))
    return dict(generation=receipt['generation'],step=receipt['step'],ranks=verified,status='pass')
