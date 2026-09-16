"""Draft global unready-target restore using pinned immutable generation."""
import hashlib
from . import manifest_schema as manifest_schema


def restore(region,targets,*,generation=None,expected_topology,expected_schema,chunk_bytes,require_controls=True):
    if len(targets)!=expected_topology['world_size']:raise ValueError('target count')
    with region.selected(generation) as selected:
        pinned_root=next(root for root,count in region.pins.values() if count and selected in root['state']['generations'])
        if selected['topology']!=expected_topology:raise ValueError('no implicit reshard')
        for target in targets:
            if target.ready:raise ValueError('restore requires fresh unready targets')
        try:
            rows=list(region.codec.read(region.backend,selected['pages'],region.data_base,region.end))
            manifest_schema.validate(rows,expected_schema,world_size=len(targets),chunk_bytes=chunk_bytes)
            owned={(ref['offset'],ref['length'],ref['sha256']) for ref in selected['extents']}
            for row in rows:
                ref=row['payload']
                if (ref['offset'],ref['length'],ref['sha256']) not in owned:raise ValueError('unowned payload')
            controls=region.read_controls(selected)
            if require_controls:
                if len(controls)!=len(targets) or {v.get('rank') for v in controls}!=set(range(len(targets))):raise ValueError('missing rank controls')
                if any(v.get('step')!=selected['step'] or type(v.get('controls')) is not dict for v in controls):raise ValueError('invalid rank controls')
            cursors={};names=set()
            for row in rows:
                rank=row['rank'];name=row['name'];offset=row['logical_offset'];length=row['logical_bytes']
                if type(rank) is not int or not 0<=rank<len(targets):raise ValueError('manifest rank')
                key=(rank,name)
                if offset!=cursors.get(key,0):raise ValueError('duplicate/missing tensor slice')
                ref=row['payload']
                if ref['logical_bytes']!=length:raise ValueError('payload logical shape')
                raw=region.backend.read(ref['offset'],ref['length'])
                data=raw[:length]
                if len(raw)!=ref['length'] or hashlib.sha256(data).hexdigest()!=ref['logical_sha256'] or any(raw[length:]):raise ValueError('payload integrity')
                targets[rank].apply_chunk(name,offset,data);cursors[key]=offset+length;names.add(key)
            for value in controls:targets[value['rank']].set_controls(value['controls'])
            for rank,target in enumerate(targets):
                target.finish_restore()
                if target.transport_safe is not True or target.verify_controls() is not True:raise ValueError('rank not locally verified')
            # No ready token is returned before every target has verified.
            token=object()
            for target in targets:target.mark_ready(token)
            if not all(target.ready for target in targets):raise ValueError('global ready publication failed')
            return targets
        except BaseException:
            for target in targets:
                target.ready=False
                if target.transport_safe:target.discard()
            # Production integration must retain a pin for any unsafe target.
            # This draft intentionally refuses to silently unpin unknown DMA.
            if any(not t.transport_safe for t in targets):
                root=pinned_root;key=root['sequence'];old=region.pins.get(key,(root,0));region.pins[key]=(root,old[1]+1)
            raise
