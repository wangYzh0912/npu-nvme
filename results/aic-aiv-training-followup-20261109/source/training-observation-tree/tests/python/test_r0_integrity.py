import numpy as np
import pytest
import copy
from incremental_manifest import build_training_state_manifest
from incremental_frame import pack_r0_frame, unpack_r0_frame
from r0_session import R0Session, state_digest


def session():
    class Parameter:
        shape = (4,)
        dtype = np.float32
    class Model:
        def parameters_and_names(self):
            return [('x', Parameter()), ('y', Parameter())]
    manifest = build_training_state_manifest({'model': Model()}, block_size=2, small_threshold=0)
    state = {f.canonical_name: np.zeros(4, np.float32) for f in manifest.fields}
    return R0Session(state, manifest)


def repack(frame, **changes):
    info = unpack_r0_frame(frame)
    args = {k: info[k] for k in ('step','generation','base_full_generation','base_delta_generation','manifest_digest','world_size','rank_id')}
    args.update(block_records=info['blocks'], control_records=info['controls'])
    args.update(changes)
    return pack_r0_frame(**args)


def changed(s):
    state = {k:v.copy() for k,v in s.current.items()}
    for v in state.values(): v[0] = 1
    s.set_current(state)
    return s.observe(8,1)


@pytest.mark.parametrize('change', [dict(base_full_generation=999),dict(step=9),dict(world_size=2),dict(manifest_digest='1'*64)])
def test_ack_requires_exact_pending_identity(change):
    s=session();f=changed(s);before=state_digest(s.persisted)
    with pytest.raises(ValueError):s.ack(repack(f,**change))
    assert state_digest(s.persisted)==before
    assert s.in_flight_generation==1


def test_duplicate_ack_is_idempotent():
    s=session();f=changed(s);result=s.ack(f)
    assert s.ack(f)==result


def test_second_invalid_record_cannot_partially_advance():
    s=session();f=changed(s);info=unpack_r0_frame(f)
    info['blocks'][1]['element_offset']=999
    bad=repack(f,block_records=info['blocks'])
    before=state_digest(s.persisted)
    with pytest.raises((ValueError,IndexError)):s.ack(bad)
    assert state_digest(s.persisted)==before


def test_signed_zero_is_a_byte_change():
    s=session();state={k:v.copy() for k,v in s.current.items()};state['model/x'][0]=-0.0
    s.set_current(state);f=s.observe(8,1);s.ack(f)
    assert state_digest(s.recover([f])['state'])==state_digest(state)


def test_manifest_export_cannot_mutate_next_export():
    s=session();before=copy.deepcopy(s.manifest.as_dict());export=s.manifest.as_dict()
    export['fields'][0]['shape'][0]=999
    assert s.manifest.as_dict()==before


def test_replay_rejects_foreign_full_root():
    s=session();f=changed(s)
    with pytest.raises(ValueError):s.recover([repack(f,base_full_generation=999)])


def test_replay_accepts_generation_gaps_with_correct_parent():
    s=session();f=changed(s)
    replay=s.recover([repack(f,generation=1000)])
    assert replay['generation']==1000


@pytest.mark.parametrize('key',['state_index','block_index','element_offset'])
def test_negative_block_identity_rejected_before_replay(key):
    s=session();f=changed(s);info=unpack_r0_frame(f);info['blocks'][0][key]=-1
    with pytest.raises(ValueError):unpack_r0_frame(repack(f,block_records=info['blocks']))


def test_empty_digest_cannot_encode_identity():
    s=session();f=changed(s)
    with pytest.raises(ValueError):repack(f,manifest_digest='')


def test_nonzero_header_padding_is_rejected():
    f=bytearray(changed(session()));f[1024]=1
    with pytest.raises(ValueError):unpack_r0_frame(f)


def test_prefix_rejects_negative_descriptor_before_payload():
    from incremental_frame import pack_r0_frame_prefix,unpack_r0_frame_prefix
    s=session();info=unpack_r0_frame(changed(s));blocks=copy.deepcopy(info['blocks'])
    blocks[0]['element_offset']=-1
    for block in blocks:block.pop('value')
    prefix=pack_r0_frame_prefix(step=8,generation=1,base_full_generation=1,base_delta_generation=0,
        manifest_digest=s.manifest.digest,blocks=blocks,controls=[],payload_bytes=16,payload_crc=0)
    with pytest.raises(ValueError):unpack_r0_frame_prefix(prefix)


def test_duplicate_block_identity_rejected():
    s=session();f=changed(s);info=unpack_r0_frame(f)
    info['blocks'][1].update(state_index=info['blocks'][0]['state_index'],block_index=info['blocks'][0]['block_index'])
    with pytest.raises(ValueError):unpack_r0_frame(repack(f,block_records=info['blocks']))


def test_initial_geometry_matches_manifest():
    s=session();bad={k:v.copy() for k,v in s.base_state.items()};bad['model/x']=np.zeros(3,np.float32)
    with pytest.raises(ValueError):R0Session(bad,s.manifest)


def test_ring_huge_generation_gap_is_bounded():
    from raw_ring import pack_ring_slot,select_recovery_chain,KIND_FULL,KIND_DELTA
    slots=[pack_ring_slot(b'root',1,1,KIND_FULL),pack_ring_slot(b'delta',2**63,2,KIND_DELTA)]
    with pytest.raises(ValueError,match='missing generation'):select_recovery_chain(slots)


def test_file_ring_restart_keeps_cursor(tmp_path):
    from s2_delta import S2DeltaOracle,FileS2Ring
    initial={'backbone.blocks.0.weight':np.zeros(4,np.float32)}
    oracle=S2DeltaOracle(initial,block_size=2,small_threshold=0)
    first=oracle.observe(1);ring=FileS2Ring(tmp_path,2,1<<20)
    stale=FileS2Ring(tmp_path,2,1<<20)
    assert ring.write(first)==0;oracle.ack(first)
    with pytest.raises(ValueError,match='generation'):stale.write(first)
    second=oracle.observe(2)
    assert FileS2Ring(tmp_path,2,1<<20).write(second)==1
    assert ring.read(0)==first
