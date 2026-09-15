"""Read and validate the committed strict catalog without payload writes."""
from dataclasses import asdict
from npu_nvme.runtime.commit import MetadataState
from npu_nvme.runtime.d1_commit import D1CommitCoordinator
from .bindings import load_backend
from .full_transport import FullTransport


def inspect_disk(pci, npu=7, library_path=None):
    backend = load_backend(library_path)
    transport = FullTransport(backend, pci=pci, npu=npu, depth=1,
        chunk_size=1024*1024, profiling_dir='.', profiling=False)
    try:
        state = MetadataState()
        transport.metadata.mount(state, transport.total_bytes, 0)
        catalog = D1CommitCoordinator(metadata_io=transport.metadata, state=state)
        records = sorted(state.meta_dict['checkpoints'].values(), key=lambda r:r['generation'])
        return {'status':'pass', 'validation':'metadata-and-manifest-only',
                'payload_verified':False, 'strict_contract':'D1',
                'metadata_generation':state.metadata_generation,
                'active_meta_slot':state.active_meta_slot, 'layout':asdict(state.layout),
                'retained_generations':[r['generation'] for r in records],
                'checkpoints':[{'generation':r['generation'],'step':r['state_step'],
                    'slot':r['slot'],'request_id':r['request_id'],
                    'manifest_sha256':r['manifest_sha256'],'fields':len(r['params'])} for r in records]}
    finally:
        transport.close(120)
