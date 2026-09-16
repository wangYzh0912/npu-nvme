"""An independent consumer cannot retire the producer's live shared memory."""
import json
from multiprocessing import shared_memory
from pathlib import Path
import subprocess
import sys


def test_consumer_exit_preserves_owner_mapping():
    owner=shared_memory.SharedMemory(create=True,size=16)
    repo=Path(__file__).resolve().parents[2]
    descriptor=dict(name=owner.name,fields=[dict(name='model/x',offset=0,nbytes=8,
        shape=[2],dtype='<i4')],controls=dict(offset=8,nbytes=8))
    source='''
import json,sys
from experiments.baselines.repro.workers.common import attach_arrays
shm,arrays=attach_arrays(json.loads(sys.argv[1]))
arrays['model/x'][:]=[12,34]
arrays['controls/state'][:]=7
shm.close()
'''
    try:
        for _ in range(2):
            p=subprocess.run([sys.executable,'-c',source,json.dumps(descriptor)],
                cwd=repo,capture_output=True,text=True,timeout=15)
            assert p.returncode==0,p.stderr
            assert 'resource_tracker' not in p.stderr
            with_mapping=shared_memory.SharedMemory(name=owner.name)
            assert bytes(with_mapping.buf[8:])==b'\x07'*8
            with_mapping.close()
    finally:
        owner.close();owner.unlink()
