"""Bounded, read-only strict D1 image source for offline migration."""
import os
from .metadata import mount_readonly
from npu_nvme.runtime.d1_schema import validate_record


class D1Image:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        self.total_bytes = os.fstat(self.fd).st_size
        self.storage_identity = ('file', os.fstat(self.fd).st_dev, os.fstat(self.fd).st_ino)
        try:
            self.state = mount_readonly(self.read, self.total_bytes)
        except BaseException:
            self.close()
            raise

    def read(self, offset, size):
        if type(offset) is not int or type(size) is not int or offset < 0 or size <= 0 or offset > self.total_bytes-size:
            raise ValueError('D1 image read extent')
        data = os.pread(self.fd, size, offset)
        if len(data) != size:
            raise ValueError('short D1 image read')
        return data

    def select(self, step=None):
        return select_record(self.state, step)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def select_record(state, step=None):
    candidates = []
    for key, record in state.meta_dict.get('checkpoints', {}).items():
        if step is None or record.get('state_step') == step:
            candidates.append((record.get('generation', -1), str(key), record))
    if not candidates:
        raise ValueError('no strict D1 FULL source record for requested step')
    _, key, record = max(candidates, key=lambda row: row[:2])
    # Do not silently migrate older data when the selected record is damaged.
    validate_record(record, state.layout)
    return key, record
