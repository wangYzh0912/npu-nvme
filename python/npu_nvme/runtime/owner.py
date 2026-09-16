"""Bounded process and filesystem ownership; the file is never unlinked."""
import fcntl
import os
import re
import threading
import time


class OwnerBusy(RuntimeError):
    pass


class OwnerLock:
    _guard = threading.Lock()
    _held = set()

    def __init__(self, pci_addr, namespace=1, directory='/tmp'):
        if not re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]', pci_addr):
            raise ValueError('PCI address must include domain, bus, device and function')
        if namespace != 1:
            raise ValueError('only namespace 1 is supported')
        self.key = f'{pci_addr.lower()}-ns{namespace}'
        self.path = os.path.join(directory, 'npu-nvme-python-owner-' + self.key)
        self.fd = None
        self.pid = None

    def acquire(self):
        with self._guard:
            if self.fd is not None or self.key in self._held:
                raise OwnerBusy(self.key)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                os.close(fd)
                raise OwnerBusy(self.key) from error
            except BaseException:
                os.close(fd)
                raise
            self.fd, self.pid = fd, os.getpid()
            self._held.add(self.key)
        return self

    def release(self):
        with self._guard:
            if self.fd is None:
                return
            if self.pid != os.getpid():
                raise RuntimeError('an inherited owner cannot release the parent lock')
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None
            self._held.remove(self.key)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()
