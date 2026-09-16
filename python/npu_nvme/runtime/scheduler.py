"""Admission, ordering and drain ownership without framework or storage imports.

Workers own capture/transport and release their lease only after safe completion.
The scheduler never infers DMA completion from a wait deadline.
"""
import math
import threading
import time

from .leases import CheckpointLease


class CheckpointBusyError(RuntimeError):
    """No request/snapshot capacity was available for nonblocking admission."""


class CheckpointQueuePoisonedError(RuntimeError):
    """A previous generation failed; new submissions require explicit reset."""


class CheckpointScheduler:
    def __init__(self, snapshot_slots, request_slots, *, generation=0, clock=time.monotonic):
        if not 0 < snapshot_slots <= 16 or not 0 < request_slots <= 64:
            raise ValueError('snapshot/request slots exceed safety budgets')
        self.snapshot_budget = threading.BoundedSemaphore(snapshot_slots)
        self.request_budget = threading.BoundedSemaphore(request_slots)
        self.changed = threading.Condition(threading.Lock())
        self.leases = set()
        self.closing = False
        self.poisoned = False
        self.request_counter = 0
        self.accepted_generation = generation
        self.order = threading.Condition()
        self.next_io_sequence = 1
        self.handles_lock = threading.Lock()
        self.active_handles = set()
        self.handle_threads = {}
        self.clock = clock

    def _deadline(self, timeout):
        timeout = 120.0 if timeout is None else float(timeout)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError('timeout must be finite and nonnegative')
        return self.clock() + timeout

    def admit(self, timeout=None, admission='block', commit_meta=True):
        if admission not in ('block', 'try'):
            raise ValueError('admission must be block or try')
        deadline = self._deadline(timeout)
        with self.changed:
            while True:
                if self.closing:
                    raise RuntimeError('checkpoint admission is closed')
                if self.poisoned:
                    raise CheckpointQueuePoisonedError(
                        'checkpoint queue is poisoned; reopen the context')
                if self.request_budget.acquire(blocking=False):
                    if self.snapshot_budget.acquire(blocking=False):
                        self.request_counter += 1
                        if commit_meta:
                            self.accepted_generation += 1
                        lease = CheckpointLease(self.request_counter,
                            self.accepted_generation if commit_meta else self.request_counter)
                        self.leases.add(lease)
                        return lease
                    self.request_budget.release()
                if admission == 'try':
                    raise CheckpointBusyError('request or snapshot admission is BUSY')
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise TimeoutError('checkpoint admission timed out')
                self.changed.wait(remaining)

    def release(self, lease):
        with self.changed:
            if lease not in self.leases or lease.released:
                raise RuntimeError('checkpoint lease already released or not owned')
            handle = lease.handle
            if handle is not None:
                with self.handles_lock:
                    self.active_handles.discard(handle)
                    self.handle_threads.pop(handle.request_id, None)
            self.snapshot_budget.release()
            self.request_budget.release()
            lease.released = True
            self.leases.remove(lease)
            self.changed.notify_all()

    def advance(self, sequence):
        with self.order:
            if sequence == self.next_io_sequence:
                self.next_io_sequence += 1
                self.order.notify_all()

    def poison(self, failed_handle):
        with self.changed:
            self.poisoned = True
            self.changed.notify_all()
        with self.handles_lock:
            return [h for h in self.active_handles if h is not failed_handle]

    def reset(self, generation, *, acknowledge=lambda: None):
        with self.changed:
            if self.closing:
                raise RuntimeError('cannot reset closed admission')
            if self.leases:
                raise RuntimeError('cannot reset checkpoint queue while active')
            acknowledge()
            self.poisoned = False
            self.accepted_generation = generation
        with self.order:
            self.next_io_sequence = self.request_counter + 1
            self.order.notify_all()

    def stop_admission(self):
        with self.changed:
            self.closing = True
            self.changed.notify_all()

    def drain(self, timeout=None):
        deadline = self._deadline(timeout)
        with self.changed:
            watermark = set(self.leases)
            while watermark & self.leases:
                remaining = deadline - self.clock()
                if remaining <= 0:
                    raise TimeoutError('background checkpoint I/O timed out')
                self.changed.wait(remaining)
