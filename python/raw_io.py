"""Checkpoint-independent raw NPU-SSD transfer helpers."""

import ctypes
import os
import selectors
import threading
import time
from concurrent.futures import Future

from c_bindings import lib, NPUNVMEContext, NPUNVMERequest
from disk_layout import HEAP_START_OFFSET, BLOCK_SIZE


_EVENTFD_COUNTER_BYTES = 8


def _as_ptr_array(ptrs):
    arr = (ctypes.c_void_p * len(ptrs))()
    for i, ptr in enumerate(ptrs):
        arr[i] = ctypes.c_void_p(ptr)
    return arr


def _as_u64_array(values):
    arr = (ctypes.c_uint64 * len(values))()
    for i, value in enumerate(values):
        arr[i] = ctypes.c_uint64(int(value))
    return arr


def _as_size_array(values):
    arr = (ctypes.c_size_t * len(values))()
    for i, value in enumerate(values):
        arr[i] = ctypes.c_size_t(int(value))
    return arr


def _validate_arrays(ptrs, offsets, sizes):
    if not (len(ptrs) == len(offsets) == len(sizes)):
        raise ValueError("ptrs, offsets and sizes must have the same length")
    if len(ptrs) == 0:
        raise ValueError("at least one I/O item is required")


def _validate_raw_layout(offsets, sizes):
    for offset, size in zip(offsets, sizes):
        offset = int(offset)
        size = int(size)
        if offset < HEAP_START_OFFSET:
            raise ValueError(
                f"raw I/O offset {offset} overlaps protected metadata area")
        if offset % BLOCK_SIZE != 0 or size <= 0 or size % BLOCK_SIZE != 0:
            raise ValueError("raw I/O offsets and sizes must be 4 KiB aligned")


def _split_raw_items(ctx, ptrs, offsets, sizes):
    chunk_size = int(lib.npu_nvme_get_max_transfer(ctx))
    if chunk_size <= 0:
        raise RuntimeError("npu_nvme_get_max_transfer returned an invalid value")

    split_ptrs = []
    split_offsets = []
    split_sizes = []
    for ptr, offset, size in zip(ptrs, offsets, sizes):
        base_ptr = int(ptr)
        base_offset = int(offset)
        remaining = int(size)
        inner_offset = 0
        while remaining > 0:
            take = min(remaining, chunk_size)
            split_ptrs.append(base_ptr + inner_offset)
            split_offsets.append(base_offset + inner_offset)
            split_sizes.append(take)
            inner_offset += take
            remaining -= take
    return split_ptrs, split_offsets, split_sizes


class NPUNVMERequestFuture:
    """Small Future-like wrapper around a C npu_nvme_request_t handle."""

    def __init__(self, req, keepalive=(), result_value=None):
        self._req = req
        self._keepalive = tuple(keepalive)
        self._result_value = result_value
        self._closed = False

    def done(self):
        """Return True when the underlying C request is no longer pending."""
        if self._closed or not self._req:
            return True
        return lib.npu_nvme_request_poll(self._req) != 0

    def result(self, timeout=None):
        """Wait for completion, release the C request, and return result_value."""
        if self._closed or not self._req:
            return self._result_value

        timeout_us = 0 if timeout is None else int(float(timeout) * 1000000)
        wait_rc = lib.npu_nvme_request_wait(self._req, ctypes.c_uint64(timeout_us))
        if wait_rc == -2:
            raise TimeoutError("npu_nvme async request timed out")
        result = lib.npu_nvme_request_result(self._req)
        free_rc = lib.npu_nvme_request_free(self._req)
        self._req = None
        self._keepalive = ()
        self._closed = True
        if wait_rc != 0 or result != 0:
            raise RuntimeError(
                f"npu_nvme async request failed: wait_rc={wait_rc}, result={result}")
        if free_rc != 0:
            raise RuntimeError(f"npu_nvme_request_free failed: rc={free_rc}")
        return self._result_value

    def close(self):
        """Release a completed request if possible."""
        if self._closed or not self._req:
            return
        if lib.npu_nvme_request_poll(self._req) != 0:
            lib.npu_nvme_request_free(self._req)
            self._req = None
            self._keepalive = ()
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _request_address(req):
    return int(ctypes.cast(req, ctypes.c_void_p).value or 0)


class NPUNVMECompletionFuture(Future):
    """Future completed by C eventfd completion notification."""

    def __init__(self, req, keepalive=(), result_value=None):
        super().__init__()
        self._req = req
        self._keepalive = tuple(keepalive)
        self._result_value = result_value
        self._completed_ns = 0

    def request(self):
        """Return the underlying C request handle while it is still owned."""
        return self._req

    def completed_ns(self):
        """Return perf_counter_ns captured when the dispatcher completed it."""
        return self._completed_ns

    def _complete_from_c(self, req=None):
        if self.done():
            return
        if req is not None:
            self._req = req
        if not self._req:
            self.set_exception(RuntimeError("npu_nvme request handle is missing"))
            return

        self._completed_ns = time.perf_counter_ns()
        result = lib.npu_nvme_request_result(self._req)
        free_rc = lib.npu_nvme_request_free(self._req)
        self._req = None
        self._keepalive = ()

        if result != 0:
            self.set_exception(
                RuntimeError(f"npu_nvme async request failed: result={result}"))
        elif free_rc != 0:
            self.set_exception(
                RuntimeError(f"npu_nvme_request_free failed: rc={free_rc}"))
        else:
            self.set_result(self._result_value)


class CompletionDispatcher:
    """Eventfd-driven dispatcher for C-layer async request completions."""

    def __init__(self, ctx, auto_start=False):
        if not hasattr(lib, "npu_nvme_get_completion_fd"):
            raise RuntimeError("completion fd API is unavailable")
        if not hasattr(lib, "npu_nvme_drain_completions"):
            raise RuntimeError("completion drain API is unavailable")

        self.ctx = ctx
        self.fd = int(lib.npu_nvme_get_completion_fd(ctx))
        if self.fd < 0:
            raise RuntimeError("npu_nvme_get_completion_fd failed")

        self._selector = selectors.DefaultSelector()
        self._selector.register(self.fd, selectors.EVENT_READ)
        self._futures = {}
        self._pending = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if auto_start:
            self.start()

    def register(self, req, keepalive=(), result_value=None):
        """Register a submitted C request and return a Future."""
        future = NPUNVMECompletionFuture(req, keepalive, result_value)
        key = _request_address(req)
        if key == 0:
            future.set_exception(RuntimeError("invalid npu_nvme request handle"))
            return future

        if hasattr(lib, "npu_nvme_request_set_user_data"):
            rc = lib.npu_nvme_request_set_user_data(
                req, ctypes.c_uint64(key))
            if rc != 0:
                future.set_exception(
                    RuntimeError("npu_nvme_request_set_user_data failed"))
                return future

        with self._lock:
            pending_req = self._pending.pop(key, None)
            if pending_req is None:
                self._futures[key] = future
                return future

        future._complete_from_c(pending_req)
        return future

    def drain_once(self, max_reqs=64):
        """Drain currently signalled completions without waiting."""
        try:
            os.read(self.fd, _EVENTFD_COUNTER_BYTES)
        except BlockingIOError:
            pass

        completed = 0
        req_array = (ctypes.POINTER(NPUNVMERequest) * int(max_reqs))()
        while True:
            count = lib.npu_nvme_drain_completions(
                self.ctx, req_array, int(max_reqs))
            if count < 0:
                raise RuntimeError("npu_nvme_drain_completions failed")
            if count == 0:
                break
            for i in range(count):
                self._complete_request(req_array[i])
            completed += count
            if count < max_reqs:
                break
        return completed

    def poll(self, timeout=None):
        """Wait for a completion event and dispatch callbacks."""
        events = self._selector.select(timeout)
        if not events:
            return 0
        return self.drain_once()

    def start(self):
        """Start a daemon thread that dispatches completion callbacks."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="npu-nvme-completion", daemon=True)
        self._thread.start()

    def close(self):
        """Stop the optional dispatcher thread and release selector resources."""
        self._stop.set()
        if self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None
        try:
            self._selector.unregister(self.fd)
        except Exception:
            pass
        self._selector.close()

    def _run(self):
        while not self._stop.is_set():
            self.poll(0.1)

    def _complete_request(self, req):
        key = 0
        if hasattr(lib, "npu_nvme_request_user_data"):
            key = int(lib.npu_nvme_request_user_data(req))
        if key == 0:
            key = _request_address(req)

        with self._lock:
            future = self._futures.pop(key, None)
            if future is None:
                self._pending[key] = req
                return

        future._complete_from_c(req)


class RawIO:
    def __init__(self, ctx, completion_dispatcher=None):
        self.ctx = ctx
        self.completion_dispatcher = completion_dispatcher

    def write_host(self, ptrs, offsets, sizes):
        _validate_arrays(ptrs, offsets, sizes)
        _validate_raw_layout(offsets, sizes)
        ptrs, offsets, sizes = _split_raw_items(self.ctx, ptrs, offsets, sizes)
        if not hasattr(lib, "npu_nvme_raw_write_batch_host"):
            raise RuntimeError("raw_write_batch_host is unavailable")
        return lib.npu_nvme_raw_write_batch_host(
            self.ctx, _as_ptr_array(ptrs), _as_u64_array(offsets),
            _as_size_array(sizes), len(ptrs))

    def read_host(self, ptrs, offsets, sizes):
        _validate_arrays(ptrs, offsets, sizes)
        _validate_raw_layout(offsets, sizes)
        ptrs, offsets, sizes = _split_raw_items(self.ctx, ptrs, offsets, sizes)
        if not hasattr(lib, "npu_nvme_raw_read_batch_host"):
            raise RuntimeError("raw_read_batch_host is unavailable")
        return lib.npu_nvme_raw_read_batch_host(
            self.ctx, _as_ptr_array(ptrs), _as_u64_array(offsets),
            _as_size_array(sizes), len(ptrs))

    def write_async(self, npu_ptrs, offsets, sizes):
        """Submit NPU HBM -> NVMe raw writes and return a request future."""
        if not hasattr(lib, "npu_nvme_raw_write_batch_async"):
            raise RuntimeError("raw_write_batch_async is unavailable")
        _validate_arrays(npu_ptrs, offsets, sizes)
        _validate_raw_layout(offsets, sizes)
        npu_ptrs, offsets, sizes = _split_raw_items(
            self.ctx, npu_ptrs, offsets, sizes)
        c_ptrs = _as_ptr_array(npu_ptrs)
        c_offs = _as_u64_array(offsets)
        c_sizes = _as_size_array(sizes)
        req = ctypes.POINTER(NPUNVMERequest)()
        rc = lib.npu_nvme_raw_write_batch_async(
            self.ctx, c_ptrs, c_offs, c_sizes, len(npu_ptrs), ctypes.byref(req))
        if rc != 0 or not req:
            raise RuntimeError(f"npu_nvme_raw_write_batch_async failed, rc={rc}")
        if self.completion_dispatcher is not None:
            return self.completion_dispatcher.register(
                req, (c_ptrs, c_offs, c_sizes))
        return NPUNVMERequestFuture(req, (c_ptrs, c_offs, c_sizes))

    def read_async(self, npu_ptrs, offsets, sizes):
        """Submit NVMe -> NPU HBM raw reads and return a request future."""
        if not hasattr(lib, "npu_nvme_raw_read_batch_async"):
            raise RuntimeError("raw_read_batch_async is unavailable")
        _validate_arrays(npu_ptrs, offsets, sizes)
        _validate_raw_layout(offsets, sizes)
        npu_ptrs, offsets, sizes = _split_raw_items(
            self.ctx, npu_ptrs, offsets, sizes)
        c_ptrs = _as_ptr_array(npu_ptrs)
        c_offs = _as_u64_array(offsets)
        c_sizes = _as_size_array(sizes)
        req = ctypes.POINTER(NPUNVMERequest)()
        rc = lib.npu_nvme_raw_read_batch_async(
            self.ctx, c_ptrs, c_offs, c_sizes, len(npu_ptrs), ctypes.byref(req))
        if rc != 0 or not req:
            raise RuntimeError(f"npu_nvme_raw_read_batch_async failed, rc={rc}")
        if self.completion_dispatcher is not None:
            return self.completion_dispatcher.register(
                req, (c_ptrs, c_offs, c_sizes))
        return NPUNVMERequestFuture(req, (c_ptrs, c_offs, c_sizes))
