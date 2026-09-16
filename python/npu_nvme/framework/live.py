"""Generation-owned pinned staging and optimizer fences (legacy live contract)."""
import ctypes
import time
import threading
import hashlib
import numpy as np

class LiveCapture:
    def __init__(self, *, acl, device_id, chunk_size, is_quarantined, retain):
        self.acl = acl
        self.npu_device_id = device_id
        self.chunk_size = chunk_size
        self.is_quarantined = is_quarantined
        self.retain = retain
        self.lock = threading.Lock()
        self.handles = set()
        self.quarantined = []

    def release(self, handle):
        if self.is_quarantined(handle):
            return
        with self.lock:
            resources = list(handle._live_buffers)
            handle._live_buffers = []
            self.handles.discard(handle)
        for resource in resources:
            events = [handle._live_pre_event, handle._live_post_event,
                      resource.get("event")]
            seen_events = set()
            for event in events:
                if event and event.value:
                    if int(event.value) in seen_events:
                        continue
                    seen_events.add(int(event.value))
                    self.acl.aclrtDestroyEvent(event)
            stream = resource.get("stream")
            if stream and stream.value:
                self.acl.aclrtDestroyStream(stream)
            buffer_ptr = resource.get("buffer")
            if buffer_ptr and buffer_ptr.value:
                self.acl.aclrtFreeHost(buffer_ptr)
        handle._live_event = None
        handle._live_pre_event = None
        handle._live_post_event = None

    def maybe_release(self, handle):
        if handle.done() and handle._live_fence_consumed:
            self.release(handle)

    @staticmethod
    def _align_live_offset(value, alignment=64):
        return (int(value) + alignment - 1) // alignment * alignment

    def stage(self, params, request_id=None):
        if self.acl is None:
            raise RuntimeError("ACL library is required for live staging")
        rc = self.acl.aclrtSetDevice(self.npu_device_id)
        if rc != 0:
            raise RuntimeError(f"aclrtSetDevice(live) failed: {rc}")
        offsets = []
        total = 0
        for item in params:
            total = self._align_live_offset(total)
            offsets.append(total)
            total += int(item["size"])
        buffer_ptr = ctypes.c_void_p()
        stream = ctypes.c_void_p()
        event = ctypes.c_void_p()
        dma_chunks = []
        allocated = False
        try:
            rc = self.acl.aclrtMallocHost(ctypes.byref(buffer_ptr), total)
            if rc != 0 or not buffer_ptr.value:
                raise MemoryError(f"aclrtMallocHost({total}) failed: {rc}")
            allocated = True
            rc = self.acl.aclrtCreateStream(ctypes.byref(stream))
            if rc != 0:
                raise RuntimeError(f"aclrtCreateStream(live) failed: {rc}")
            rc = self.acl.aclrtCreateEvent(ctypes.byref(event))
            if rc != 0:
                raise RuntimeError(f"aclrtCreateEvent(live) failed: {rc}")
            dma_submit_ns = time.monotonic_ns()
            staged = []
            for item, offset in zip(params, offsets):
                target = int(buffer_ptr.value) + offset
                copy_item = dict(item)
                array_type = ctypes.c_ubyte * int(item["size"])
                array = np.ctypeslib.as_array(array_type.from_address(target))
                if item["ptr"]:
                    inner_offset = 0
                    while inner_offset < int(item["size"]):
                        take = min(self.chunk_size,
                                   int(item["size"]) - inner_offset)
                        submit_ns = time.monotonic_ns()
                        rc = self.acl.aclrtMemcpyAsync(
                            ctypes.c_void_p(target + inner_offset), take,
                            ctypes.c_void_p(item["ptr"] + inner_offset), take,
                            2, stream)
                        if rc != 0:
                            raise RuntimeError(
                                f"live D2H submit failed for {item['name']}: {rc}")
                        dma_chunks.append({
                            "name": item["name"], "offset": inner_offset,
                            "size": take, "submit_ns": submit_ns,
                            "complete_ns": None,
                            "completion_event_scope": "generation",
                        })
                        inner_offset += take
                else:
                    source = np.ascontiguousarray(item["np_arr"]).view(np.uint8).reshape(-1)
                    np.copyto(array, source)
                copy_item.update(ptr=target, np_arr=array,
                                 placement="host_staging")
                staged.append(copy_item)
            rc = self.acl.aclrtRecordEvent(event, stream)
            if rc != 0:
                raise RuntimeError(f"aclrtRecordEvent(live) failed: {rc}")
            return staged, {"buffer": buffer_ptr, "stream": stream,
                            "event": event, "dma_chunks": dma_chunks, "bytes": total,
                            "dma_submit_ns": dma_submit_ns}
        except BaseException:
            if dma_chunks:
                resources = getattr(self, "quarantined", [])
                resources.append(dict(request_id=request_id, buffer=buffer_ptr, stream=stream, event=event,
                                      params=params, dma_chunks=dma_chunks))
                self.quarantined = resources
                self.retain(self)
                raise
            if event.value:
                self.acl.aclrtDestroyEvent(event)
            if stream.value:
                self.acl.aclrtDestroyStream(stream)
            if allocated:
                self.acl.aclrtFreeHost(buffer_ptr)
            raise

    def install_update_fence(self, handle, stream_ptr):
        """Insert a device-side wait before the next optimizer launch."""
        if not handle._live_event:
            return False
        stream = ctypes.c_void_p(int(stream_ptr))
        status = ctypes.c_int()
        rc = self.acl.aclrtQueryEventStatus(handle._live_event,
                                           ctypes.byref(status))
        if rc != 0:
            raise RuntimeError(f"aclrtQueryEventStatus(update fence) failed: {rc}")
        handle.update_deadline_missed = (status.value == 0)
        pre = ctypes.c_void_p()
        post = ctypes.c_void_p()
        for event in (pre, post):
            rc = self.acl.aclrtCreateEvent(ctypes.byref(event))
            if rc != 0:
                raise RuntimeError(f"aclrtCreateEvent(update fence) failed: {rc}")
        rc = self.acl.aclrtRecordEvent(pre, stream)
        if rc == 0:
            rc = self.acl.aclrtStreamWaitEvent(stream, handle._live_event)
        if rc == 0:
            rc = self.acl.aclrtRecordEvent(post, stream)
        if rc != 0:
            self.acl.aclrtDestroyEvent(pre)
            self.acl.aclrtDestroyEvent(post)
            raise RuntimeError(f"device update fence submission failed: {rc}")
        handle._live_pre_event = pre
        handle._live_post_event = post
        handle.update_fence_install_ns = time.monotonic_ns()
        return True

    def collect_update_wait(self, handle):
        """Collect device event-to-event wait after the optimizer step ends."""
        if not handle._live_post_event:
            return 0
        rc = self.acl.aclrtSynchronizeEvent(handle._live_post_event)
        elapsed_ms = ctypes.c_float()
        if rc == 0:
            rc = self.acl.aclrtEventElapsedTime(
                ctypes.byref(elapsed_ms), handle._live_pre_event,
                handle._live_post_event)
        if rc != 0:
            raise RuntimeError(f"update fence timing failed: {rc}")
        handle.update_wait_ns = max(0, int(elapsed_ms.value * 1_000_000))
        handle.update_fence_release_ns = time.monotonic_ns()
        handle._live_fence_consumed = True
        self.maybe_release(handle)
        return handle.update_wait_ns

    def complete(self, handle, _live_staging, params, layout):
        rc = self.acl.aclrtSynchronizeEvent(_live_staging["event"])
        if rc != 0:
            raise RuntimeError(f"live D2H event failed (rc={rc})")
        handle.dma_complete_ns = time.monotonic_ns()
        dma_chunks = []
        for chunk in _live_staging.get("dma_chunks", []):
            dma_chunks.append({
                "name": chunk["name"], "offset": chunk["offset"],
                "size": chunk["size"],
                "submit_ns": chunk["submit_ns"],
                "complete_ns": handle.dma_complete_ns,
                "completion_event_scope": "generation",
            })
        handle.dma_chunks = dma_chunks
        layout_by_name = {item["name"]: item for item in layout}
        state_digest = hashlib.sha256()
        fields = 0
        total_state_bytes = 0
        for item in params:
            view = item.get("np_arr")
            if view is not None:
                raw = np.ascontiguousarray(view).reshape(-1).tobytes()
                item["sha256"] = hashlib.sha256(raw).hexdigest()
                layout_by_name[item["name"]]["sha256"] = item["sha256"]
        for item in sorted(params, key=lambda value: value["name"]):
            if item.get("category") != "parameter":
                continue
            raw = np.ascontiguousarray(
                item["np_arr"]).reshape(-1).tobytes()
            state_digest.update(item["name"].encode())
            state_digest.update(raw)
            fields += 1
            total_state_bytes += int(item["size"])
        handle.snapshot_state_digest = {
            "sha256": state_digest.hexdigest(), "fields": fields,
            "bytes": total_state_bytes}
        checksum = hashlib.sha256()
        for item in sorted(params, key=lambda value: value["name"]):
            checksum.update(item["name"].encode("utf-8"))
            checksum.update(str(item.get("sha256") or "").encode("ascii"))
        handle.checksum = checksum.hexdigest()
