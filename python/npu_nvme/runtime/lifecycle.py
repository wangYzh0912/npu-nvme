"""Legacy bounded close orchestration; incomplete drains retain all owners."""
import time
import math


def cleanup_legacy(*, scheduler, lib, ctx, initialized, drain, live_quarantine,
                   live_handles, release_live, retain, timeout=120.0):
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("timeout must be finite and nonnegative")
    deadline = time.monotonic() + timeout
    scheduler.stop_admission()
    if live_quarantine:
        raise RuntimeError("live DMA stop is unproven; context retained")
    # On incomplete drain retain context, snapshots and live DMA resources.
    drain(timeout=max(0.0, deadline-time.monotonic()))
    if ctx and hasattr(lib, "npu_nvme_wait_quiescent"):
        milliseconds = max(1, int(max(0.0, deadline-time.monotonic()) * 1000))
        rc = lib.npu_nvme_wait_quiescent(ctx, milliseconds)
        if rc != 0:
            raise RuntimeError(f"Reactor did not become quiescent (rc={rc})")
    for handle in list(live_handles):
        if handle._live_post_event:
            handle.collect_update_wait()
        release_live(handle)
    if initialized and ctx:
        if not hasattr(lib, "npu_nvme_close"):
            raise RuntimeError("bounded native close unavailable; context retained")
        milliseconds = max(1, int(max(0.0, deadline-time.monotonic()) * 1000))
        rc = lib.npu_nvme_close(ctx, milliseconds)
        if rc != 0:
            retain()
            raise RuntimeError(f"native close incomplete (rc={rc}); context retained")
        lib.npu_nvme_cleanup(ctx)
        return True
    return False
