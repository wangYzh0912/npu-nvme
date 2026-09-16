"""Legacy FULL ordering, publication and release; dependencies are explicit."""
import time
import os
import pickle
from dataclasses import dataclass
from .handle import CheckpointHandle
from npu_nvme.types import CheckpointState

@dataclass(frozen=True)
class WorkerServices:
    scheduler: object
    io_mutex: object
    transport: object
    commit: object
    flush: object
    publish: object
    complete_live: object
    release_snapshot: object
    release_live: object
    poison: object
    record_error: object
    record_layout: object
    retain: object

@dataclass(frozen=True)
class TransferJob:
    io_mode: object
    io_sequence: object
    handle: object
    lease: object
    _live_staging: object
    params: object
    layout: object
    commit_meta: object
    step: object
    _checkpoint_meta: object
    meta_path: object
    t_start: object
    T_Prep: object
    T_Layout: object
    rank_id: object


def run_legacy_transfer(services, job, c_ptrs_d, c_offs_d, c_sizes_d, n_dev, d_sz,
                        c_ptrs_h, c_offs_h, c_sizes_h, n_host, h_sz):
    io_mode = job.io_mode
    io_sequence = job.io_sequence
    handle = job.handle
    lease = job.lease
    _live_staging = job._live_staging
    params = job.params
    layout = job.layout
    commit_meta = job.commit_meta
    step = job.step
    _checkpoint_meta = job._checkpoint_meta
    meta_path = job.meta_path
    t_start = job.t_start
    T_Prep = job.T_Prep
    T_Layout = job.T_Layout
    rank_id = job.rank_id
    owns_io_mutex = False
    try:
        with services.scheduler.order:
            while (io_sequence != services.scheduler.next_io_sequence and
                   not services.scheduler.poisoned and
                   handle.state != CheckpointState.CANCELLED):
                services.scheduler.order.wait(timeout=0.1)
        if services.scheduler.poisoned or handle.state == CheckpointState.CANCELLED:
            if not handle.done():
                handle.status = CheckpointHandle.CANCELLED
                handle.transition(CheckpointState.CANCELLED)
                handle._done.set()
            return
        services.io_mutex.acquire()
        owns_io_mutex = True
        if services.scheduler.poisoned or handle.state == CheckpointState.CANCELLED:
            return
        if _live_staging is not None:
            services.complete_live(handle, _live_staging, params, layout)
        if _live_staging is None:
            handle.transition(CheckpointState.DMA_COPYING)
        t_spdk_start = time.perf_counter()
        total_written = 0

        if n_dev > 0:
            handle.acl_event = "c-layer-per-dma-slot"
            services.transport.write_device(c_ptrs_d, c_offs_d, c_sizes_d, n_dev, io_mode)
            total_written += d_sz

        if n_host > 0:
            services.transport.write_host(c_ptrs_h, c_offs_h, c_sizes_h, n_host, io_mode)
            total_written += h_sz

        handle.transition(CheckpointState.NVME_WRITING)
        handle.nvme_completion = "all-data-completions-observed"
        t_spdk_end = time.perf_counter()
        T_SPDK = t_spdk_end - t_spdk_start

        # The data durability barrier is deliberately before any
        # metadata write.  _persist_metadata() performs additional
        # barriers for the replica and superblock commit point.
        handle.transition(CheckpointState.FLUSHING)
        services.flush()
        if handle.state == CheckpointState.TIMED_OUT:
            raise TimeoutError("checkpoint timed out before metadata commit")

        t_meta_start = time.perf_counter()
        generation_delay_ms = float(os.environ.get(
            "NPU_NVME_TEST_GENERATION_DELAY_MS", "0"))
        if generation_delay_ms > 0:
            time.sleep(generation_delay_ms / 1000.0)
        services.record_layout(layout)
        if commit_meta:
            handle.transition(CheckpointState.METADATA_COMMITTING)
            services.publish(
                step, layout, checkpoint_meta=_checkpoint_meta)
            handle.metadata_generation = services.commit.state.metadata_generation
        else:
            handle.transition(CheckpointState.METADATA_COMMITTING)

        with open(meta_path, "wb") as f:
            pickle.dump(services.commit.state.meta_dict, f)

        t_meta_end = time.perf_counter()
        T_Meta = t_meta_end - t_meta_start

        real_time = time.perf_counter() - t_start
        bw = (total_written / 1024 / 1024 / real_time
              if real_time > 0 else 0)

        print(f"\n{'='*54}")
        print(f"[Timeline][Rank {rank_id}] Step {step} | "
              f"Background SPDK Flush ENDED at {time.perf_counter():.3f}s")
        print(f"[Breakdown][Rank {rank_id}] "
              f"Prep: {T_Prep*1000:.2f}ms | Layout: {T_Layout*1000:.2f}ms | "
              f"SPDK(H/W): {T_SPDK*1000:.2f}ms | Meta: {T_Meta*1000:.2f}ms")
        print(f"[DirectCkpt][Rank {rank_id}] Background Safe Write: "
              f"{total_written/1024/1024:.2f} MB | BW: {bw:.2f} MB/s")
        print(f"{'='*54}\n", flush=True)
        handle._complete()
    except BaseException as error:
        services.record_error(error)
        handle._fail(error)
        services.poison(handle)
        if (_live_staging is not None or
                not services.transport.quiescent()):
            lease.quarantined = True
            services.retain(lease)
        print(f"[Fatal][Rank {rank_id}] Background checkpoint "
              f"failed: {error}", flush=True)
    finally:
        if owns_io_mutex:
            services.io_mutex.release()
        if not lease.quarantined:
            try:
                if _live_staging is None:
                    services.release_snapshot(params)
                else:
                    services.release_live(handle)
            finally:
                services.scheduler.advance(io_sequence)
                services.scheduler.release(lease)

