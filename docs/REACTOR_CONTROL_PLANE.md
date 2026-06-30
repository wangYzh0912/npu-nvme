# Single-Threaded I/O Control Plane for NPU-to-NVMe Direct Checkpointing

## Abstract

Training large language models requires periodic checkpointing to durable
storage.  On Ascend NPU clusters with direct NVMe access, the checkpoint
control plane must coordinate HBM-to-NVMe DMA transfers, background
Fire-and-Forget (FaF) persistence, and Python-side metadata I/O.  Prior
designs used multi-threaded control planes with recursive mutexes,
leading to lock contention, deadlocks during model loading, and busy-wait
CPU overhead.  We present a single-threaded control plane based on the
SPDK Reactor model, where all I/O operations are serialised through a
single event-driven thread.  Python-side requests are communicated via
lock-free MPSC rings and executed as asynchronous finite state machines.
The design eliminates `io_lock` entirely, resolves the `load()` deadlock,
and reduces reactor CPU utilisation to below 5% in steady state.  On an
Ascend 910B platform with a 3.84 TB NVMe SSD, the system achieves
4,432 MB/s write bandwidth — consistent with the original multi-threaded
implementation.

## 1.  Introduction

Deep learning training jobs require periodic checkpointing to protect
against hardware failures and enable fault-tolerant training.  On Ascend
NPU platforms, the NPU-NVMe direct checkpointing system [1] bypasses the
host CPU by performing DMA transfers directly between NPU High Bandwidth
Memory (HBM) and NVMe storage via SPDK (Storage Performance Development
Kit) in user space.

The control plane for such a system must handle three classes of I/O:

1. **Synchronous write/read requests** from the Python training loop
   (blocking calls that return after data reaches the storage medium).
2. **Asynchronous Fire-and-Forget (FaF) writes** triggered by training-step
   boundaries, where persistence runs in the background and Python
   continues training without waiting.
3. **Metadata I/O** for superblock management and JSON ledger updates,
   required during initialisation, save, and load operations.

Prior to this work, the control plane employed three threading models
simultaneously: a pthread-based listener thread for step detection, the
Python main thread for synchronous I/O, and an internal pipeline thread.
All three contended on a single recursive `io_lock` protecting the shared
SPDK queue pair (qpair) and DMA buffer pool.  This architecture suffered
from three documented problems: (a) **lock contention**: the FaF
listener held `io_lock` for the entire duration of a write (~1 second),
blocking all Python-side operations; (b) **load deadlock**: `load()`
required `io_lock` during checkpoint restoration, causing a deadlock if
the listener thread simultaneously triggered a background write; and (c)
**CPU overhead**: the busy-wait pipeline loop consumed 100% CPU per I/O
operation.

We present a refactored control plane based on the SPDK Reactor model
[2].  The key insight is that SPDK qpairs are inherently single-threaded
(submission and completion queues are not thread-safe); rather than
serialising access with locks, we consolidate all I/O execution onto a
single reactor thread and communicate requests via lock-free MPSC
(ulti-Producer, Single-Consumer) rings.  Python becomes a pure control
plane: it constructs request descriptors, enqueues them, and polls for
completion — never touching the SPDK qpair or DMA buffers directly.

## 2.  Architecture

### 2.1  Overall Structure

The refactored system consists of two threads:

**Python Main Thread** (control plane): issues write, read, and metadata
requests via three SPDK rings (`write_ring`, `read_ring`, `meta_ring`).
Each ring is a lock-free MPSC queue of depth 16 (4 for metadata).  The
Python thread polls a per-request `done` flag with `usleep(1000)`,
yielding the CPU while waiting.

**Reactor Thread** (data plane): the sole I/O executor, running an SPDK
event loop (`spdk_thread_poll`).  Four pollers are registered:

| Poller                | Period  | Function                                          |
|-----------------------|---------|---------------------------------------------------|
| `step_poller_fn`      | 10 ms   | Reads HBM step counter; triggers FaF writes        |
| `write_fsm_poller_fn` | 0 (each)| Consumes `write_ring`, executes async write FSM    |
| `read_fsm_poller_fn`  | 0 (each)| Consumes `read_ring`, executes async read FSM      |
| `meta_poller_fn`      | 0 (each)| Consumes `meta_ring`, uses dedicated `meta_qpair`  |

A fifth, non-I/O lock (`state_lock`) protects shared listener state
(registered task pointers, step counter address, probe flag) with
microsecond-length critical sections.

### 2.2  Asynchronous Write FSM (V3)

The write FSM decomposes a blocking write pipeline into bounded-work
steps.  Each invocation of `write_fsm_tick` performs:

1. **Completion reaping**: call `spdk_nvme_qpair_process_completions`,
   which triggers SPDK write-completion callbacks.  Callbacks update
   per-chunk state to `CHUNK_DONE` and return DMA buffers to the free
   pool via an atomic SPSC ring.

2. **SPDK submission**: iterate over chunks whose DMA copy has completed
   (`CHUNK_NPU_DONE`) and submit them to the NVMe controller via
   `spdk_nvme_ns_cmd_write`.

3. **DMA initiation**: pop one free buffer from the DMA pool, perform a
   single `aclrtMemcpy` (HBM-to-host, ~0.9 ms for 4 MiB), and mark the
   chunk `CHUNK_NPU_DONE`.  Only **one** DMA is issued per tick,
   bounding reactor latency.

4. **Completion check**: when `completed_count` reaches `num_tasks`,
   set `req->done = 1` and transition the FSM to `IDLE`.

For FaF writes, the `step_poller_fn` resets task state under `state_lock`
and initiates the FSM directly (both run on the reactor thread, so no
ring is needed).  Backpressure is enforced by checking `fsm->state == IDLE`
before initiating — at most one write is in flight.

### 2.3  Asynchronous Read FSM (V4)

The read FSM mirrors the write FSM for NVMe-to-HBM transfers.  Each tick:

1. Reap SPDK read completions.
2. For completed chunks (`CHUNK_SPDK_DONE`): perform `aclrtMemcpy`
   (host-to-HBM) to transfer data from the DMA buffer to NPU memory.
3. Submit one new read command to SPDK.
4. Check completion and signal the Python caller.

### 2.4  Metadata I/O via Dedicated qpair

Metadata I/O (superblock and JSON ledger) uses a dedicated second qpair
(`meta_qpair`) with a queue depth of 64.  Because no other I/O path
uses this qpair, metadata requests from the Python thread do not contend
with data-plane operations — no lock is needed.  The `meta_poller_fn`
processes `meta_ring` requests synchronously within the reactor (busy-wait
for completion is acceptable for small metadata I/O ≤ 1 MiB).

### 2.5  SPSC Ring with ARM64 Memory Barriers

The DMA buffer free pool is managed by an SPSC (Single-Producer,
Single-Consumer) ring.  In the refactored design, the reactor thread is
the sole producer (SPDK callbacks push buffers back) and the Python
thread is the sole consumer (`ring_pop` in the read path, now also
executed via the reactor).  To ensure correctness on ARM64's weakly
ordered memory model, `ring_push` uses `__atomic_store_n(..., __ATOMIC_RELEASE)`
and `ring_pop` uses `__atomic_load_n(..., __ATOMIC_ACQUIRE)`.

## 3.  Implementation

The refactoring was decomposed into five independently testable versions:

| Version | Scope                                  | Key Changes                    |
|---------|----------------------------------------|--------------------------------|
| V0      | SPDK thread/poller feasibility         | Standalone executable verifies `spdk_env_init` → `spdk_thread_lib_init` → `spdk_thread_create` → `spdk_poller_register` → `spdk_thread_poll` |
| V1      | Minimal reactor init/cleanup           | Reactor pthread integrated into `npu_nvme_init`/`cleanup`; pipelines untouched |
| V2      | Poller replaces listener thread        | `step_poller_fn` replaces `probe_listener_thread`; init order fix (`spdk_env_init` before `spdk_thread_lib_init`); `spdk_thread_create(NULL)` cpumask fix for ARM64 |
| V3      | Async write FSM + spdk\_ring           | `write_fsm_poller_fn` replaces `run_write_pipeline`; Python writes via `write_ring`; ACL context fix for HBM DMA |
| V4      | Async read FSM + remove io\_lock       | `read_fsm_poller_fn` replaces `run_read_pipeline`; `meta_qpair` for metadata; `state_lock` replaces `io_lock` |
| V5      | Cleanup                                | Remove 515 lines of dead code; consolidate debug output |

### 3.1  Critical Fixes

During implementation, four unexpected issues were discovered and resolved:

1. **Init ordering (V2)**: `spdk_thread_lib_init` was called before
   `spdk_env_init`, causing `rte_lcore_count() = 0` and all DPDK
   allocations to fail.  The correct order is `spdk_env_init` →
   diagnostics → `spdk_thread_lib_init` → `spdk_thread_create`.

2. **cpumask SEGV on ARM64 (V2)**: Passing `NPUNVMEContext*` as the
   `spdk_cpuset*` parameter to `spdk_thread_create` caused out-of-bounds
   memory access because SPDK internally dereferences the pointer as a
   potentially large CPU bitmask.  Fixed by passing `NULL` and using a
   static `g_reactor_ctx` variable.

3. **ARM64 memory ordering (V3)**: The SPSC free-ring used plain loads
   and stores.  On ARM64, the reactor thread's `ring_push` (from SPDK
   callbacks) was not visible to the Python thread's `ring_pop`, causing
   the read pipeline to spin forever on an apparently empty ring.  Fixed
   by adding `__atomic` acquire/release barriers.

4. **ACL context in FSM (V3)**: The reactor thread's FSM called
   `aclrtMemcpy` without first binding the ACL context via
   `aclrtSetDevice`/`aclrtSetCurrentContext`.  The call silently failed
   (returning error), and the FSM retried indefinitely.  Fixed by calling
   `ensure_acl_context` at the start of `write_fsm_tick`.

### 3.2  C-Layer Batch Profiling

To isolate pure data-path latency from Python overhead, we added
batch-level timestamps directly in the C-layer FSM.  In `initiate_write_fsm`
and `initiate_read_fsm`, `ts_batch_start` is recorded at the moment the first
DMA submission begins.  In the FSM completion path (when `completed_count`
reaches `num_tasks`), `ts_batch_end` is recorded and the difference stored
in `ctx->last_write_io_us` or `ctx->last_read_io_us`.  A public API
(`npu_nvme_get_last_io_us`) exposes this value to Python.

On a 1 GB host write (256 chunks x 4 MiB), C-layer latency was 259.2 ms
(4,143 MB/s), while the Python-level measurement (including numpy allocation
and ctypes marshalling) was 260.5 ms (4,122 MB/s).  **Python overhead:
0.5%**, confirming that the command-offload architecture introduces negligible
software overhead—the data path is dominated by DMA and NVMe transfer time.

## 4.  Evaluation

### 4.1  Experimental Setup

| Component       | Specification                              |
|-----------------|--------------------------------------------|
| NPU             | Ascend 910B, 64 GB HBM                     |
| NVMe SSD        | 3.84 TB Samsung PM9A3 (PCIe Gen4 ×4)       |
| CPU             | ARM64 (Kunpeng 920), 96 cores              |
| OS              | openEuler 22.03 LTS (Linux 5.10)           |
| SPDK            | v26.01-pre (DPDK 25.07)                    |
| Test model      | GPT-2 XL, 3.28 GB FP16 parameters           |

### 4.2  Bandwidth

Sequential write bandwidth measured with 4 MiB chunks and pipeline
depth 8:

| Metric              | V2 (busy-wait) | V6 (async FSM)  |
|---------------------|----------------|------------------|
| Best single-trial   | 4,121 MB/s     | 4,432 MB/s       |
| 3-trial average     | 3,707 MB/s     | 4,300+ MB/s      |
| C-layer (1 GB host) | N/A            | 4,143 MB/s       |
| Python overhead     | N/A            | 0.5%             |
| Pipe-depth scan     | All pass       | All pass         |
| Chunk-size scan     | Best 4 MB      | Best 4 MB        |

The async FSM introduces no bandwidth degradation; minor improvements
are attributed to reduced lock contention.  The C-layer profiling
confirms that Python marshalling overhead is negligible (0.5%).

### 4.3  Reactor CPU Utilisation

In steady state (no active I/O), the reactor thread's CPU usage is
dominated by the 100 μs sleep between `spdk_thread_poll` iterations.
With four registered pollers (all returning immediately when idle),
measured CPU is < 1% — well within the 5% target.

### 4.4  Correctness

- **Data integrity**: 1 MB host write/readback produces bit-exact match
- **Multiple init/cleanup cycles**: 15+ cycles in raw\_bw test, all pass
- **Smoke test**: init → write → read → verify → cleanup, no errors
- **Lock removal verification**: `grep -n 'io_lock' src/npu_nvme.c`
  returns zero matches; `state_lock` references are limited to 32,
  all in listener-state protection

## 5.  Related Work

The SPDK Reactor model is widely used in storage targets (NVMe-oF, vhost)
but has not been previously applied to NPU-to-NVMe checkpointing control
planes.  CheckFreq [3] uses a two-phase pipeline with at-most-1
in-flight write, while PCcheck [4] extends this to N-slot concurrent
pipelines.  Both couple the control plane tightly to the data path.
Our work demonstrates that command offloading via lock-free rings
preserves throughput while eliminating thread-safety hazards.

## 6.  Conclusion

We have presented a single-threaded, event-driven control plane for
NPU-to-NVMe direct checkpointing.  By consolidating all I/O execution
onto a single SPDK reactor thread and communicating requests via
lock-free MPSC rings, we eliminate the recursive `io_lock`, resolve the
`load()` deadlock, and reduce CPU overhead to negligible levels.  The
five-version refactoring methodology ensures each change is independently
testable and committable.  The implementation achieves 4,432 MB/s write
bandwidth on Ascend 910B, matching the original multi-threaded
performance without its synchronisation hazards.

## References

[1] NPU-NVMe Direct Checkpointing System.  `npu-nvme` repository.
[2] SPDK: Storage Performance Development Kit.  https://spdk.io
[3] CheckFreq: Frequent, Fine-Grained DNN Checkpointing.  FAST 2021.
[4] PCcheck: Persistent Checkpointing for Large-Scale DNN Training.
    ASPLOS 2025.
