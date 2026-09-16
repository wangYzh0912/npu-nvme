"""Completion observation with explicitly supplied framework fence services."""
import math
import threading
import time
from dataclasses import dataclass, field
from npu_nvme.types import CheckpointState, TERMINAL_STATES, require_transition

@dataclass
class HandleServices:
    install_fence: object = None
    collect_fence: object = None
    chunk_count: int = 0
    stats: dict = field(default_factory=dict)

class CheckpointHandle:
    """Observable completion state for one frozen checkpoint generation."""

    DISPATCHED = "DISPATCHED"
    PERSISTED = "PERSISTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"

    def __init__(self, owner, request_id, generation, step, rank_id=0,
                 snapshot_slot=None, timeout=None, snapshot_generation=None):
        self.services = owner
        self.request_id = request_id
        self.generation = generation
        self.snapshot_generation = (generation if snapshot_generation is None
                                    else snapshot_generation)
        self.metadata_generation = None
        self.step = step
        self.rank_id = int(rank_id)
        self.snapshot_slot = snapshot_slot
        self.dma_slot = None
        self.acl_event = None
        self.nvme_completion = None
        self.checksum = None
        self.timeout = timeout
        self.status = self.DISPATCHED
        self.state = CheckpointState.CREATED
        self.error = None
        self.api_enter_ns = None
        self.api_return_ns = None
        self.admission_wait_ns = 0
        self.freeze_wait_ns = 0
        self.update_wait_ns = 0
        self.update_deadline_missed = False
        self.dma_submit_ns = None
        self.dma_complete_ns = None
        self.dma_chunks = []
        self.update_fence_install_ns = None
        self.update_fence_release_ns = None
        self.snapshot_state_digest = None
        self.training_dependency = None
        self._live_event = None
        self._live_pre_event = None
        self._live_post_event = None
        self._live_buffers = []
        self._live_fence_consumed = False
        self.events = []
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._record_event(self.state)

    def _record_event(self, state):
        self.events.append({"state": CheckpointState(state).value,
                            "monotonic_ns": time.monotonic_ns()})

    def transition(self, state):
        state = CheckpointState(state)
        with self._lock:
            require_transition(self.state, state)
            self.state = state
            self._record_event(state)

    def _complete(self):
        if self.state == CheckpointState.TIMED_OUT:
            # The underlying request is not cancellable. Preserve the
            # caller-visible timeout even if the reactor drains later.
            self._done.set()
            return
        if self.state in (CheckpointState.FAILED, CheckpointState.CANCELLED):
            self._done.set()
            return
        if self.state != CheckpointState.PERSISTED:
            self.transition(CheckpointState.PERSISTED)
        self.status = self.PERSISTED
        self._done.set()

    def _fail(self, error):
        if self.state == CheckpointState.TIMED_OUT:
            self.error = error
            self.status = self.TIMED_OUT
            self._done.set()
            return
        if self.state == CheckpointState.PERSISTED:
            return
        with self._lock:
            if self.state not in TERMINAL_STATES:
                require_transition(self.state, CheckpointState.FAILED)
                self.state = CheckpointState.FAILED
                self._record_event(self.state)
        self.status = self.FAILED
        self.error = error
        self._done.set()

    def as_dict(self):
        return {
            "request_id": self.request_id,
            "checkpoint_step": self.step,
            "generation": self.generation,
            "snapshot_generation": self.snapshot_generation,
            "metadata_generation": self.metadata_generation,
            "rank_id": self.rank_id,
            "snapshot_slot": self.snapshot_slot,
            "dma_slot": self.dma_slot,
            "acl_event": self.acl_event,
            "nvme_completion": self.nvme_completion,
            "checksum": self.checksum,
            "timeout": self.timeout,
            "state": self.state.value,
            "status": self.status,
            "error": repr(self.error) if self.error is not None else None,
            "api_enter_ns": self.api_enter_ns,
            "api_return_ns": self.api_return_ns,
            "admission_wait_ns": self.admission_wait_ns,
            "freeze_wait_ns": self.freeze_wait_ns,
            "update_wait_ns": self.update_wait_ns,
            "update_deadline_missed": self.update_deadline_missed,
            "dma_submit_ns": self.dma_submit_ns,
            "dma_complete_ns": self.dma_complete_ns,
            "dma_chunks": list(self.dma_chunks),
            "update_fence_install_ns": self.update_fence_install_ns,
            "update_fence_release_ns": self.update_fence_release_ns,
            "training_dependency": self.training_dependency,
            "events": list(self.events),
        }

    def install_update_fence(self, stream_ptr):
        if not self._live_event:
            return False
        return self.services.install_fence(self, stream_ptr)

    def collect_update_wait(self):
        if not self._live_post_event:
            return 0
        return self.services.collect_fence(self)

    def wait(self, timeout=None):
        """Wait for durable completion and raise the original failure."""
        if timeout is None:
            timeout = self.timeout if self.timeout is not None else 120.0
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("wait timeout must be finite and nonnegative")
        if not self._done.wait(timeout=timeout):
            # Observation timeout neither cancels DMA nor poisons the request.
            raise TimeoutError("checkpoint did not reach a terminal state")
        if self.status == self.CANCELLED:
            raise RuntimeError("checkpoint cancelled before execution")
        if self.status == self.FAILED:
            raise RuntimeError("checkpoint persistence failed") from self.error
        if self.status != self.PERSISTED:
            raise TimeoutError("checkpoint did not reach PERSISTED state")
        return self

    def done(self):
        """Return whether this generation reached a terminal state."""
        return self._done.is_set()

    def __iter__(self):
        """Legacy tuple compatibility for existing benchmark callers."""
        yield 0
        yield getattr(self.services, "chunk_count", 0)
        yield 0.0
        yield 0.0
        yield getattr(self.services, "stats", {})

