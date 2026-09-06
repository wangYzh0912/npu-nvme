"""Adapter lifecycle, timing events, and durable completion handles."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from python.full_checkpoint_protocol import CheckpointState, require_transition


TERMINAL = frozenset({"persisted", "failed"})


@dataclass
class Event:
    event: str
    monotonic_ns: int = field(default_factory=time.monotonic_ns)
    generation: Optional[int] = None
    request_id: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event, generation=None, request_id=None, **detail):
        row = asdict(Event(event, generation=generation,
                           request_id=request_id, detail=detail))
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        return row


class Handle:
    """Tracks source, input-buffer, data, and durable completion separately."""

    def __init__(self, adapter, generation, request_id, events):
        self.adapter = adapter
        self.generation = int(generation)
        self.request_id = str(request_id)
        self.events = events
        self.admitted = False
        self.source_released = False
        self.input_buffer_released = False
        self.data_completed = False
        self.persisted = False
        self.failed = False
        self.error = None
        self.sha256 = None
        self.state = CheckpointState.CREATED
        self.transitions = []
        self.timestamps_ns = {CheckpointState.CREATED.value: time.monotonic_ns()}
        self._condition = threading.Condition()

    def mark(self, state, **detail):
        if state not in {
                "admitted", "source_released", "input_buffer_released",
                "data_completed", "persisted", "failed"}:
            raise ValueError(f"invalid handle state: {state}")
        with self._condition:
            if self.failed and state != "failed":
                return
            if state == "failed":
                self.failed = True
                self.error = detail.get("error")
            else:
                setattr(self, state, True)
                if "sha256" in detail:
                    self.sha256 = detail["sha256"]
            row = self.events.emit(state, self.generation, self.request_id, **detail)
            self.timestamps_ns[state] = row["monotonic_ns"]
            self._condition.notify_all()

    def transition(self, state, **detail):
        target = CheckpointState(state)
        with self._condition:
            require_transition(self.state, target)
            self.state = target
            row = self.events.emit(target.value, self.generation,
                                   self.request_id, **detail)
            self.transitions.append(row)
            self.timestamps_ns[target.value] = row["monotonic_ns"]
            if target == CheckpointState.PERSISTED:
                self.persisted = True
                self.sha256 = detail.get("sha256", self.sha256)
            elif target in (CheckpointState.FAILED, CheckpointState.CANCELLED,
                            CheckpointState.TIMED_OUT):
                self.failed = True
                self.error = detail.get("error")
            self._condition.notify_all()
        return self

    def _wait(self, field, timeout):
        deadline = time.monotonic() + float(timeout)
        with self._condition:
            while not getattr(self, field) and not self.failed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"{self.adapter} generation {self.generation} "
                        f"timed out waiting for {field}")
                self._condition.wait(remaining)
            if self.failed:
                raise RuntimeError(self.error or "checkpoint request failed")
        return self

    def wait_source_release(self, timeout):
        return self._wait("source_released", timeout)

    def wait_input_release(self, timeout):
        return self._wait("input_buffer_released", timeout)

    def wait_persisted(self, timeout):
        return self._wait("persisted", timeout)

    def as_dict(self):
        return {
            "adapter": self.adapter,
            "generation": self.generation,
            "request_id": self.request_id,
            "admitted": self.admitted,
            "source_released": self.source_released,
            "input_buffer_released": self.input_buffer_released,
            "data_completed": self.data_completed,
            "persisted": self.persisted,
            "failed": self.failed,
            "error": self.error,
            "sha256": self.sha256,
            "state": self.state.value,
            "transitions": list(self.transitions),
            "timestamps_ns": dict(self.timestamps_ns),
        }


class AdapterError(RuntimeError):
    pass


class DependencyBlocked(AdapterError):
    pass


class BuildFailed(AdapterError):
    pass
