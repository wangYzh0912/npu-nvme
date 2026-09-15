"""Small adapter contract used by the common runner."""

from __future__ import annotations

import importlib.util
import subprocess
import os
from pathlib import Path

from ..host_bridge import SharedSnapshot
from ..protocol import AdapterError, DependencyBlocked, EventLog, Handle
from ..state_bridge import load_raw_snapshot, save_raw_snapshot


def adapter_status(name, config):
    return {"adapter": name, "status": "not_attempted", "reason": None}


class Adapter:
    name = "base"
    kind = "inspired"

    def __init__(self, config, run_dir):
        self.config = config
        self.run_dir = Path(run_dir)
        self.events = EventLog(self.run_dir / "events.jsonl")
        self.next_generation = 1
        self.handles = []

    @classmethod
    def preflight(cls, config):
        return {"adapter": cls.name, "kind": cls.kind,
                "status": "ready", "reason": None}

    def prepare(self, state_schema, config):
        return {"adapter": self.name, "kind": self.kind,
                "status": "ready", "state_fields": len(state_schema["fields"])}

    def submit(self, generation, state_source, controls):
        raise NotImplementedError

    def wait_source_release(self, handle, timeout):
        return handle.wait_source_release(timeout)

    def wait_persisted(self, handle, timeout):
        return handle.wait_persisted(timeout)

    def restore(self, generation, destination):
        raise NotImplementedError

    def drain(self, timeout):
        for handle in self.handles:
            self.wait_persisted(handle, timeout)

    def close(self):
        return None


class NoneAdapter(Adapter):
    """No-checkpoint training control used as the wall-clock baseline."""

    name = "none"
    kind = "training-reference"

    def submit(self, generation, state_source, controls):
        raise AdapterError("none adapter does not create checkpoint generations")

    def restore(self, generation, destination):
        raise AdapterError("none adapter has no persisted state")



def require_path(path, label):
    if not path:
        raise DependencyBlocked(f"{label} is not configured")
    if not Path(path).exists():
        raise DependencyBlocked(f"{label} does not exist: {path}")


def require_python_module(python, module):
    if not python:
        raise DependencyBlocked(f"worker Python is not configured for {module}")
    probe = Path(python)
    if not probe.exists():
        raise DependencyBlocked(f"worker Python does not exist: {python}")


def probe_worker(python, code, timeout=30):
    """Run a side-effect-free import/build probe in a locked worker venv."""
    if not python:
        return {"status": "missing", "reason": "worker Python is not configured"}
    try:
        env = os.environ.copy()
        user_site = "/home/user7/.local/lib/python3.9/site-packages"
        if Path(user_site).exists():
            env["PYTHONPATH"] = user_site + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run([str(python), "-c", code], capture_output=True,
                              text=True, check=False, timeout=timeout, env=env)
    except Exception as error:
        return {"status": "error", "reason": repr(error)}
    result = {"status": "ready" if proc.returncode == 0 else "failed",
              "returncode": proc.returncode}
    if proc.stdout.strip():
        result["stdout"] = proc.stdout.strip()[-4000:]
    if proc.stderr.strip():
        result["stderr"] = proc.stderr.strip()[-4000:]
    return result
