"""ACL capture plus isolated upstream CPU-worker persistence."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from python.full_checkpoint_protocol import CheckpointState

from ..host_bridge import SharedSnapshot, snapshot_from_descriptor
from ..state_bridge import fsync_directory, write_json
from .acl_semantic import ACLSemanticAdapter


class WorkerSemanticAdapter(ACLSemanticAdapter):
    worker_config_key = None
    worker_module = None
    upstream_python_roots = ()

    def __init__(self, config, run_dir):
        super().__init__(config, run_dir)
        worker_python = config[self.worker_config_key]
        env = os.environ.copy()
        roots = [str(Path(config["project_root"]).resolve())]
        roots.extend(str(Path(config["upstream_root"]) / item)
                     for item in self.upstream_python_roots)
        # The locked worker venvs intentionally remain isolated, but a small
        # set of user-installed pure-Python dependencies (e.g. pydantic's
        # annotated-types) is needed when the coordinator is run as root.
        user_site = "/home/user7/.local/lib/python3.9/site-packages"
        if Path(user_site).exists():
            roots.append(user_site)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = os.pathsep.join(roots + ([existing] if existing else []))
        self._worker = subprocess.Popen(
            [worker_python, "-m", self.worker_module], cwd=config["project_root"],
            env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        self._worker_lock = threading.Lock()

    def _worker_request(self, request):
        with self._worker_lock:
            if self._worker.poll() is not None:
                stderr = self._worker.stderr.read()
                raise RuntimeError(
                    f"{self.name} worker exited {self._worker.returncode}: {stderr[-4000:]}")
            self._worker.stdin.write(json.dumps(request, sort_keys=True) + "\n")
            self._worker.stdin.flush()
            response = None
            noise = []
            while response is None:
                line = self._worker.stdout.readline()
                if not line:
                    stderr = self._worker.stderr.read()
                    raise RuntimeError(
                        f"{self.name} worker returned no response: {stderr[-4000:]}"
                        f" stdout={''.join(noise)[-2000:]}")
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    # ByteCheckpoint/DeepSpeed emit import diagnostics to
                    # stdout in their locked environments.  They are evidence,
                    # not protocol frames; retain them in the run timeline.
                    noise.append(line)
                    generation = request.get("generation")
                    request_id = (f"{self.name}-{int(generation):06d}"
                                  if generation is not None else self.name)
                    self.events.emit("worker_log", generation, request_id,
                                     line=line.rstrip())
            if response.get("status") != "ok":
                raise RuntimeError(
                    f"{self.name} worker failure: {response.get('error')}\n"
                    f"{response.get('traceback', '')}")
            return response

    def _generation_dir(self, generation, slot_id):
        return Path(self.config["fs_test_dir"]) / "repro_checkpoints" / \
            self.run_dir.name / f"generation_{int(generation):06d}"

    def _persist(self, handle, snapshot, slot):
        owner = None
        try:
            handle.transition(CheckpointState.NVME_WRITING,
                              slot_id=slot.slot_id)
            ipc_begin = time.monotonic_ns()
            owner, descriptor = SharedSnapshot.from_snapshot(
                snapshot, prefix=f"{self.name}_save")
            handle.mark("input_buffer_released", ipc_copy_ns=time.monotonic_ns() - ipc_begin)
            self._free_slots.put(slot)
            slot = None
            response = self._worker_request({
                "operation": "save", "generation": handle.generation,
                "checkpoint_dir": str(self._generation_dir(
                    handle.generation, descriptor.get("slot_id", 0))),
                "descriptor": descriptor,
                "timeout_seconds": self.config["timeout_seconds"],
            })
            handle.mark("data_completed", sha256=response["sha256"],
                        worker=response)
            handle.transition(CheckpointState.FLUSHING,
                              worker_data_done_ns=response.get("data_done_ns"))
            handle.transition(CheckpointState.METADATA_COMMITTING)
            checkpoint_dir = self._generation_dir(handle.generation, 0)
            bridge = {
                "generation": handle.generation, "sha256": response["sha256"],
                "schema": snapshot.schema,
                "controls_metadata": snapshot.controls_metadata,
                "upstream_hook": response.get("upstream_hook"),
                "worker_evidence": response,
            }
            write_json(checkpoint_dir / "bridge.json.tmp", bridge)
            with (checkpoint_dir / "bridge.json.tmp").open("rb", buffering=0) as stream:
                os.fsync(stream.fileno())
            os.replace(checkpoint_dir / "bridge.json.tmp",
                       checkpoint_dir / "bridge.json")
            fsync_directory(checkpoint_dir)
            handle.transition(CheckpointState.PERSISTED,
                              sha256=response["sha256"],
                              upstream_hook=response.get("upstream_hook"))
        except BaseException as error:
            if handle.state not in (CheckpointState.FAILED,
                                    CheckpointState.CANCELLED,
                                    CheckpointState.TIMED_OUT):
                handle.transition(CheckpointState.FAILED, error=repr(error))
        finally:
            if owner is not None:
                owner.close(unlink=True)
            if slot is not None:
                self._free_slots.put(slot)

    def restore(self, generation, destination):
        checkpoint_dir = self._generation_dir(generation, 0)
        bridge = json.loads((checkpoint_dir / "bridge.json").read_text())
        if int(bridge["generation"]) != int(generation):
            raise ValueError("worker checkpoint generation mismatch")
        owner, descriptor = SharedSnapshot.empty_from_metadata(
            bridge["schema"], bridge["controls_metadata"],
            prefix=f"{self.name}_restore")
        try:
            response = self._worker_request({
                "operation": "restore", "generation": int(generation),
                "checkpoint_dir": str(checkpoint_dir),
                "descriptor": descriptor,
                "timeout_seconds": self.config["timeout_seconds"],
            })
            snapshot = snapshot_from_descriptor(descriptor)
            if response["sha256"] != bridge["sha256"] or \
                    snapshot.digest() != bridge["sha256"]:
                raise ValueError("worker restore checksum mismatch")
            return snapshot
        finally:
            owner.close(unlink=True)

    def close(self):
        if self._closed:
            return
        super().close()
        try:
            with self._worker_lock:
                self._worker.stdin.write(json.dumps({"operation": "close"}) + "\n")
                self._worker.stdin.flush()
                # close is the only non-OK worker response (the worker has
                # intentionally left its request loop).  Ignore import noise
                # exactly as in _worker_request and require the closed ACK.
                while True:
                    line = self._worker.stdout.readline()
                    if not line:
                        raise RuntimeError("worker exited without close ACK")
                    try:
                        response = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if response.get("status") != "closed":
                        raise RuntimeError(f"worker close failed: {response}")
                    break
        finally:
            self._worker.stdin.close()
            self._worker.wait(timeout=30)
