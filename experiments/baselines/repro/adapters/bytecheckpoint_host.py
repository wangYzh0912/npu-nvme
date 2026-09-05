from pathlib import Path

from .base import Adapter
from ..protocol import DependencyBlocked


class ByteCheckpointHostAdapter(Adapter):
    name = "bytecheckpoint_host"
    kind = "host-adapted"

    @classmethod
    def preflight(cls, config):
        worker = config.get("worker_python_bytecheckpoint")
        root = Path(config.get("upstream_root", "")) / "ByteCheckpoint"
        if not root.exists():
            return {"adapter": cls.name, "kind": cls.kind,
                    "status": "dependency_blocked",
                    "reason": f"missing locked source: {root}"}
        if not worker or not Path(worker).exists():
            return {"adapter": cls.name, "kind": cls.kind,
                    "status": "dependency_blocked",
                    "reason": f"missing CPU worker Python: {worker}"}
        return {"adapter": cls.name, "kind": cls.kind,
                "status": "build_pending", "source": str(root), "worker": worker}

    def submit(self, generation, state_source, controls):
        raise DependencyBlocked(
            "ByteCheckpoint worker is not prepared; no ordinary pickle/file fallback is allowed")

    def restore(self, generation, destination):
        raise DependencyBlocked("ByteCheckpoint worker is not prepared")

