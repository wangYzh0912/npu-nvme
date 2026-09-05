from pathlib import Path

from .base import Adapter
from ..protocol import DependencyBlocked


class FastPersistHostAdapter(Adapter):
    name = "fastpersist_host"
    kind = "host-adapted"

    @classmethod
    def preflight(cls, config):
        worker = config.get("worker_python_fastpersist")
        root = Path(config.get("upstream_root", "")) / "DeepSpeedExamples"
        ds = Path(config.get("upstream_root", "")) / "DeepSpeed"
        missing = [str(path) for path in (root, ds) if not path.exists()]
        if missing:
            return {"adapter": cls.name, "kind": cls.kind,
                    "status": "dependency_blocked", "reason": "missing locked source: " + ", ".join(missing)}
        if not worker or not Path(worker).exists():
            return {"adapter": cls.name, "kind": cls.kind,
                    "status": "dependency_blocked",
                    "reason": f"missing CPU worker Python: {worker}"}
        return {"adapter": cls.name, "kind": cls.kind,
                "status": "build_pending", "source": str(root), "worker": worker}

    def submit(self, generation, state_source, controls):
        raise DependencyBlocked(
            "FastPersist worker/AIO extensions are not prepared; no torch.save fallback is allowed")

    def restore(self, generation, destination):
        raise DependencyBlocked("FastPersist worker is not prepared")

