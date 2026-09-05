from pathlib import Path

from .base import Adapter, require_path
from ..protocol import DependencyBlocked


class DataStatesAclAdapter(Adapter):
    name = "datastates_acl"
    kind = "native"

    @classmethod
    def preflight(cls, config):
        root = Path(config.get("upstream_root", "")) / "datastates-llm"
        if not root.exists():
            return {"adapter": cls.name, "kind": cls.kind,
                    "status": "dependency_blocked",
                    "reason": f"missing locked source: {root}"}
        return {"adapter": cls.name, "kind": cls.kind,
                "status": "build_failed", "source": str(root),
                "reason": "locked checkout hard-requires nvcc/CUDA; ACL C ABI port not built",
                "attempt": "cmake configure with BUILD_PYTHON_BINDINGS=OFF failed: nvcc not found"}

    def submit(self, generation, state_source, controls):
        raise DependencyBlocked(
            "DataStates ACL adapter requires the ACL C ABI build; no CUDA fallback is permitted")

    def restore(self, generation, destination):
        raise DependencyBlocked("DataStates ACL adapter is not built")
