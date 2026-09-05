from pathlib import Path

from .base import Adapter
from ..protocol import DependencyBlocked


class PCCheckAclAdapter(Adapter):
    name = "pccheck_acl"
    kind = "native"

    @classmethod
    def preflight(cls, config):
        root = Path(config.get("upstream_root", "")) / "pccheck"
        if not root.exists():
            return {"adapter": cls.name, "kind": cls.kind,
                    "status": "dependency_blocked",
                    "reason": f"missing locked source: {root}"}
        return {"adapter": cls.name, "kind": cls.kind,
                "status": "build_failed", "source": str(root),
                "reason": "locked checkout is CUDA/x86 and its Makefile references missing main.cpp",
                "note": "ACL/ARM writer patch required; no CUDA/x86 fallback is used"}

    def submit(self, generation, state_source, controls):
        raise DependencyBlocked(
            "PCcheck ACL/ARM writer is not built; CUDA/x86 writer cannot be used")

    def restore(self, generation, destination):
        raise DependencyBlocked("PCcheck ACL/ARM adapter is not built")
