from pathlib import Path

from .base import MechanismOnlyAdapter


class DataStatesAclAdapter(MechanismOnlyAdapter):
    name = "datastates_acl"
    kind = "mechanism-only"
    upstream_name = "DataStates-LLM"
    degradation_reason = ("locked checkout hard-requires CUDA/nvcc and liburing; "
                          "ACL C ABI was not available")

    @classmethod
    def preflight(cls, config):
        root = Path(config.get("upstream_root", "")) / "datastates-llm"
        status = super().preflight(config)
        status.update({"source": str(root),
                       "attempt": "CMake configure failed: nvcc not found"})
        return status
