from pathlib import Path

from .base import MechanismOnlyAdapter


class PCCheckAclAdapter(MechanismOnlyAdapter):
    name = "pccheck_acl"
    kind = "mechanism-only"
    upstream_name = "PCcheck"
    degradation_reason = ("locked checkout is CUDA/x86, uses clwb/sfence and "
                          "the expected main.cpp is absent")

    @classmethod
    def preflight(cls, config):
        root = Path(config.get("upstream_root", "")) / "pccheck"
        status = super().preflight(config)
        status.update({"source": str(root),
                       "note": "Host state writer exercises protocol only"})
        return status
