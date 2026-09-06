from pathlib import Path

from .acl_semantic import ACLSemanticAdapter


class DataStatesAclAdapter(ACLSemanticAdapter):
    name = "datastates_acl"
    kind = "npu-semantic-port"
    upstream_name = "DataStates-LLM"
    upstream_core_invoked = False
    mechanisms_preserved = (
        "device-tier to pinned-host-tier to file-tier pipeline",
        "region queue and bounded Host memory pool",
        "separate source-release and persistence completion",
    )
    platform_substitutions = (
        "CUDA runtime -> ACL runtime",
        "upstream file tier -> durable XFS writer",
    )

    @classmethod
    def preflight(cls, config):
        root = Path(config.get("upstream_root", "")) / "datastates-llm"
        status = super().preflight(config)
        status.update({"source": str(root),
                       "source_exists": root.is_dir(),
                       "implementation": "semantic tier port using upstream tier ordering",
                       "upstream_patch_required": True,
                       "status": "ready" if root.is_dir() else "dependency_blocked"})
        return status
