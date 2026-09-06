from pathlib import Path

from .acl_semantic import ACLSemanticAdapter


class PCCheckAclAdapter(ACLSemanticAdapter):
    name = "pccheck_acl"
    kind = "npu-semantic-port"
    upstream_name = "PCcheck"
    upstream_core_invoked = False
    configured_write_chunks = True
    mechanisms_preserved = (
        "bounded concurrent checkpoint slots",
        "background 4 MiB chunk writer and explicit slot backpressure",
        "generation publish after durable data completion",
    )
    platform_substitutions = (
        "CUDA capture -> ACL capture",
        "x86 CLWB/SFENCE persistence -> XFS fsync and atomic rename",
    )

    def _generation_dir(self, generation, slot_id):
        return Path(self.config["fs_test_dir"]) / "repro_checkpoints" / \
            self.run_dir.name / "slots" / f"slot_{int(slot_id):02d}"

    @classmethod
    def preflight(cls, config):
        root = Path(config.get("upstream_root", "")) / "pccheck"
        status = super().preflight(config)
        status.update({"source": str(root),
                       "source_exists": root.is_dir(),
                       "implementation": "semantic N-slot port of PCcheck writer",
                       "max_async": int(config.get("max_inflight", 2)),
                       "upstream_patch_required": True,
                       "status": "ready" if root.is_dir() else "dependency_blocked"})
        return status
