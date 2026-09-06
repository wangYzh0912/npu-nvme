from pathlib import Path

from .base import MechanismOnlyAdapter, probe_worker


class ByteCheckpointHostAdapter(MechanismOnlyAdapter):
    name = "bytecheckpoint_host"
    kind = "mechanism-only"
    upstream_name = "ByteCheckpoint"
    degradation_reason = ("CPU worker is available, but the upstream planner "
                          "is not yet connected to the MindSpore state bridge")

    @classmethod
    def preflight(cls, config):
        worker = config.get("worker_python_bytecheckpoint")
        root = Path(config.get("upstream_root", "")) / "ByteCheckpoint"
        status = super().preflight(config)
        probe = probe_worker(
            worker,
            "import torch, bytecheckpoint; "
            "import bytecheckpoint.workflow.state_dict; "
            "print('torch='+torch.__version__)" )
        status.update({"source": str(root), "worker": worker,
                       "worker_status": probe,
                       "upstream_integration": "not_attempted",
                       "fallback_reason": "common Host bridge used; BC planner/engine not invoked"})
        return status
