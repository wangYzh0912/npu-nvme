from pathlib import Path

from .base import probe_worker
from .worker_semantic import WorkerSemanticAdapter


class ByteCheckpointHostAdapter(WorkerSemanticAdapter):
    name = "bytecheckpoint_host"
    kind = "host-adapted-semantic-port"
    upstream_name = "ByteCheckpoint"
    worker_config_key = "worker_python_bytecheckpoint"
    worker_module = "experiments.baselines.repro.workers.bytecheckpoint_worker"
    upstream_python_roots = ("ByteCheckpoint",)
    upstream_core_invoked = True
    mechanisms_preserved = (
        "model and optimizer DDPSavePlanner/DDPLoadPlanner workflows",
        "official extra_state workflow for persisted controls",
        "shared three-component CKPTCounter and async LocalStorageWriter queues",
    )
    platform_substitutions = (
        "PyTorch model extraction -> common ACL FULL capture and shared memory",
        "CUDA pinned D2H -> ACL pinned D2H",
    )

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
                       "source_exists": root.is_dir(),
                       "worker_status": probe,
                       "upstream_integration": (
                           "model+optimizer DDP planners, extra_state workflow, "
                           "shared CKPTCounter(3)"),
                       "status": "ready" if (
                           root.is_dir() and probe["status"] == "ready")
                       else "dependency_blocked"})
        return status
