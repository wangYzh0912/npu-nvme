from pathlib import Path

from .base import probe_worker
from .worker_semantic import WorkerSemanticAdapter


class FastPersistHostAdapter(WorkerSemanticAdapter):
    name = "fastpersist_host"
    kind = "host-adapted-semantic-port"
    upstream_name = "FastPersist/DeepNVMe"
    worker_config_key = "worker_python_fastpersist"
    worker_module = "experiments.baselines.repro.workers.fastpersist_worker"
    upstream_python_roots = ("DeepSpeed", "DeepSpeedExamples")
    upstream_core_invoked = True
    mechanisms_preserved = (
        "patched PyTorch legacy serialization storage-list hook",
        "DeepNVMe FastFileWriter and AIO",
        "64 MiB double-buffered drain before durable completion",
    )
    platform_substitutions = (
        "PyTorch model extraction -> common ACL FULL capture and shared memory",
        "CUDA pinned memory -> DeepNVMe CPU locked memory",
        "GDS disabled; Host AIO retained",
        "CUDA-only UtilsBuilder -> equivalent CPU byte view",
    )

    @classmethod
    def preflight(cls, config):
        worker = config.get("worker_python_fastpersist")
        root = Path(config.get("upstream_root", "")) / "DeepSpeedExamples"
        ds = Path(config.get("upstream_root", "")) / "DeepSpeed"
        status = super().preflight(config)
        probe = probe_worker(
            worker,
            "import torch, deepspeed; "
            "from deepspeed.ops.op_builder import AsyncIOBuilder, PinMemoryBuilder; "
            "print('builders=available'); "
            "print('torch='+torch.__version__+',deepspeed='+deepspeed.__version__)" ,
            timeout=30)
        status.update({"source": str(root), "deepspeed": str(ds),
                       "source_exists": root.is_dir() and ds.is_dir(),
                       "worker": worker,
                       "worker_status": probe,
                       "upstream_integration": "patched serializer+FastFileWriter+AIO",
                       "status": "ready" if (
                           root.is_dir() and ds.is_dir()
                           and probe["status"] == "ready")
                       else "dependency_blocked"})
        return status
