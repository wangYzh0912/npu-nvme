from pathlib import Path

from .base import MechanismOnlyAdapter, probe_worker


class FastPersistHostAdapter(MechanismOnlyAdapter):
    name = "fastpersist_host"
    kind = "mechanism-only"
    upstream_name = "FastPersist/DeepNVMe"
    degradation_reason = ("CPU worker and AIO extensions are available, but "
                          "the upstream legacy serializer is not yet connected "
                          "to the MindSpore state bridge")

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
                       "worker": worker,
                       "worker_status": probe,
                       "upstream_integration": "not_attempted",
                       "fallback_reason": "common Host bridge used; FastFileWriter not invoked"})
        return status
