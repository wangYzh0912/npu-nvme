"""Canonical NPU-NVMe package; import as npu_nvme only."""
if __name__ != "npu_nvme":
    raise ImportError("Use npu_nvme, not python.npu_nvme, to preserve module identity")
