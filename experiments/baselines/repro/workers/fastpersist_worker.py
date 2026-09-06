"""Persistent FastPersist patched-serializer and DeepNVMe AIO worker."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import torch
from deepspeed.io import FastFileWriter, FastFileWriterConfig
from deepspeed.ops.op_builder import AsyncIOBuilder

from .common import attach_arrays, canonical_digest, durable_tree, serve


class Worker:
    def __init__(self):
        self.aio = AsyncIOBuilder().load(verbose=False).aio_handle(
            block_size=8 * 1024 * 1024, queue_depth=8,
            single_submit=False, overlap_events=False,
            intra_op_parallelism=1)
        example = torch.empty(0, dtype=torch.uint8)
        self.pinned = self.aio.new_cpu_locked_tensor(64 * 1024 * 1024,
                                                     example)
        # DeepNVMe's UtilsBuilder is CUDA-only in the locked ARM checkout, but
        # its operation is a pure byte reinterpretation.  Keep FastFileWriter,
        # AIO, double buffering and legacy storage-list submission intact and
        # replace only this unavailable helper with an equivalent CPU op.
        from deepspeed.io import fast_file_writer as ffw
        try:
            ffw.UtilsBuilder().load()
        except ValueError as error:
            if "not been implemented on CPU" not in str(error):
                raise
            class _CPUUtils:
                @staticmethod
                def cast_to_byte_tensor(tensors):
                    return [tensor.contiguous().reshape(-1).view(torch.uint8)
                            for tensor in tensors]
            class _CPUUtilsBuilder:
                def load(self, *args, **kwargs):
                    return _CPUUtils()
            ffw.UtilsBuilder = _CPUUtilsBuilder
            self.utils_substitution = "CPU view(torch.uint8) equivalent"
        else:
            self.utils_substitution = None
        source = Path(__file__).resolve().parents[5] / "npu-nvme-baseline-upstreams"
        source = source / "DeepSpeedExamples" / "deepnvme" / \
            "model_checkpoint" / "torch" / "serialization_fast_v2.6.0.py"
        if not source.exists():
            source = Path("/home/user7/npu-nvme-baseline-upstreams/DeepSpeedExamples/") / \
                "deepnvme/model_checkpoint/torch/serialization_fast_v2.6.0.py"
        spec = importlib.util.spec_from_file_location("fastpersist_serialization",
                                                      source)
        self.serializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.serializer)
        self.serializer_source = str(source)

    @staticmethod
    def _state(arrays):
        return {name: torch.from_numpy(array) for name, array in arrays.items()}

    def execute(self, request):
        if request["operation"] == "save":
            return self.save(request)
        if request["operation"] == "restore":
            return self.restore(request)
        raise ValueError(f"unknown operation {request['operation']}")

    def save(self, request):
        checkpoint_dir = Path(request["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        target = checkpoint_dir / "checkpoint.pt"
        shm, arrays = attach_arrays(request["descriptor"])
        writer = None
        try:
            expected = canonical_digest(arrays)
            config = FastFileWriterConfig(
                dnvme_handle=self.aio, pinned_tensor=self.pinned,
                double_buffer=True, num_parallel_writers=1, writer_rank=0)
            writer = FastFileWriter(file_path=str(target), config=config)
            hook_calls = [0]
            original_hook = writer.save_torch_storage_object_list

            def traced_hook(*args, **kwargs):
                hook_calls[0] += 1
                return original_hook(*args, **kwargs)

            writer.save_torch_storage_object_list = traced_hook
            begin = time.monotonic_ns()
            self.serializer.save(
                obj=self._state(arrays), f=writer,
                _use_new_zipfile_serialization=False)
            writer.close()
            writer = None
            data_done = time.monotonic_ns()
            if hook_calls[0] <= 0:
                raise RuntimeError("FastPersist storage-list hook was not invoked")
            durable_tree(checkpoint_dir)
            durable = time.monotonic_ns()
            return {
                "sha256": expected, "worker_begin_ns": begin,
                "data_done_ns": data_done, "durable_ns": durable,
                "storage_list_hook_calls": hook_calls[0],
                "upstream_hook": "patched legacy serializer+FastFileWriter+AIO",
                "serializer_source": self.serializer_source,
                "utils_substitution": self.utils_substitution,
                "torch_version": torch.__version__,
            }
        finally:
            if writer is not None:
                writer.close()
            shm.close()

    def restore(self, request):
        checkpoint_dir = Path(request["checkpoint_dir"])
        shm, arrays = attach_arrays(request["descriptor"])
        try:
            loaded = torch.load(checkpoint_dir / "checkpoint.pt",
                                map_location="cpu", weights_only=True)
            if set(loaded) != set(arrays):
                raise ValueError("FastPersist restored state field mismatch")
            for name, destination in arrays.items():
                source = loaded[name].detach().cpu().numpy()
                if source.dtype != destination.dtype or source.shape != destination.shape:
                    raise ValueError(f"FastPersist dtype/shape mismatch for {name}")
                destination[...] = source
            return {
                "sha256": canonical_digest(arrays),
                "upstream_hook": "torch.load legacy checkpoint",
            }
        finally:
            shm.close()

    def close(self):
        self.aio.free_cpu_locked_tensor(self.pinned)
        return {}


if __name__ == "__main__":
    serve(Worker())
