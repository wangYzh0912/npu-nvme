"""Weights-only in-place compatibility restore; no strict-ready guarantee."""
from __future__ import annotations
import time
import mindspore as ms
from mindspore import ops
from npu_nvme.framework.capture import rebuild_chunks_from_meta
from npu_nvme.storage.chunks import build_ctypes_arrays

class LegacyWeightsRestore:
    def __init__(self, *, binding, context, commit, total_bytes, rank_id, chunk_size, pointer_of):
        self.binding = binding
        self.ctx = context
        self.commit = commit
        self.total_bytes = total_bytes
        self.rank_id = rank_id
        self.chunk_size = chunk_size
        self.pointer_of = pointer_of

    def load(self, model: ms.nn.Cell, step: int = None,
             meta_path: str = "checkpoint_meta.pkl"):
        t_start = time.time()

        # NVMe metadata is the source of truth.  The pickle is retained only
        # as a diagnostic sidecar and must never make a restarted process
        # appear to have a checkpoint that was not persisted to the device.
        self.commit.mount(self.total_bytes, self.rank_id)

        if step is not None:
            ckpt_key = f"step_{step}"
        else:
            valid_keys = [k for k in self.commit.state.meta_dict.get("checkpoints", {}).keys()
                          if "complete" not in k]
            if not valid_keys:
                raise FileNotFoundError(
                    "No checkpoints found in Meta Dictionary!")
            ckpt_key = sorted(valid_keys,
                              key=lambda x: int(x.split('_')[1]))[-1]

        if ckpt_key not in self.commit.state.meta_dict.get("checkpoints", {}):
            raise FileNotFoundError(
                f"Checkpoint for {ckpt_key} not found!")

        meta_info = self.commit.state.meta_dict["checkpoints"][ckpt_key]
        chunk_size = min(meta_info.get("chunk_size", self.chunk_size),
                         self.chunk_size)

        t_rebuild = time.time()
        dev_chunks, host_chunks, buffers = rebuild_chunks_from_meta(
            model, meta_info["params"], chunk_size, pointer_of=self.pointer_of)
        t_rebuild_end = time.time()

        t0 = time.time()
        total_read = 0

        # Read NPU-resident parameters via DMA.
        if dev_chunks:
            c_ptrs, c_offs, c_sizes = build_ctypes_arrays(dev_chunks)
            rc = self.binding.npu_nvme_read_batch(
                self.ctx, c_ptrs, c_offs, c_sizes, len(dev_chunks))
            if rc != 0:
                raise RuntimeError("read_batch failed")
            total_read += sum(c[2].value for c in dev_chunks)

        # Read CPU-resident parameters through the explicit Host path.
        if host_chunks:
            if not hasattr(self.binding, "npu_nvme_read_batch_host"):
                raise RuntimeError(
                    "C library missing npu_nvme_read_batch_host")
            c_ptrs, c_offs, c_sizes = build_ctypes_arrays(host_chunks)
            rc = self.binding.npu_nvme_read_batch_host(
                self.ctx, c_ptrs, c_offs, c_sizes, len(host_chunks))
            if rc != 0:
                raise RuntimeError("read_batch_host failed")
            total_read += sum(c[2].value for c in host_chunks)

        t1 = time.time()
        t_update = time.time()

        for buf in buffers:
            if buf.get("use_dev", False):
                continue
            param = buf["param_ref"]
            if buf["np_arr"] is not None:
                tensor = ms.Tensor(buf["np_arr"], dtype=param.dtype)
                ops.assign(param, tensor)
        t_end = time.time()

        pure_read_time = t1 - t0
        total_time = t_end - t_start
        bw_e2e = total_read / 1024 / 1024 / total_time if total_time > 0 else 0

        stats = {
            "prepare_time": t_rebuild_end - t_rebuild,
            "read_time": pure_read_time,
            "set_data_time": t_end - t_update,
            "total_time": total_time,
            "bw_pure": (total_read / 1024 / 1024 / pure_read_time
                       if pure_read_time > 0 else 0),
            "bw_e2e": bw_e2e,
        }
        return total_read, len(dev_chunks) + len(host_chunks), total_time, bw_e2e, stats

