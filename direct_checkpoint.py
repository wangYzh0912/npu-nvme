import torch
import ctypes
import time
import os
from typing import Dict, List

libnpu_nvme = ctypes. CDLL('./out/lib/libnpu_nvme.so')

class NPUNVMEContext(ctypes.Structure):
    pass

# 接口定义保持不变
libnpu_nvme.npu_nvme_init.argtypes = [ctypes. POINTER(ctypes. POINTER(NPUNVMEContext)), ctypes.c_char_p]
libnpu_nvme.npu_nvme_init.restype = ctypes.c_int

libnpu_nvme.npu_nvme_write_pipeline.argtypes = [
    ctypes.POINTER(NPUNVMEContext), 
    ctypes.c_void_p, 
    ctypes.c_uint64, 
    ctypes. c_size_t,
    ctypes.c_int
]
libnpu_nvme.npu_nvme_write_pipeline.restype = ctypes.c_int

libnpu_nvme.npu_nvme_write. argtypes = [
    ctypes.POINTER(NPUNVMEContext), 
    ctypes.c_void_p, 
    ctypes.c_uint64, 
    ctypes.c_size_t
]
libnpu_nvme.npu_nvme_write.restype = ctypes.c_int

libnpu_nvme.npu_nvme_cleanup.argtypes = [ctypes.POINTER(NPUNVMEContext)]
libnpu_nvme.npu_nvme_cleanup.restype = None

libnpu_nvme.npu_nvme_write_batch.argtypes = [
    ctypes.POINTER(NPUNVMEContext),
    ctypes.POINTER(ctypes.c_void_p),    # void **npu_buffers
    ctypes.POINTER(ctypes.c_uint64),    # uint64_t *nvme_offsets
    ctypes.POINTER(ctypes.c_size_t),    # size_t *sizes
    ctypes.c_int,                        # int num_items
    ctypes.c_int                         # int pipeline_depth
]
libnpu_nvme.npu_nvme_write_batch.restype = ctypes.c_int


libnpu_nvme. npu_nvme_write_batch_async.argtypes = [
    ctypes.POINTER(NPUNVMEContext),
    ctypes.POINTER(ctypes.c_void_p),    # void **npu_buffers
    ctypes.POINTER(ctypes.c_uint64),    # uint64_t *nvme_offsets
    ctypes.POINTER(ctypes.c_size_t),    # size_t *sizes
    ctypes.c_int,                        # int num_items
    ctypes.c_int                         # int pipeline_depth
]
libnpu_nvme.npu_nvme_write_batch_async.restype = ctypes.c_int



class DirectCheckpoint:
    def __init__(self, nvme_device:  str = "0000:83:00.0", use_pipeline: bool = True):
        self.ctx = ctypes. POINTER(NPUNVMEContext)()
        ret = libnpu_nvme. npu_nvme_init(ctypes.byref(self. ctx), nvme_device.encode())
        if ret != 0:
            raise RuntimeError("Failed to initialize NPU-NVMe interface")
        
        self.nvme_offset = 0
        self.metadata = {}
        self.use_pipeline = use_pipeline
        
        print(f"[Checkpoint] Mode: {'Pipeline' if use_pipeline else 'Sequential'}")

    def save(self, model:  torch.nn.Module, pipeline_depth: int = 4, chunk_size: int = 4 * 1024 * 1024) -> int:
        """保存模型到NVMe - 使用真正的Pipeline"""
        print("\n=== Starting checkpoint save ===")
        
        # 收集所有参数信息
        params_info = []
        total_size = 0
        
        for idx, (name, param) in enumerate(model.named_parameters()):
            if not param.is_npu:
                print(f"[{idx}] Skip {name}: not on NPU")
                continue
            
            npu_address = param.data_ptr()
            size = param.numel() * param.element_size()
            
            if npu_address == 0 or size == 0:
                print(f"[{idx}] Skip {name}: invalid")
                continue
            
            params_info.append({
                'name': name,
                'address': npu_address,
                'size': size,
                'offset': self.nvme_offset,
                'shape': list(param.shape),
                'dtype': str(param.dtype)
            })
            
            self.nvme_offset += size
            total_size += size
        
        print(f"[Checkpoint] Total parameters: {len(params_info)}")
        print(f"[Checkpoint] Total size: {total_size / 1024 / 1024:.2f} MB")
        
        # 开始传输（整体计时）
        start_time = time.time()
        
        if self.use_pipeline:
            # Pipeline模式：一次性传输所有数据
            self._save_all_pipeline(params_info, pipeline_depth, chunk_size)
        else:
            # Sequential模式
            self._save_all_sequential(params_info, chunk_size)
        
        elapsed = time.time() - start_time
        speed = total_size / elapsed / 1024 / 1024
        
        print(f"\n[Checkpoint] Transfer completed!")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Speed: {speed:.2f} MB/s")
        
        # 保存元数据
        for info in params_info:
            self.metadata[info['name']] = {
                'offset': info['offset'],
                'size': info['size'],
                'shape': info['shape'],
                'dtype': info['dtype']
            }

        # 将元数据保存到本地csv文件
        with open("checkpoint_metadata.csv", "w") as f:
            f.write("name,offset,size,shape,dtype\n")
            for name, meta in self.metadata.items():
                shape_str = "x".join(map(str, meta['shape']))
                f.write(f"{name},{meta['offset']},{meta['size']},{shape_str},{meta['dtype']}\n")

        
        torch.save(self.metadata, "checkpoint_metadata.pth")
        print(f"[Checkpoint] Metadata saved")
        
        return total_size

    def _save_all_pipeline(self, params_info:  List[dict], pipeline_depth: int, chunk_size: int):
        """Pipeline模式：使用Batch API一次性传输所有参数"""
        
        print(f"[Checkpoint] Using Batch Pipeline API")
        
        num_params = len(params_info)
        
        # 准备C数组
        npu_buffers = (ctypes.c_void_p * num_params)()
        nvme_offsets = (ctypes.c_uint64 * num_params)()
        sizes = (ctypes.c_size_t * num_params)()
        
        for i, info in enumerate(params_info):
            npu_buffers[i] = info['address']
            nvme_offsets[i] = info['offset']
            sizes[i] = info['size']
        
        # 调用batch API
        ret = libnpu_nvme.npu_nvme_write_batch(
            self.ctx,
            npu_buffers,
            nvme_offsets,
            sizes,
            num_params,
            pipeline_depth,
            chunk_size
        )
        
        if ret != 0:
            raise RuntimeError("Batch write failed")

    def _save_all_sequential(self, params_info: List[dict], chunk_size: int):
        """Sequential模式：普通传输"""
        for idx, info in enumerate(params_info):
            print(f"\n[{idx+1}/{len(params_info)}] {info['name']}: {info['size']/1024/1024:.2f} MB")
            
            ret = libnpu_nvme.npu_nvme_write(
                self.ctx,
                info['address'],
                info['offset'],
                info['size'],
                chunk_size
            )
            
            if ret != 0:
                raise RuntimeError(f"Failed to save {info['name']}")

    def load(self, model: torch.nn.Module, metadata_path: str = "checkpoint_metadata.pth", chunk_size: int = 4 * 1024 * 1024):
        """从NVMe加载模型"""
        self.metadata = torch.load(metadata_path)

        for name, param in model.named_parameters():
            if name not in self.metadata:
                continue
                
            meta = self.metadata[name]
            npu_address = param.data_ptr()
            size = meta['size']

            ret = libnpu_nvme. npu_nvme_read(
                self.ctx,
                npu_address,
                meta['offset'],
                size,
                chunk_size
            )
            if ret != 0:
                raise RuntimeError(f"Failed to load parameter:  {name}")
        
        print("Checkpoint loaded successfully!")

    def cleanup(self):
        """清理资源"""
        libnpu_nvme. npu_nvme_cleanup(self.ctx)