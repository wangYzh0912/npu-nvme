import torch
import ctypes
from typing import Dict

# 加载共享库
libnpu_nvme = ctypes.CDLL('./out/lib/libnpu_nvme.so')

# 定义C类型接口
class NPUNVMEContext(ctypes.Structure):
    pass

libnpu_nvme.npu_nvme_init.argtypes = [ctypes.POINTER(ctypes. POINTER(NPUNVMEContext)), ctypes.c_char_p]
libnpu_nvme.npu_nvme_init.restype = ctypes.c_int

libnpu_nvme.npu_nvme_init.argtypes = [ctypes. POINTER(ctypes. POINTER(NPUNVMEContext)), ctypes.c_char_p]
libnpu_nvme.npu_nvme_init.restype = ctypes.c_int

libnpu_nvme.npu_nvme_write.argtypes = [ctypes.POINTER(NPUNVMEContext), ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
libnpu_nvme.npu_nvme_write.restype = ctypes.c_int

libnpu_nvme.npu_nvme_read.argtypes = [ctypes.POINTER(NPUNVMEContext), ctypes.c_void_p, ctypes. c_uint64, ctypes. c_size_t]
libnpu_nvme.npu_nvme_read.restype = ctypes.c_int

libnpu_nvme.npu_nvme_write_pipeline.argtypes = [
    ctypes.POINTER(NPUNVMEContext), 
    ctypes.c_void_p, 
    ctypes.c_uint64, 
    ctypes.c_size_t,
    ctypes.c_int  # pipeline_depth
]
libnpu_nvme.npu_nvme_write_pipeline. restype = ctypes.c_int

libnpu_nvme.npu_nvme_cleanup.argtypes = [ctypes.POINTER(NPUNVMEContext)]
libnpu_nvme.npu_nvme_cleanup.restype = None


class DirectCheckpoint:
    def __init__(self, nvme_device: str = "0000:83:00.0", use_pipeline: bool = False, pipeline_depth: int = 4):
        self.ctx = ctypes.POINTER(NPUNVMEContext)()
        ret = libnpu_nvme. npu_nvme_init(ctypes.byref(self. ctx), nvme_device.encode())
        if ret != 0:
            raise RuntimeError("Failed to initialize NPU-NVMe interface")
        
        self.nvme_offset = 0
        self.metadata = {}
        self.use_pipeline = use_pipeline
        self.pipeline_depth = pipeline_depth

    def save(self, model: torch.nn.Module):
        """保存模型到NVMe"""
        print("\n=== Starting checkpoint save ===")
        
        size_total = 0

        for idx, (name, param) in enumerate(model.named_parameters()):
            # 获取参数信息
            npu_address = param.data_ptr()
            size = param.numel() * param.element_size()
            
            print(f"\n[{idx}] Saving parameter: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  Dtype: {param.dtype}")
            print(f"  NPU address: 0x{npu_address:x}")
            print(f"  Size: {size} bytes ({size / 1024 / 1024:.2f} MB)")
            print(f"  NVMe offset: {self.nvme_offset} bytes")
            
            # 检查参数是否在NPU上
            if not param.is_npu:
                print(f"  WARNING: Parameter is not on NPU!  Device: {param.device}")
                continue
            
            # 检查地址是否有效
            if npu_address == 0:
                print(f"  ERROR: Invalid NPU address (NULL)")
                raise RuntimeError(f"Invalid NPU address for parameter: {name}")
            
            # 检查大小是否合理
            if size == 0:
                print(f"  WARNING: Parameter has zero size, skipping")
                continue
            
            # 尝试写入
            print(f"  Calling npu_nvme_write...")
            if self.use_pipeline:
                print(f"  Using pipeline with depth {self.pipeline_depth}")
                ret = libnpu_nvme. npu_nvme_write_pipeline(
                    self.ctx,
                    npu_address,
                    self.nvme_offset,
                    size,
                    self.pipeline_depth
                )
            else:
                print(f"  Using direct write")
                ret = libnpu_nvme.npu_nvme_write(
                    self.ctx,
                    npu_address,
                    self.nvme_offset,
                    size
                )
            
            if ret != 0:
                print(f"  ERROR: Write failed with return code: {ret}")
                raise RuntimeError(f"Failed to save parameter: {name}")
            
            print(f"  ✓ Write successful")
            
            # 记录元数据
            self.metadata[name] = {
                'offset': self.nvme_offset,
                'size': size,
                'shape': list(param.shape),
                'dtype': str(param.dtype)
            }
            self.nvme_offset += size
            size_total += size
        
        # 保存元数据
        torch.save(self.metadata, "checkpoint_metadata.pth")
        print(f"\n=== Checkpoint saved successfully ===")
        print(f"Total size: {self.nvme_offset / 1024 / 1024:.2f} MB")
        print(f"Metadata saved to: checkpoint_metadata.pth")
        return size_total

    def load(self, model: torch.nn.Module, metadata_path: str = "checkpoint_metadata.pth"):
        """从NVMe加载模型"""
        self.metadata = torch.load(metadata_path)

        for name, param in model.named_parameters():
            meta = self.metadata[name]
            npu_address = param.data_ptr()
            size = meta['size']

            ret = libnpu_nvme. npu_nvme_read(
                self.ctx,
                npu_address,
                meta['offset'],
                size
            )
            if ret != 0:
                raise RuntimeError(f"Failed to load parameter: {name}")
        
        print("Checkpoint loaded successfully!")

    def cleanup(self):
        """清理资源"""
        libnpu_nvme.npu_nvme_cleanup(self.ctx)