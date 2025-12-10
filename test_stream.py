import torch
import torch_npu
import time

# 创建多个stream
streams = [torch_npu.npu.Stream() for _ in range(4)]

# 准备数据
data_npu = [torch.randn(10, 1024, 1024).  to('npu:7') for _ in range(4)]
data_cpu = [torch.zeros(10, 1024, 1024) for _ in range(4)]

# 测试1: 串行拷贝
start = time.time()
for i in range(4):
    data_cpu[i].copy_(data_npu[i])
serial_time = time.time() - start

# 测试2: 并行拷贝（使用stream）
start = time.time()
for i in range(4):
    with torch_npu.npu.stream(streams[i]):
        data_cpu[i].copy_(data_npu[i])
# 等待所有stream
for s in streams:
    s.synchronize()
parallel_time = time.time() - start

print(f"Serial: {serial_time:.3f}s, {160/serial_time:.1f} MB/s")
print(f"Parallel: {parallel_time:.3f}s, {160/parallel_time:.1f} MB/s")
print(f"Speedup: {serial_time/parallel_time:.2f}x")