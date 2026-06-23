#!/usr/bin/env python3
"""Clean up emotional comments and dev-phase markers in direct_checkpoint.py."""
with open('python/direct_checkpoint.py', 'r', encoding='utf-8') as f:
    c = f.read()

reps = [
    # dev-phase markers
    ('# 【修改点1】：获取名字', '# chunk metadata: parameter name for debugging'),
    ('# 【修改点2】：把名字塞进元组里', ''),
    ('# 【修改点3】：这里多解包一个 name', ''),
    ('# 【核心防爆修复】：检查 Superblock', '# Verify the stored stack start'),
    ('中的历史堆栈起点是否还能容纳当前的分布式阵列！', ' still fits the current multi-rank configuration'),
    ('# 【物理保险丝】：严格检查是否越出了这张 NVMe 盘的绝对物理容量',
     '# Bounds check: disk physical capacity'),
    ('# 客货分流：走主机内存的单独放入 host_params',
     '# Host-resident params go to host_params; device params go to dev_params'),
    ('# 【新增 Debug 代码】：落盘一份对照表，看看死锁的 Task 到底叫什么名字！',
     '# Debug: write chunk-to-parameter mapping for stall diagnosis'),
    ('# 【同步显存】保证前向/反向计算和优化器已完全结束',
     '# Ensure all pending device operations are complete before reading buffers'),
    ('# 3. 后台计算耗时并打印铁证', ''),
    ('# 主线程极速返回，让 MindSpore 跑 Step 16',
     '# Return immediately; I/O proceeds in background thread'),
    ('# 因为真实写入在后台，此处返回空壳数据应对外层的回调格式',
     '# Timing data is printed from the background thread; return values are for API compatibility only'),
    ('# C返回0表示成功，数据已写入buf。需从header解析实际帧大小。',
     '# Read succeeded; parse header to get actual frame size'),
    ('# I3 Delta (增量) I/O — S2/S3: 端到端打通',
     '# Delta frame I/O (incremental checkpoint)'),
    # docstrings
    ('利用底层同步拷贝的安全拦截特性，在初始化时自动摸清本卡的真实张量',
     'Probe each parameter with a 1-byte aclrtMemcpy to determine whether it resides on this rank.'),
    # ACK comment
    ('# 2 代表 ACL_MEMCPY_DEVICE_TO_HOST', ''),
    ('# 1 代表 ACL_MEMCPY_HOST_TO_DEVICE', ''),
    # replace long Chinese docstring
    ('防止下一步的优化器更新破坏 NPU 显存中的参数',
     'so the next optimizer step does not overwrite parameters being written'),
    # one-shot
    ('one-shot selftest', 'one-shot self-test'),
]

for old, new in reps:
    if old in c:
        c = c.replace(old, new)

# Clean up extra blank lines left by replaced comments
while '\n\n\n' in c:
    c = c.replace('\n\n\n', '\n\n')

with open('python/direct_checkpoint.py', 'w', encoding='utf-8') as f:
    f.write(c)
print('Done')
