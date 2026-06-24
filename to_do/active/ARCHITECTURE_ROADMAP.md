# 架构速览 (2026-06-27)

## 代码结构

```
npu-nvme/
  python/          11 个模块 (direct_checkpoint 入口, 6 子模块 + 3 工具 + profiler)
  src/              2 个 C 文件 (894 + 279 行), 16 公共 API
  include/          1 公共头文件 + 4 内部头文件
  experiments/      ~27 实验脚本 + common.py (5 共享函数)
  to_do/active/     4 活跃规划文件
  to_do/archive/   17 归档文档
```

## 核心 API (C 层 17 函数)

| 类别 | 函数 |
|------|------|
| Init/Cleanup | `npu_nvme_init`, `npu_nvme_cleanup` |
| Query | `get_total_blocks`, `get_max_transfer` |
| Sync I/O | `npu_nvme_sync_meta_io` |
| Batch I/O | `npu_nvme_write_batch`, `npu_nvme_read_batch`, `npu_nvme_write_batch_host` |
| FaF | `register_tasks`, `set_probe_flag_ptr`, `set_probe_flag_value`, `get_probe_flag_dev_ptr`, `set_step_ptr` |
| Delta | `delta_init`, `delta_get_area_offset`, `delta_get_slot_size`, `delta_get_slot_count` |

## 重构历史 (2026-06-24)

| 指标 | 重构前 | 重构后 |
|------|:---:|:---:|
| `npu_nvme.c` | 1354 行 | 894 行 (-34%) |
| `direct_checkpoint.py` | 1525 行 | 1049 行 + 7 子模块 |
| to_do 目录 | 20 文件 | active/4 + archive/23 |

## 当前任务

1. **Phase A**: NPU 服务器验证 (编译 + 测试)
2. **Phase B**: I3 Step 3 全路径打通
3. **Phase C**: 论文实验 (E0→E1/E2/E3)
