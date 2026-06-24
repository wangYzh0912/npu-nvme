# 架构速览 (2026-06-24)

## 代码结构

```
npu-nvme/
  python/          12 个模块 (direct_checkpoint 入口 + 6 子模块 + 3 工具 + profiler + delta_cell)
  src/              2 个 C 文件 (894 + 271 行)
  include/          1 公共头文件 (19 API) + 4 内部头文件
  experiments/      ~30 实验脚本 + common.py (4 共享函数) + delta_e2e/ (待建)
  to_do/active/     5 活跃规划文件
  to_do/archive/   23 归档文档
```

## 核心 API (C 层 19 函数)

| 类别 | 函数 |
|------|------|
| Init/Cleanup | `npu_nvme_init`, `npu_nvme_cleanup` |
| Query | `get_total_blocks`, `get_max_transfer` |
| Sync I/O | `npu_nvme_sync_meta_io` |
| Batch I/O | `write_batch`, `read_batch`, `write_batch_host`, `read_batch_host` |
| FaF | `register_tasks`, `set_probe_flag_ptr`, `set_probe_flag_value`, `get_probe_flag_dev_ptr`, `set_step_ptr` |
| Delta | `delta_init`, `delta_get_area_offset`, `delta_get_slot_size`, `delta_get_slot_count` |

## 重构历史 (2026-06-24)

| 指标 | 重构前 | 重构后 |
|------|:---:|:---:|
| `npu_nvme.c` | 1354 行 | 894 行 (-34%) |
| `direct_checkpoint.py` | 1525 行 | 1049 行 + 7 子模块 |
| to_do 目录 | 20 文件 | active/5 + archive/23 |

## 当前任务

1. **服务器验证**: ✅ 完成 (C compile 15/15 + A.5/A.6/A.7 + Step 1c BW)
2. **增量检查点全路径打通**: 🔲 待实现 (设计完成, 见 [DELTA_CHECKPOINT_DESIGN.md](DELTA_CHECKPOINT_DESIGN.md))
3. **论文实验**: 🔲 待实现
