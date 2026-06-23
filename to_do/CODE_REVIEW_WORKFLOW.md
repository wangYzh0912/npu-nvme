# C 层代码审查与重构流程

> 持久化于 2026-06-23，基于 `src/npu_nvme.c` 审查重构实践总结。

---

## 一、审查流程

### Phase 1: 基线建立

```bash
# 记录当前 commit 到 README 或独立文件，确保可回退
COMMIT=$(git rev-parse HEAD)
echo "<!-- BACKUP: baseline commit $COMMIT before refactor -->" >> README.md
git add README.md && git commit -m "backup: record baseline commit before refactor"
```

### Phase 2: 调用图分析

对每个待审查的函数/变量，做全局 grep 确认调用者：

```bash
# 对每个符号名，搜索 Python 和 C 中的调用
grep -rn "function_name" --include="*.py" --include="*.c" --include="*.h" .
```

**判定规则**：
| 调用者数 | 位置 | 判定 |
|:---:|------|:---:|
| 0 | 仅自身定义 + Python ctypes 绑定 | 死代码，可删除（绑定一起删） |
| 0 | 仅自身定义，无 Python 绑定 | 死代码，可删除 |
| >0 | 仅旧实验文件（非 `baselines/` 目录） | 保留，加 TODO 标记 |
| >0 | `direct_checkpoint.py` 或 `baselines/` | 保留，API 不可变 |

### Phase 3: 代码分类

将每个文件/代码块按以下维度标记：

| 标记 | 含义 | 处理 |
|------|------|------|
| **保留** | 核心功能，有活跃调用者 | 只改注释，不动逻辑 |
| **保留+改** | 有调用者但实现有 bug | 修复 bug，保持 API 兼容 |
| **死代码** | 0 调用者 | 删除 |
| **降级** | 有调用者但不符设计预期 | 保留+加 TODO，后续统一迁移 |

### Phase 4: 问题分级

按严重程度排列：

| 级别 | 标志 | 示例 |
|------|:---:|------|
| 🔴 严重 | 运行时崩溃/死循环/数据错误 | 全局变量竞态、死循环、返回值错误 |
| 🟡 中等 | 命名误导、未声明函数、不规范写法 | 函数返回 bytes 却叫 `get_blocks` |
| 🟢 轻微 | 死代码、注释风格、备份文件 | `.bak` 文件、无调用者的 stub |

### Phase 5: 量化方案

为每个变更制定量化指标（行数、文件数、Bug 数），分步独立 commit。

---

## 二、注释规范

### 目标风格

```c
/* =======================================================================
 * npu_nvme.c — 功能一句话描述
 *
 * 详细描述，说明模块职责和数据流方向。
 * ======================================================================= */

/* ---- 小分隔：功能区块标题 ---- */

/** 函数注释（Doxygen 风格，可选） */
static int example_func(int arg) {
    int result = 0;
    // 行内注释：解释非显而易见的逻辑
    return result;
}
```

### 禁止项

| 类型 | 示例 | 替换为 |
|------|------|------|
| 情绪标记 | `【修复】`、`【核心修改】`、`【非常关键】` | 直接删除标记，保留内容 |
| 感叹号 | `！`、`！！` | `.` |
| 比喻/拟人 | `暴毙`、`幽灵 DMA`、`狠狠踹`、`拯救` | 标准技术描述 |
| 战争隐喻 | `防线`、`保险丝`、`安全防御` | `validation`、`check`、`matching` |
| 开发阶段标签 | `Phase 2-A`、`Step 3`、`E2.1`、`I1/I2/I3` | 删除，写功能描述 |
| 节编号 | `Section 1/2/3/5/6/7`（有缺口） | 删除编号，用 `----` 分隔 |
| 重复注释 | 同一行出现两次 | 删除重复行 |
| 开发笔记 | `参考之前的回复`、`保持不变` | 删除 |

### 保留项

- `—`（em-dash）、`→`（箭头）等标点符号
- 技术数字和参数说明
- 硬件行为描述（如 `NPU driver pre-allocates...`）

---

## 三、Commit 规范

### 每步独立

每个 commit 只做一件事。如果出错可以单独 revert。

### Commit message 格式

```
Step N / 简短分类: 一句话描述

变更详情（可选）：
- 项目1
- 项目2

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

### 示例

```
Step 1: remove dead files — backups, transfer wrapper, aicpu_probe

Deleted (9 files, ~2800 lines):
- src/npu_nvme.c.bak, .bak2, .clean — backup copies
- src/aicpu_probe.cc — WaitProbe dead code
...

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

## 四、批量修改工具

对于 20+ 处的重复模式修改，使用 Python 脚本而非逐个 Edit：

```python
# tools/normalize_comments.py
with open('src/npu_nvme.c', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    (old_string, new_string),
    ...
]

for old, new in replacements:
    if old in content:
        content = content.replace(old, new)

with open('src/npu_nvme.c', 'w', encoding='utf-8') as f:
    f.write(content)
```

**注意事项**：
- 文件编码必须是 `utf-8`，否则中文/特殊字符匹配会失败
- 脚本文件本身用纯 ASCII 写（避免 em-dash、中文引号等）
- 运行后检查 `WARNING` 行，确认 encoding 导致的 miss 并手工修复

---

## 五、验证清单

每轮审查/重构完成后逐项检查：

| 检查项 | 命令 |
|------|------|
| 是否有残留调用者引用已删函数 | `grep -r "deleted_func" .` |
| 编译通过 | `bash build.sh`（如有环境）或肉眼审查 CMakeLists |
| Python import 不报错 | `python -c "from direct_checkpoint import DirectCheckpoint"` |
| `【】` 残留 | `grep -c '【\|】' src/*.c include/*.h python/*.py` |
| Section 编号残留 | `grep 'Section [0-9]' src/*.c` |
| 情绪化 `！！` 残留 | `grep '！！' src/*.c` |
| 死文件残留 | `ls src/*.bak src/*.bak2 2>/dev/null` |
| diff 审查：无意外逻辑变更 | `git diff baseline..HEAD -- src/npu_nvme.c` |
| 全局变量竞态 | 检查是否有多 context 共享的全局变量 |

---

## 六、当前重构历史（参考）

| Commit | 内容 | 变化 |
|------|------|:---:|
| `e3d95d4` | 记录基线 commit | +1 |
| `1877d95` | Step 1: 删除死文件 | -2954 |
| `f4f03bc` | Step 2: 删除死 C 代码 | -122 |
| `ab2668f` | Step 3: 删除 max_transfer 死字段 | ±4 |
| `c2635c1` | Step 4+6: 注释规范化 | +356/-85 |
| `e0643f7` | Step 5: 修复 5 个 Bug | +25/-8 |
| `cb7ec4c` | Step 7: 删除 Python 死绑定 | -30 |
| `2f7af95` | fixup: 编码遗漏的注释 | +14/-20 |
| `8f0a04b` | 清理: 开发标签+情绪标记 | +579/-90 |
