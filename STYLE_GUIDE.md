# NPU-NVMe 编码规范

> 基于 2026-06-23 C/Python 代码审查重构实践总结。

---

## 一、通用原则

1. **可读性优先**：代码是写给人看的。注释描述**为什么**这么做，而非重复代码做了什么。
2. **无情绪语言**：禁止 `【】`、`！！`、`！` 等情绪标记。使用平实的技术描述。
3. **禁止比喻**：禁止 `暴毙`、`幽灵`、`狠狠踹`、`防线`、`保险丝`、`看门狗`、`死锁`（除非确指 deadlock）、`铁证`、`空壳`、`客货分流` 等比喻/拟人词汇。
4. **禁止开发标记**：禁止在注释中出现 `Phase N`、`Step N`、`I1/I2/I3`、`E2.1`、`P2-1` 等开发阶段标签。代码注释只描述当前功能。
5. **禁止编号缺口**：不要使用 `Section 1/2/3/5/6/7` 这类有缺口的节编号。
6. **magic number 必须命名**：除 `0`、`1`、`-1` 外，所有字面常量必须有 `#define`（C）或模块级常量（Python）。

---

## 二、C 代码规范 (`src/*.c`, `include/*.h`)

### 文件头

```c
/* =======================================================================
 * filename.c — one-line description
 *
 * Detailed description of what this file provides.
 * ======================================================================= */
```

### Section 分隔

```c
/* ---- Section Title ---- */
```

### 函数注释

```c
/**
 * @brief Brief description.
 *
 * @param ctx  context handle
 * @param ptr  source pointer
 * @return     0 on success, -1 on error
 */
int npu_nvme_do_something(NPUNVMEContext *ctx, void *ptr);
```

内部函数使用单行注释：

```c
/* Brief description. */
static int internal_helper(...) { ... }
```

### 行内注释

```c
int result = 0;
// Explain non-obvious logic here.
if (edge_case) {
    result = fallback_path();
}
```

### 命名

| 类型 | 规范 | 示例 |
|------|------|------|
| 公开 API | `npu_nvme_` 前缀 + snake_case | `npu_nvme_write_batch` |
| 内部函数 | `static` + snake_case | `ensure_hugepages` |
| 结构体 | snake_case + `_t` 后缀 | `io_task_t` |
| 枚举值 | `UPPER_CASE` + 前缀 | `CHUNK_IDLE`, `CHUNK_DONE` |
| 宏/常量 | `UPPER_CASE` | `MIN_PIPE_DEPTH`, `ALIGN_4K(x)` |

### 禁止

- 全局可变状态（除非有明确的多 context 共享理由并标注）
- 函数内部的 `#define`（宏统一放文件顶部）
- `#if 0` 死代码块（用 git history 回退）
- `int` 用于非负值（用 `uint32_t`、`size_t`）
- 返回 bytes 的函数名叫 `get_blocks`

---

## 三、Python 代码规范 (`python/*.py`)

### 文件头

```python
"""Module-level docstring describing what this module provides.

Usage:
- Import DirectCheckpoint from this module.
- Used by training scripts under experiments/.
"""
```

### 方法 docstring

```python
def method_name(self, arg1: type, arg2: int = 0):
    """Brief description.

    Args:
        arg1: description
        arg2: description (default 0)

    Returns:
        description of return value

    Raises:
        RuntimeError: when something goes wrong
    """
```

### 常量

Python 使用简洁的 `#` 注释分组，不使用 `====` 分隔线（那是 C 的风格）。

```python
# -- Group description --
CONSTANT_NAME = value
```

### 命名

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `DirectCheckpoint`, `ProbeTrainOneStepCell` |
| 方法/函数 | snake_case | `build_chunks`, `_commit_metadata` |
| 私有方法 | `_` 前缀 | `_prepare_params`, `_mount_filesystem` |
| 常量 | `UPPER_CASE` | `SUPERBLOCK_OFFSET`, `MAGIC_NUMBER` |
| 模块级变量 | `_UPPER_CASE`（内部） | `_LIB_PATH` |

### 禁止

- 模块级代码中混入属于函数体的 import（如函数 `return` 之后的 `from x import y`）
- `hasattr` 检查 C 函数是否存在，但 C 层从未实现该函数（dead binding，应删除）
- docstring 中的情绪/开发标记（同通用原则）
- 延迟 import 不做 fallback（如 `from i3_delta_writer import ...` 应验证模块存在）

### DEPRECATED 标记

```python
# DEPRECATED: brief reason.  Used only by <list of callers>.
# New code should use <replacement API>.
class OldClass:
    ...
```

---

## 四、注释规范（通用）

### 允许

- `—`（em-dash，`U+2014`）、`→`（箭头，`U+2192`）等标点符号
- 技术数字和参数说明
- 硬件行为描述
- 设计决策的简要说明（"为什么这样做"）

### 禁止

| 类别 | 示例 | 替换 |
|------|------|------|
| 情绪标记 | `【修复】`、`【核心修改】` | 直接删除标记 |
| 感叹号 | `！`、`！！` | `.` |
| 比喻/拟人 | `暴毙`、`幽灵 DMA`、`狠狠踹` | 标准技术描述 |
| 战争隐喻 | `防线`、`保险丝` | `validation`、`check` |
| 开发标签 | `Phase 2-A`、`E2.1`、`I1/I2/I3` | 删除，写功能描述 |
| 过时引用 | `参考之前的回复` | 删除 |
| 重复注释 | 同一内容出现两次 | 删除重复行 |

### ctypes 绑定注释

```python
lib.some_func.argtypes = [
    ctypes.c_void_p,    # ctx
    ctypes.c_size_t,    # size
    ctypes.c_int        # direction
]
# 每个参数一行，注释在右侧
```

---

## 五、Git Commit 规范

### 消息格式

```
<分类>: <简短描述>

<详细说明（可选）>
- 变更项1
- 变更项2

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

### 分类前缀

| 前缀 | 用途 |
|------|------|
| `backup:` | 记录基线 commit |
| `Step N:` | 独立的重构步骤 |
| `fixup:` | 上一步的遗漏修复 |
| `cleanup:` | 纯化妆品/注释修改 |
| `docs:` | 文档变更 |
| `feat:` | 新功能 |
| `fix:` | Bug 修复 |

### 每步原则

- 一个 commit 只做一件事
- 每步可独立 `git revert`
- 逻辑变更和化妆品变更分离

---

## 六、代码审查流程

参见 `to_do/CODE_REVIEW_WORKFLOW.md`。

简要流程：
1. 记录基线 commit
2. 全局 grep 分析调用图
3. 分类：保留 / 保留+改 / 死代码 / 降级
4. 按严重程度分级（🔴 严重 → 🟡 中等 → 🟢 轻微）
5. 制定量化修改方案
6. 分步独立 commit 执行
7. `git diff baseline..HEAD` 审查确认

---

## 七、批量修改工具

对于 20+ 处的重复模式修改，使用 Python 脚本而非逐个手动编辑：

```python
with open('target_file', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    (old_string, new_string),
    ...
]
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)

with open('target_file', 'w', encoding='utf-8') as f:
    f.write(content)
```

**注意**：文件编码必须是 `utf-8`。脚本结束后检查 WARNING 输出，手工修复因编码不匹配而遗漏的项。
