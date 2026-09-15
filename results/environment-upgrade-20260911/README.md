# 用户环境升级：首轮实现与阻塞记录

日期：2026-09-11。整体状态：**部分实现，环境升级尚未验收**。

## 证据

- `inventory-final.json`：最终启动配置下的真实盘点；`inventory.json` 保留早先盘点。
- `P1_P9_REVIEWED.md` / `evidence_summary.json`：只读复核 9 月 9–10 日原始结果，
  不覆盖历史报告，也没有重跑该轮硬件实验。
- 完整后续流程：`docs/USER_ENVIRONMENT_UPGRADE_PLAN.md`。

驱动文件为 24.1.rc3，SHA-256 为
`fdb4e9d9c658d5e935f8be57f4f235d4d00e68f69d2e710a77448f2bee639709`。
CANN 为发行版 8.0.RC3、内部 runtime 7.5.0.1.129；Python 3.9.25、MindSpore 2.5.0、
MindFormers 1.3.2。未安装候选栈，未修改驱动、固件、系统 CANN 或默认 shell 配置。

## 已完成

1. 用户环境启动器：显式 old/candidate，清除继承的搜索路径与 shell hooks，识别
   Python 包版本、CANN 组件、驱动和项目库；拒绝缺失依赖、stub、其他 CANN 及私有
   目录外的候选路径。`candidate` 为 null，默认晋升未实现，仍使用 old。
2. 实际路径兼容：MindSpore 2.5 直接通过版本目录导入会在 GetAscendPath 中止；原有
   latest 入口及修正后的启动器均成功导入 2.5.0。old 沿用已有 latest，但校验项目
   依赖的真实文件仍在 8.0.RC3，系统链接未改动。导入成功不是硬件回归通过。
3. 环境盘点：权限错误与缺失分开，记录实际所选 Python 版本及兼容性待定状态；
   输出拒绝覆写，阻塞返回非零；CANN 发行版与 runtime 内部版本分开保存。
4. 证据身份：experiment_id、environment_id、parent_run_id、run_role；P2 子负载
   通过独立子进程环境继承父 ID。既有原始记录不回写。
5. 统计与阶段状态：P1 主运行 60/60；辅助 5 个为一个 smoke 加四个 P2 内嵌 P1。
   P2 主运行六个全部 degraded；P3 一个 serial。P2 退出与恢复旧编排时都检查时间
   闭合，不以零退出码代替通过。
6. P2 tracing 使用唯一命名 instance；不可创建时降级，不清空或开关全局 tracefs。

## 验证结果

- 最终针对性测试 21/21：环境隔离 10、结果归属/阶段状态/tracefs 8、既有证据包 3。
- 修改模块语法编译及 `git diff --check` 通过。
- 旧环境完整 unittest discovery 在当时版本运行 94 项：92 通过，2 个模块导入失败，
  原因为旧环境没有 pytest（test_category_delta、test_p6_vector_timeline）。后续增加的
  P2 时间闭合测试已单独通过；未向旧环境安装包。
- 该 unittest 命令不收集 pytest 风格函数，不能作为完整 pytest 矩阵验收。
- 未执行 C 构建、NPU/NVMe 回环、训练、安装或多卡恢复；本轮未修改 C 核心实现。

## 继续执行需要的条件

当前会话没有 `/dev/davinci*`、`/dev/uio*`；firmware 和 Qwen3-8B 配置读取被拒绝。
GitHub DNS/HTTPS 与官方文档请求失败；本地搜索也未发现可用的 CANN/MindSpore/
MindFormers 安装包，找到的仅是历史自定义算子包。当前写权限仅覆盖工作区和 /tmp，
不覆盖约定的 `/models/npu_nvme_exp/user7-stack/`；`.git` 也只读。

因此候选兼容性结论仍为 **not_determined**，不是“不支持 Qwen3”。远端新 master
内容未核实，未切换/合并/提交分支。要继续，需在可读取官方兼容资料和模型、暴露设备、
可写私有安装目录及集成 checkout 的目标机执行上下文中恢复 E0；固定驱动约束不变。

后续仍待实际完成：发行组合筛选、隔离安装、GPT-2 回归、Qwen3-8B 四卡状态分片与
Native/Ours 完整恢复、旧→新→旧回退、环境晋升、XL live 严格续训修复以及新性能矩阵。
