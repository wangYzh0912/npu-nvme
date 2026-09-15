# 保留的方法实现

统一入口为 train.py；配置 schema 2，方法注册在 adapters。mindspore_native_save 调用框架保存；datastates_acl/pccheck_acl 保留各自语义，未执行 upstream 核心；bytecheckpoint_host/fastpersist_host 执行锁定 upstream CPU worker；ours 使用严格 FULL ABI 2。none 只作训练对照。

fit/benchmark 独立执行源和恢复进程，任何恢复子进程失败均使总结果失败。continue-on-failure 只允许继续收集其他方法，不改变最终失败退出码。timing 不能证明恢复正确，verify 必须校验状态与有覆盖的续训 oracle。
