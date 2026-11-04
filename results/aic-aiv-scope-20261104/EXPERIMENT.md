# KV 整理边界与布局对照

固定设备 910B3 4–7、TP4、BF16、greedy seed42、prefix cache关闭。共享服务与驱动不修改；包、镜像与源码身份分别留证。

顺序：图模式筛查 → 0.18三次模型重复 → 两个GQA模型各三次native/dense/PA重复 → 正确时点block table补测 → 8B长生成/长输入批次扩展 → 6616 blocks 同段PMU/HBM → 三次独立合成co-run。

核心矩阵每进程四个负载：b1/input128/output1、b1/input128/output32、b1/input2048/output32、b8/input128/output32。扩展矩阵对8B增加b1/input128/output128与b8/input2048/output32，各路径三次独立进程。未执行的交叉组合不是已测覆盖。

控制CV>5%的时延不纳入稳定比较；task profiler开销>10%的时长只用于诊断，task计数/shape和flow仍可用于路径归因。系统采集的PMU/HBM覆盖率与采样开销单独验收。

整缓存Slice要求完整名称、BF16 dtype和与实测K/V逻辑元素数相同的输出几何，避免小同名Slice混入。KV block数与kernel block dim区分。

连续视图控制只对未绑定、零初始化、无padding、普通GQA缓存生效，保持storage owner和分配字节数，原生cache writer更新同一视图。它不是预先整理后长期复用的过期副本。

b1精确block table序列是逻辑映射对照；b8异步batch进入/退出和物理ID分配可能不同，严格映射不相等时保留为不同调度的端到端输出等价证据。补测在_update_states之后、attention之前读取CPU侧映射，无NPU tensor读回。

真实模型的ready队列/依赖完成时间戳不可用，AIC调度机会保持unknown。合成co-run只测试已知独立矩阵/复制操作的任务envelope重叠，不替代真实模型依赖证明。

输出：REPORT.md、acceptance.json、每条件JSON/CSV、独立重复与未通过门槛清单、PNG/SVG/PDF图、镜像/代码/模型身份、命令/失败日志、PMU/HBM归一化输入与最终SHA256目录。
