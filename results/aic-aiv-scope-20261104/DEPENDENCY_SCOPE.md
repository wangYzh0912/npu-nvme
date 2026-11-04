# 调度可并行性的证据边界

当前调用链中，FIA attention 是 K/V 整理结果的消费者；直接让同一 attention 跳过必要的转换完成依赖不合法。CPU/设备 flow 可证明发射来源，设备任务时间可观察 envelope 重叠，但均不给出 runnable 队列或 ready 时间。

本轮不依据低 AIC busy 断言“现有调度遗漏了本可执行的AIC任务”。真实模型 ready 状态为 unknown，后续相邻层 co-run 在获得明确独立数据/依赖证明前不实施。

独立合成 co-run 使用两个没有数据依赖的固定任务：BF16 4096矩阵乘和块交错视图的连续复制。串行与分流 co-run 完成相同工作；数值输出验收、原生task flow归因、任务envelope交叠和无profiler耗时分别保存。即使有效并发，也只是设备能运行独立工作，不代表真实模型有相同工作可用。
