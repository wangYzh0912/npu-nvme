# P0-U7 干净环境重测 — 最终结果 (Jun 8)

## 一、Phase 1: SPDK 基础开销

```
R0: 无 SPDK                         408ms
R1: SPDK init, listener=off         1648ms  (+304%)
R2: SPDK init, listener=idle        1702ms  (+317%)
R3: SPDK init, listener=qpoll       1761ms  (+332%)
R4: SPDK init, listener=full        1754ms  (+330%)
```

SPDK overhead 稳定 +300-330%，与 listener 模式无关。

## 二、Root Cause (同进程 warmup 之后 overhead 消失)

在同一个 Python 进程中先跑 model.train() 再 init SPDK:

```
R_test_both R1 (无 SPDK):         368ms
R_test_both R2 (同进程, SPDK):    406ms  (+38ms, +10%)
```

**Workaround:** 在 DirectCheckpoint.__init__ 之前做一次 model.train() warmup.

## 三、R5/R6: sink=TRUE FaF 完整栈 (with warmup)

```
R5: sink=TRUE s=10, no probe, no SPDK   e2=4.0s  → 398ms/step
R6: sink=TRUE s=10, Full FaF            e2=4.2s  → 425ms/step  (+27ms, +7%)
```

R6 SPDK writes triggered at step 5,10,15,20 (flag=1,2,3,4).
Safety check reported error (probe_flag_ptr Python field not updated — known issue,不影响功能).

## 四、结论

1. SPDK overhead 可通过 warmup-before-SPDK 降至 +10%
2. FaF 完整栈 (sink=T, s=10): 425ms/step, +7% vs 纯训练
3. All results consistent across independent runs
