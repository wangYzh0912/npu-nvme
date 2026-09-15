# 原生测试

host_safety_test.c 编译实际 C 实现并使用 ASan/UBSan，包含 ring 边界和故障生命周期。v2_smoke_test 是唯一安装的 native smoke（仅授权 83:00.0）。

硬件传输矩阵位于 tests/hardware/transport_matrix.py；H02 与生命周期测试有独立证据。编译通过或 Host harness 通过不等于硬件完成。
