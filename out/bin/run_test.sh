#!/bin/bash
SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
INSTALL_ROOT=$(dirname $SCRIPT_DIR)

if [ -f '/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash' ]; then
    source '/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash'
fi

# 永久添加LD_LIBRARY_PATH（支持libnpu_nvme的运行时路径）
export LD_LIBRARY_PATH=$INSTALL_ROOT/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH

# 日志输出（帮助调试）
echo ;[INFO];Executable:;$INSTALL_ROOT/bin/test_npu_nvme;
echo ;[INFO];LD_LIBRARY_PATH:;$LD_LIBRARY_PATH;

# 使用sudo运行目标测试程序且保留环境变量
sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH $INSTALL_ROOT/bin/test_npu_nvme $@