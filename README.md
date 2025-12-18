# 环境设置和运行指南
## 准备步骤
切换到root环境：
```bash
sudo su -
```
激活user3下的conda环境：
```bash
source /home/user3/.bashrc
conda activate spdk
```
设置环境变量：
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64:/usr/local/Ascend/ascend-toolkit/latest/hccl/lib64
export HF_ENDPOINT=https://hf-mirror.com
export ASCEND_HOME_PATH=/usr/local/Ascend
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/$(arch):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:$LD_LIBRARY_PATH
export PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
export PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:$PATH
export ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
export ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
export TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
export ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
```

## 运行测试脚本
项目文件夹为/home/user3/npu-nvme，测试脚本为test.py。
```bash
python test.py
```
测试模型、流水线深度、NVMe一次写入的Chunk size等参数均可在test.py文件中直接修改。
保存检查点时，会输出checkpoint_metadata.csv文件。
该文件记录模型中各个变量的名称、写入偏移、大小（单位为Byte）。

由于目前ACL的调用有些bug没修好，运行后最后输出这些内容是正常的，不用在意：
```bash
[W compiler_depend.ts:316] Warning: NPU warning, error code is 100037[Error]: 
[Error]: Repeated deinitialization.
        Check whether the acl.finalize interface is repeatedly invoked for deinitialization.
EH9999: Inner Error!
EH9999: [PID: 476369] 2025-12-14-19:00:28.262.491 [Finalize][Acl]repeatedly finalized[FUNC:ReportInnerError][FILE:log_inner.cpp][LINE:145]
        TraceBack (most recent call last):
 (function operator())
 ```