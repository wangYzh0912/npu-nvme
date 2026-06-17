#!/usr/bin/env python3
"""Merge WaitProbe + TriggerProbe into MindSpore internal AICPU config."""
import json

MS_CONFIG = '/root/miniconda3/envs/ms_2.5/lib/python3.9/site-packages/mindspore/lib/plugin/ascend/custom_aicpu_ops/op_impl/cpu/config/cust_aicpu_kernel.json'

with open(MS_CONFIG) as f:
    cfg = json.load(f)

cfg['WaitProbe'] = {
    'input0': {'format': 'ND', 'name': 'flag', 'paramType': 'required', 'type': 'DT_UINT32'},
    'input1': {'format': 'ND', 'name': 'expected', 'paramType': 'required', 'type': 'DT_UINT32'},
    'opInfo': {
        'computeCost': '100', 'engine': 'DNN_VM_AICPU', 'flagAsync': 'False',
        'flagPartial': 'False', 'formatAgnostic': 'False', 'functionName': 'RunCpuKernel',
        'kernelSo': 'libcust_aicpu_kernels_ms.so', 'opKernelLib': 'CUSTAICPUKernel',
        'opsFlag': 'OPS_FLAG_CLOSE', 'subTypeOfInferShape': '1',
        'userDefined': 'True', 'workspaceSize': '1024'
    },
    'output0': {'format': 'ND', 'name': 'y', 'paramType': 'required', 'type': 'DT_UINT32'}
}

cfg['TriggerProbe'] = {
    'input0': {'format': 'ND', 'name': 'step', 'paramType': 'required', 'type': 'DT_INT32'},
    'input1': {'format': 'ND', 'name': 'interval', 'paramType': 'required', 'type': 'DT_INT32'},
    'input2': {'format': 'ND', 'name': 'trigger_buf', 'paramType': 'required', 'type': 'DT_UINT32'},
    'input3': {'format': 'ND', 'name': 'expected', 'paramType': 'required', 'type': 'DT_UINT32'},
    'opInfo': {
        'computeCost': '100', 'engine': 'DNN_VM_AICPU', 'flagAsync': 'False',
        'flagPartial': 'False', 'formatAgnostic': 'False', 'functionName': 'RunCpuKernel',
        'kernelSo': 'libtrigger_probe_kernels_ms.so', 'opKernelLib': 'CUSTAICPUKernel',
        'opsFlag': 'OPS_FLAG_CLOSE', 'subTypeOfInferShape': '1',
        'userDefined': 'True', 'workspaceSize': '1024'
    },
    'output0': {'format': 'ND', 'name': 'y', 'paramType': 'required', 'type': 'DT_INT32'}
}

with open(MS_CONFIG, 'w') as f:
    json.dump(cfg, f, indent=2)

print(f'Merged {len(cfg)} ops into MS config (added WaitProbe + TriggerProbe)')
