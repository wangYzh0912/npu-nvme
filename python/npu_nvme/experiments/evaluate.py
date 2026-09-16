"""Fixed-batch forward-only comparison during canonical validation runs."""
import ctypes
import json
from pathlib import Path

import numpy as np


def compare(controller, network):
    from mindformers.wrapper.wrapper import get_real_models
    from qwen_native_state import capture_control, restore_control
    ms=controller.ms
    if not hasattr(controller,'evaluation_model'):
        from mindspore.parallel.auto_parallel import AutoParallel
        # Restoring a local TP tensor with set_data resets MindSpore's sliced
        # marker. Its bytes are already local; compiling another graph must
        # not slice them a second time. Verified by a two-rank hardware probe.
        for parameter in controller.parameters.values():
            parameter.sliced=True
        model=AutoParallel(get_real_models(network),parallel_mode='semi_auto')
        model.no_init_parameters_in_compile()
        model.dataset_strategy('full_batch')
        controller.evaluation_model=model
    model=controller.evaluation_model
    ids=np.load(Path(controller.output)/f'rank_{controller.rank}'/'input_ids.npy',allow_pickle=False)
    length=len(ids)-1
    batch=dict(input_ids=ms.Tensor(ids[:-1][None,:]),labels=ms.Tensor(ids[1:][None,:]),
        loss_mask=ms.Tensor(np.ones((1,length),np.float32)),
        position_ids=ms.Tensor(np.arange(length,dtype=np.int32)[None,:]),
        attention_mask=ms.Tensor(np.triu(np.ones((1,1,length,length),np.uint8),k=1)))
    state=capture_control(ms,step=0,data_sha256='',lr_horizon=20,small={})
    def loss():
        values=model(**batch)
        values=values if isinstance(values,(tuple,list)) else (values,)
        scalars=[float(np.asarray(value.asnumpy()).mean()) for value in values]
        if len(scalars)==5:return scalars[0]/max(scalars[1],1e-12)
        if len(scalars) not in (1,3):raise ValueError('unknown evaluation loss output contract: '+str(len(scalars)))
        return scalars[0]
    # Compile the forward path before changing storage addresses, then refresh
    # pointers: graph compilation is allowed to alter allocation ownership.
    original={name:parameter.asnumpy().copy() for name,parameter in controller.parameters.items()}
    if not hasattr(controller,'evaluation_aliases'):
        from mindspore.parallel import _utils as parallel_utils
        original_slice=parallel_utils._slice_parameter
        aliases={name:[parameter] for name,parameter in controller.parameters.items()}
        diagnostics=[]
        def preserve_local_slice(parameter,phase,layout):
            name=parameter.name
            diagnostics.append(dict(name=name,matched=name in original,shape=list(parameter.shape),sliced_before=parameter.sliced))
            if name in original:
                candidates=[parameter]
                if parameter.inited_param is not None:candidates.append(parameter.inited_param)
                for candidate in candidates:
                    if tuple(candidate.shape)!=original[name].shape:
                        raise ValueError('evaluation compilation changed TP geometry: '+name)
                    if not np.array_equal(candidate.asnumpy(),original[name]):
                        raise ValueError('evaluation compilation changed TP content: '+name)
                    diagnostics.append(dict(name=name,python_id=id(candidate),sliced_before=candidate.sliced,
                                            shape=list(candidate.shape),is_registered=candidate is controller.parameters[name]))
                    candidate.sliced=True
                    if all(candidate is not other for other in aliases[name]):aliases[name].append(candidate)
            return original_slice(parameter,phase,layout)
        parallel_utils._slice_parameter=preserve_local_slice
        try:true_loss=loss()
        finally:
            parallel_utils._slice_parameter=original_slice
            (Path(controller.output)/f'rank_{controller.rank}'/'evaluation-compile-aliases.json').write_text(json.dumps(diagnostics)+'\n')
        controller.evaluation_aliases=aliases
    else:true_loss=None
    for name,array in original.items():
        if not np.array_equal(controller.parameters[name].asnumpy(),array):raise ValueError('forward evaluation changed training weights: '+name)
    from npu_nvme.framework.parameters import get_dev_ptr
    ms.runtime.synchronize()
    controller.pointers={name:get_dev_ptr(parameter,expected_device=controller.rank)
                         for name,parameter in controller.parameters.items()}
    targets={name:sorted({get_dev_ptr(parameter,expected_device=controller.rank)
                         for parameter in parameters})
             for name,parameters in controller.evaluation_aliases.items()}
    pinned=ctypes.c_void_p()
    if controller.acl.aclrtMallocHost(ctypes.byref(pinned),controller.staging_bytes):raise MemoryError("evaluation pinned staging")
    def copy(pointer,raw):
        for offset in range(0,raw.nbytes,controller.staging_bytes):
            size=min(controller.staging_bytes,raw.nbytes-offset)
            ctypes.memmove(pinned,raw.ctypes.data+offset,size)
            rc=controller.acl.aclrtMemcpyAsync(ctypes.c_void_p(pointer+offset),size,pinned,size,1,controller.stream)
            if rc or controller.acl.aclrtSynchronizeStream(controller.stream):
                if controller.acl.aclrtSynchronizeStream(controller.stream):
                    controller.failed=True;controller.retained_evaluation=(raw,pinned)
                    marker=Path(controller.output)/f'incremental-rank-{controller.rank}-status.json'
                    marker.write_text(json.dumps(dict(status='retained',rank=controller.rank,
                        reason='evaluation DMA stream stop not proven'))+'\n')
                raise RuntimeError('evaluation weight copy failed')
    try:
        if true_loss is None:
            for name,array in original.items():
                for pointer in targets[name]:
                    if pointer!=controller.pointers[name]:copy(pointer,array)
            restore_control(ms,network,state)
            true_loss=loss()
        for name,array in original.items():
            if not np.array_equal(controller.parameters[name].asnumpy(),array):
                raise ValueError('forward evaluation changed training weights: '+name)
        for name in controller.parameters:
            for pointer in targets[name]:copy(pointer,controller.media_shadow(name))
        restore_control(ms,network,state)
        shadow_loss=loss()
    finally:
        if not controller.failed:
            for name,array in original.items():
                for pointer in targets[name]:copy(pointer,array)
            restore_control(ms,network,state)
            ms.runtime.synchronize()
            for name,array in original.items():
                if not np.array_equal(controller.parameters[name].asnumpy(),array):
                    raise ValueError('evaluation changed training weights: '+name)
            if controller.acl.aclrtFreeHost(pinned):raise RuntimeError('evaluation pinned buffer release failed')
    if not np.isfinite(true_loss) or not np.isfinite(shadow_loss):raise ValueError('nonfinite evaluation loss')
    return dict(full_loss=true_loss,shadow_loss=shadow_loss,loss_difference=shadow_loss-true_loss,
                mode='same fixed batch; forward only; training mode with zero dropout')
