"""Experiment-only worker; metadata hooks are restored before timing."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
import torch_npu
from aic_activity.worker import ActivityWorker
from .geometry import dense_cache_views


def tensor_metadata(tensor):
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                dtype=str(tensor.dtype), element_size=tensor.element_size(),
                logical_bytes=tensor.numel() * tensor.element_size(),
                storage_offset=tensor.storage_offset(), data_ptr=tensor.data_ptr(),
                storage_ptr=tensor.untyped_storage().data_ptr(),
                storage_bytes=tensor.untyped_storage().nbytes(),
                contiguous=tensor.is_contiguous())


class KVWorker(ActivityWorker):
    def init_device(self):
        super().init_device()
        if os.environ.get('AIC_KV_LAYOUT','native')=='dense':
            original=self.model_runner._reshape_kv_cache_tensors
            @functools.wraps(original)
            def reshape(*args,**kwargs):
                caches=original(*args,**kwargs)
                for name,cache in caches.items():
                    if not isinstance(cache,tuple) or len(cache)!=2:
                        raise ValueError('dense control only supports ordinary two-tensor attention caches')
                    key,value=cache
                    # Allocation is still zero-initialized and unbound here.
                    # Reinterpret the same backing as [all K][all V], avoiding
                    # an extra cache allocation or a stale shadow copy.
                    caches[name]=dense_cache_views(key,value)
                return caches
            self.model_runner._reshape_kv_cache_tensors=reshape

    def kv_snapshot(self):
        runner = self.model_runner
        layers = []
        for name, layer in runner.compilation_config.static_forward_context.items():
            impl = getattr(layer, 'impl', None)
            if impl is None or not hasattr(impl, 'key_cache'):
                continue
            if impl.key_cache is None:
                raise RuntimeError('KV snapshot requires a completed warmup')
            layers.append(dict(layer=name, impl=type(impl).__name__,
                               key=tensor_metadata(impl.key_cache),
                               value=tensor_metadata(impl.value_cache),
                               use_bnsd_kv_cache=impl.use_bnsd_kv_cache,
                               num_heads=impl.num_heads, num_kv_heads=impl.num_kv_heads,
                               head_size=impl.head_size, scale=impl.scale))
        if not layers:
            raise RuntimeError('no attention cache layers observed')
        from vllm_ascend.attention.utils import using_paged_attention
        from vllm_ascend.utils import get_ascend_config
        return dict(rank=self.rank, device=str(self.device),
                    graph_mode=str(runner.vllm_config.compilation_config.cudagraph_mode),
                    resolved_layout=str(runner.vllm_config.cache_config.get_resolved_kv_cache_layout()),
                    num_blocks=runner.kv_cache_config.num_blocks,
                    experiment_layout=os.environ.get('AIC_KV_LAYOUT','native'),
                    pa_shapes=get_ascend_config().pa_shape_list,
                    pa_eligible={str(n):using_paged_attention(n,runner.vllm_config,layers[0]['head_size']) for n in (1,8)},
                    kernel_block_sizes=runner.kernel_block_sizes, layers=layers)

    def kv_probe_start(self):
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        cls = AscendAttentionBackendImpl
        self.kv_probe_records = []
        self.kv_probe_originals = {}
        self.kv_probe_step = -1
        self.kv_probe_block_tables = []
        self.kv_probe_execute = self.model_runner.execute_model
        @functools.wraps(self.kv_probe_execute)
        def execute(*args, **kwargs):
            self.kv_probe_step += 1
            return self.kv_probe_execute(*args, **kwargs)
        self.model_runner.execute_model = execute
        self.kv_probe_update_states=self.model_runner._update_states
        @functools.wraps(self.kv_probe_update_states)
        def update_states(*args,**kwargs):
            result=self.kv_probe_update_states(*args,**kwargs)
            batch=self.model_runner.input_batch
            num_reqs=len(batch.req_id_to_index)
            groups=[]
            for table in batch.block_table.block_tables:
                array=table.get_numpy_array()[:num_reqs]
                counts=table.num_blocks_per_row[:num_reqs].tolist()
                active=[array[i,:count].tolist() for i,count in enumerate(counts)]
                groups.append(dict(shape=list(array.shape),sha256=hashlib.sha256(json.dumps(active,separators=(',',':')).encode()).hexdigest(),
                    num_blocks_per_row=counts,block_ids=active))
            self.kv_probe_block_tables.append(dict(step=self.kv_probe_step,groups=groups,num_reqs=num_reqs,
                observation_point='after_update_states_before_attention'))
            return result
        self.model_runner._update_states=update_states
        self.kv_probe_fia = torch_npu.npu_fused_infer_attention_score
        @functools.wraps(self.kv_probe_fia)
        def fia(*args, **kwargs):
            if len(self.kv_probe_records)<512:
                self.kv_probe_records.append(dict(method='torch_npu.npu_fused_infer_attention_score',
                    step=self.kv_probe_step,
                    positional_count=len(args), parameters={key:tensor_metadata(value) if isinstance(value,torch.Tensor)
                        else value for key,value in kwargs.items()}))
            return self.kv_probe_fia(*args, **kwargs)
        torch_npu.npu_fused_infer_attention_score = fia
        for method in ('forward_fused_infer_attention', 'forward_paged_attention', 'reshape_and_cache', 'do_kv_cache_update'):
            original = getattr(cls, method)
            self.kv_probe_originals[method] = original
            def make_wrapper(original, method):
                @functools.wraps(original)
                def wrapper(impl, *args, **kwargs):
                    # Bounded metadata sample: first layer only, no tensor reads.
                    layer = getattr(impl, '_layer_name', None)
                    if len(self.kv_probe_records) < 256 and layer and '.0.' in layer:
                        bound = inspect.signature(original).bind(impl, *args, **kwargs)
                        values = bound.arguments
                        metadata = values.get('attn_metadata')
                        tensors = {key: tensor_metadata(values[key]) for key in ('query', 'key', 'value')
                                   if isinstance(values.get(key), torch.Tensor)}
                        self.kv_probe_records.append(dict(method=method, layer=layer,step=self.kv_probe_step,
                            attn_state=str(getattr(metadata, 'attn_state', None)),
                            seq_lens_list=getattr(metadata, 'seq_lens_list', None),
                            tensors=tensors,
                            block_table=tensor_metadata(metadata.block_tables) if metadata is not None else None))
                    return original(impl, *args, **kwargs)
                return wrapper
            setattr(cls, method, make_wrapper(original, method))
        return dict(rank=self.rank, enabled=True)

    def kv_probe_stop(self, output):
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl
        for name, original in self.kv_probe_originals.items():
            setattr(AscendAttentionBackendImpl, name, original)
        self.kv_probe_originals = {}
        self.model_runner.execute_model = self.kv_probe_execute
        self.model_runner._update_states=self.kv_probe_update_states
        torch_npu.npu_fused_infer_attention_score = self.kv_probe_fia
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        (out / f'kv-probe-rank{self.rank}.json').write_text(json.dumps(self.kv_probe_records, indent=2) + '\n')
        (out / f'block-table-probe-rank{self.rank}.json').write_text(json.dumps(self.kv_probe_block_tables,indent=2)+'\n')
        return dict(rank=self.rank, observations=len(self.kv_probe_records))

    def kv_source_identity(self):
        import torch_npu
        import vllm_ascend.attention.attention_v1 as attention
        import vllm_ascend.worker.model_runner_v1 as runner
        import vllm_ascend.attention.utils as attention_utils
        import vllm.config.cache as cache_config
        files = {}
        for module in (attention, runner, attention_utils, cache_config):
            path = Path(inspect.getsourcefile(module))
            files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return dict(rank=self.rank, files=files, torch=str(torch.__version__), torch_npu=str(torch_npu.__version__))
