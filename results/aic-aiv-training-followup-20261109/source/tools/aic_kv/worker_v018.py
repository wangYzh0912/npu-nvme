"""Explicit observation adapter for the retained 0.18 CANN 8.5.1 image."""
from .worker import KVWorker,tensor_metadata


class V018Worker(KVWorker):
    def kv_snapshot(self):
        from vllm_ascend.attention.utils import using_paged_attention
        from vllm_ascend.utils import get_ascend_config
        runner=self.model_runner
        layers=[]
        for name,layer in runner.compilation_config.static_forward_context.items():
            impl=getattr(layer,'impl',None)
            if impl is None or not hasattr(impl,'key_cache'):continue
            if impl.key_cache is None:raise RuntimeError('warmup did not bind cache')
            layers.append(dict(layer=name,impl=type(impl).__name__,
                key=tensor_metadata(impl.key_cache),value=tensor_metadata(impl.value_cache),
                num_heads=impl.num_heads,num_kv_heads=impl.num_kv_heads,head_size=impl.head_size,scale=impl.scale))
        if not layers:raise RuntimeError('no attention cache observed')
        return dict(rank=self.rank,device=str(self.device),
            graph_mode=str(runner.vllm_config.compilation_config.cudagraph_mode),
            resolved_layout='v018_native_separate_k_v',num_blocks=runner.kv_cache_config.num_blocks,
            kernel_block_sizes=runner.kernel_block_sizes,layers=layers,
            pa_shapes=get_ascend_config().pa_shape_list,
            pa_eligible={str(n):using_paged_attention(n,runner.vllm_config) for n in (1,8)})
