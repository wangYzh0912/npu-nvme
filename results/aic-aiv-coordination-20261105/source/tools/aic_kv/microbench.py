"""Controlled FIA replay: strided inputs, dense inputs, and charged conversion."""
import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--metadata',type=Path,default=Path('/output/kv-20261103/metadata-a2/b1-l128-o32/kv-layout.json'))
    p.add_argument('--native-probe',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    import torch_npu
    torch.npu.set_device(0);torch.manual_seed(42)
    snapshots=json.loads(a.metadata.read_text());layer=snapshots[0]['layers'][0]
    observations=json.loads(a.native_probe.read_text())
    native=next(r['parameters'] for r in observations if r['method']=='torch_npu.npu_fused_infer_attention_score'
                and r['parameters'].get('actual_seq_lengths_kv')==[129])
    if native['input_layout']!='TND' or native['num_heads']!=layer['num_heads']:
        raise ValueError('FIA replay contract differs from native metadata probe')
    shape=layer['key']['shape'];stride=layer['key']['stride']
    if len(shape)!=4 or stride[0]!=2*shape[1]*shape[2]*shape[3]:
        raise ValueError('measured block-interleaved K/V geometry differs from replay contract')
    full=shape[0];block=shape[1];heads=layer['num_heads'];kv_heads=layer['num_kv_heads'];head=layer['head_size']
    results=[]
    for blocks in (full//4,full//2,full):
        torch.manual_seed(42)
        backing=torch.zeros((blocks,2,block,kv_heads,head),device='npu',dtype=torch.bfloat16)
        k=backing[:,0].view(blocks,block,-1);v=backing[:,1].view(blocks,block,-1)
        k[0:2].normal_();v[0:2].normal_()
        q=torch.randn((1,heads,head),device='npu',dtype=torch.bfloat16)
        table=torch.arange(native['block_table']['shape'][1],device='npu',dtype=torch.int32).view(1,-1)
        mask_info=native.get('atten_mask')
        if mask_info is not None and mask_info['dtype']!='torch.int8':
            raise ValueError('native attention mask dtype differs from replay contract')
        # Match get_splitfuse_attn_mask: upper-triangular, int8 on device.
        mask=None if mask_info is None else torch.triu(torch.ones(mask_info['shape']),diagonal=1).to(torch.int8).to('npu')
        kwargs=dict(block_table=table,input_layout='TND',block_size=block,
                    actual_seq_lengths=[1],actual_seq_lengths_kv=[block+1],
                    num_key_value_heads=kv_heads,num_heads=heads,scale=layer['scale'],
                    sparse_mode=native['sparse_mode'],atten_mask=mask)
        def attention(key,value):
            return torch_npu.npu_fused_infer_attention_score(query=q,key=key,value=value,**kwargs)[0]
        dense_k=k.contiguous();dense_v=v.contiguous()
        reference=attention(dense_k,dense_v)
        observed=attention(k,v)
        torch.npu.synchronize()
        if not torch.isfinite(observed).all().item() or not torch.allclose(observed,reference,rtol=.01,atol=.01):
            raise RuntimeError('strided vs dense FIA output mismatch')
        max_error=(observed.float()-reference.float()).abs().max().item()
        # Change an active cache location, then remake dense inputs before reuse.
        k[1,0].fill_(.25);v[1,0].fill_(8)
        updated=attention(k,v)
        dense_k=k.contiguous();dense_v=v.contiguous()
        updated_reference=attention(dense_k,dense_v)
        if not torch.equal(dense_k[1,0],k[1,0]) or not torch.equal(dense_v[1,0],v[1,0]):
            raise RuntimeError('dense conversion lost an active KV update')
        if not torch.allclose(updated,updated_reference,rtol=.01,atol=.01):
            raise RuntimeError('cache update visibility differs across layouts')
        update_change=(updated.float()-observed.float()).abs().max().item()
        if update_change==0:
            raise RuntimeError('active KV update did not change attention output')
        def convert():return (k.contiguous(),v.contiguous())
        def charged():
            key,value=convert()
            return attention(key,value)
        functions={'strided_fia':lambda:attention(k,v),'dense_fia':lambda:attention(dense_k,dense_v),
                   'conversion_only':convert,'conversion_plus_fia':charged}
        for mode,fn in functions.items():
            for _ in range(5):result=fn()
            torch.npu.synchronize()
            times=[]
            for rep in range(3):
                begin=time.perf_counter_ns()
                for _ in range(30):result=fn()
                torch.npu.synchronize()
                times.append((time.perf_counter_ns()-begin)/30/1e6)
            results.append(dict(blocks=blocks,mode=mode,mean_batch_ms=times,median_ms=statistics.median(times),
                                output_max_abs_error=max_error,cache_update_matches=True,
                                active_update_output_change=update_change))
        # One short trace per layout verifies which path emits Contiguous/Slice.
        config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,data_simplification=False)
        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                experimental_config=config,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.output/f'trace-n{blocks}'))) as profiler:
            for mode,fn in functions.items():
                with torch.profiler.record_function('KV_MICRO::'+mode):
                    for _ in range(3):result=fn()
                    torch.npu.synchronize()
        (a.output/'results.json').write_text(json.dumps(dict(results=results,layer=layer,
            semantics='synthetic finite KV values, measured layout/heads; dense copies refreshed after cache writes',
            repetitions='three timing batches within one process, not independent model runs'),indent=2)+'\n')
        del backing,k,v,dense_k,dense_v
        torch.npu.empty_cache()


if __name__=='__main__':main()
