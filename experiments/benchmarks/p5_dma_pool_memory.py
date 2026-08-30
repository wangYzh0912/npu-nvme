#!/usr/bin/env python3
"""P5 isolated host-DMA ring memory and wait measurement."""
import argparse, csv, ctypes, json, mmap, re, resource, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT/"python"))
from c_bindings import NPUNVMEContext, lib
from ppt_evidence import EvidenceBundle, environment_snapshot, stats, command

def memory():
    text=Path("/proc/self/status").read_text(); mem=Path("/proc/meminfo").read_text()
    def field(source,name):
        match=re.search(rf"^{name}:\s+(\d+)\s+kB$",source,re.MULTILINE); return int(match.group(1))*1024 if match else None
    return {"rss":field(text,"VmRSS"),"rss_anon":field(text,"RssAnon"),"vm_pin":field(text,"VmPin"),"hugepages_free":int(re.search(r"^HugePages_Free:\s+(\d+)$",mem,re.MULTILINE).group(1))}

def slot_wait(path):
    try:
        with path.open(newline="") as stream: return sum(float(row.get("slot_wait_us",0) or 0) for row in csv.DictReader(stream))/1000.0
    except OSError: return None

def main():
    p=argparse.ArgumentParser(); p.add_argument("--slots",type=int,required=True); p.add_argument("--chunk-size",type=int,required=True); p.add_argument("--total-bytes",type=int,default=1024**3); p.add_argument("--warmups",type=int,default=10); p.add_argument("--samples",type=int,default=30); p.add_argument("--npu",type=int,default=7); p.add_argument("--pci",default="0000:83:00.0"); p.add_argument("--shm-id",type=int,default=16000); p.add_argument("--output-root",default=None); args=p.parse_args()
    bundle=EvidenceBundle("P5",vars(args),root=args.output_root,repo_root=ROOT,environment=environment_snapshot(pci=args.pci,npu=str(args.npu),repo_root=ROOT,npu_info=command(["npu-smi","info"])))
    baseline=memory(); ctx=ctypes.POINTER(NPUNVMEContext)(); payload=None; values=[]; waits=[]
    try:
        rc=lib.npu_nvme_init(ctypes.byref(ctx),args.pci.encode(),args.npu,args.slots,args.chunk_size,True,str(bundle.raw_dir).encode());
        if rc: raise RuntimeError(f"init failed: {rc}")
        after_init=memory(); payload=mmap.mmap(-1,args.total_bytes,access=mmap.ACCESS_WRITE); address=ctypes.addressof(ctypes.c_char.from_buffer(payload)); count=(args.total_bytes+args.chunk_size-1)//args.chunk_size; ptrs=(ctypes.c_void_p*count)(); offsets=(ctypes.c_uint64*count)(); sizes=(ctypes.c_size_t*count)()
        for i in range(count): ptrs[i]=address+i*args.chunk_size; offsets[i]=64*1024**3+i*args.chunk_size; sizes[i]=min(args.chunk_size,args.total_bytes-i*args.chunk_size)
        peak=after_init
        for i in range(args.warmups+args.samples):
            timeline_path=bundle.raw_dir/"time_write.csv"
            timeline_path.unlink(missing_ok=True)
            start=time.perf_counter_ns(); rc=lib.npu_nvme_write_batch_host(ctx,ptrs,offsets,sizes,count); elapsed=(time.perf_counter_ns()-start)/1e6
            if rc: raise RuntimeError(f"write failed: {rc}")
            wait_ms=slot_wait(timeline_path); current=memory(); peak={k:(max(peak[k],current[k]) if peak[k] is not None and current[k] is not None else peak[k] or current[k]) for k in peak}
            if i>=args.warmups: values.append(elapsed); waits.append(wait_ms if wait_ms is not None else 0.0); bundle.add_sample({"status":"pass","sample":i-args.warmups,"latency_ms":elapsed,"slot_wait_ms":wait_ms,"memory":current})
    except BaseException as exc: bundle.add_failure({"error":repr(exc)}); after_init=memory(); peak=after_init
    finally:
        if ctx: lib.npu_nvme_cleanup(ctx)
        if payload: payload.close()
    hugepage_bytes=max(0,(baseline["hugepages_free"]-after_init["hugepages_free"])*2*1024**2)
    result=bundle.finalize(metrics={"mode":"dma_ring","chunk_size":args.chunk_size,"slot_count":args.slots,"pipeline_depth":args.slots,"latency_mean":stats(values),"foreground_wait":stats(waits),"host_rss_peak":peak.get("rss"),"pinned_dram_peak":hugepage_bytes,"baseline_rss":baseline.get("rss"),"incremental_rss":(after_init.get("rss")-baseline.get("rss") if after_init.get("rss") is not None and baseline.get("rss") is not None else None),"expected_pool_bytes":args.slots*args.chunk_size,"memory_note":"hugetlb allocations may not appear in VmRSS; HugePages_Free delta is recorded separately"},status="pass" if len(values)==args.samples and not bundle.failures else "fail")
    print(json.dumps({"run_id":result["run_id"],"status":result["status"]},sort_keys=True))
if __name__=="__main__": main()
