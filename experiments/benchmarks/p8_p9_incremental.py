#!/usr/bin/env python3
"""P8 real write-volume producer and P9 fresh-process recovery consumer."""
from __future__ import annotations
import argparse, ctypes, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; sys.path[:0]=[str(ROOT),str(ROOT/"python")]
from category_delta import CategoryAwarePolicy, CategoryConfig, apply_frame
from c_bindings import lib
from direct_checkpoint import DirectCheckpoint
from disk_layout import META_SLOT_BYTES, SUPERBLOCK_HEADER_BYTES
from experiments.benchmarks.r0_real_e2e import build, train_one, control_state
from experiments.benchmarks.s2_real_policy_scan import assign_state
from experiments.benchmarks.s2_real_trajectory import snapshot_state
from ppt_evidence import EvidenceBundle, environment_snapshot, stats, command
from raw_ring import (KIND_DELTA, pack_ring_metadata, pack_ring_slot,
                      select_ab_metadata, unpack_ring_slot)

DELTA_BASE=128*1024**3
DELTA_META_A=DELTA_BASE-8192
DELTA_META_B=DELTA_BASE-4096

def state_digest(state):
    digest=hashlib.sha256()
    for name in sorted(state):
        value=np.ascontiguousarray(state[name]); digest.update(name.encode()); digest.update(value.dtype.str.encode()); digest.update(np.asarray(value.shape,dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()

def state_nrmse(actual,restored):
    num=den=0.0
    for name in actual:
        left=np.asarray(actual[name],dtype=np.float64); right=np.asarray(restored[name],dtype=np.float64); diff=left-right; num+=float(np.vdot(diff,diff)); den+=float(np.vdot(left,left))
    return float(np.sqrt(num/max(den,1e-30)))

def full_physical_bytes(ckpt):
    payload=sum(((int(item["size"])+4095)//4096)*4096
                for item in ckpt.last_layout)
    return payload+META_SLOT_BYTES+SUPERBLOCK_HEADER_BYTES

def chunks(buffer,base,size):
    ptr=ctypes.addressof(buffer); values=[]
    for inner in range(0,len(buffer),size): values.append((ptr+inner,base+inner,min(size,len(buffer)-inner)))
    n=len(values); ps=(ctypes.c_void_p*n)(); os_=(ctypes.c_uint64*n)(); ss=(ctypes.c_size_t*n)()
    for i,(p,o,s) in enumerate(values): ps[i]=p; os_[i]=o; ss[i]=s
    return ps,os_,ss,n

def write_frame(ckpt,frame,offset):
    buf=ctypes.create_string_buffer(frame,len(frame)); arrays=chunks(buf,offset,ckpt.chunk_size)
    start=time.perf_counter_ns(); rc=lib.npu_nvme_write_batch_host(ckpt.ctx,*arrays)
    if rc: raise RuntimeError(f"delta write failed: {rc}")
    ckpt.flush_nvme(); return (time.perf_counter_ns()-start)/1e6

def read_frame(ckpt,size,offset):
    buf=ctypes.create_string_buffer(size); arrays=chunks(buf,offset,ckpt.chunk_size)
    rc=lib.npu_nvme_read_batch_host(ckpt.ctx,*arrays)
    if rc: raise RuntimeError(f"delta read failed: {rc}")
    return bytes(buf.raw[:size])

def configs(args):
    return {"model":CategoryConfig(args.model_fraction,"raw",args.max_age),
            "adam_m":CategoryConfig(args.m_fraction,args.m_encoding,args.max_age),
            "adam_v":CategoryConfig(1.0,args.v_encoding,args.max_age,args.v_refresh),
            "other":CategoryConfig(1.0,"raw",0)}

def producer(args):
    bundle=EvidenceBundle("P8",vars(args),root=args.output_root,repo_root=ROOT,environment=environment_snapshot(pci=args.pci,npu=str(args.npu),repo_root=ROOT,npu_info=command(["npu-smi","info"])))
    args.model=args.model_name; ms,model,opt,cell=build(args); ckpt=None; records=[]
    try:
        keep_last_n=(args.keep_last_n if args.keep_last_n is not None else
                     max(3,args.steps//args.full_interval+2))
        ckpt=DirectCheckpoint(nvme_addr=args.pci,npu_device_id=args.npu,pipeline_depth=args.depth,requested_chunk_size=args.chunk_size,spdk_shm_id=args.shm_id,keep_last_n=keep_last_n,slot_size_gb=args.slot_size_gb)
        existing_steps=[int(name.split("_",1)[1]) for name in
                        ckpt.meta_dict.get("checkpoints",{})
                        if name.startswith("step_")]
        disk_step_base=(max(existing_steps)+100) if existing_steps else 0
        initial=snapshot_state(model,opt,True); full_bytes=sum(v.nbytes for v in initial.values())
        handle=ckpt.save_state({"model":model,"optimizer":opt},control_state(ms,opt,0,args.seed),disk_step_base,meta_path=str(bundle.raw_dir/"base.pkl")); handle.wait(); full_physical=full_physical_bytes(ckpt)
        policy=CategoryAwarePolicy(initial,args.block_size,configs(args)); base_step=0; full_disk_step=disk_step_base
        for step in range(1,args.steps+1):
            loss,train_ms=train_one(cell,ms,step,args.seq_len); current=snapshot_state(model,opt,True)
            periodic_full=step%args.full_interval==0
            if periodic_full:
                full_disk_step=disk_step_base+step
                handle=ckpt.save_state({"model":model,"optimizer":opt},control_state(ms,opt,step,args.seed),full_disk_step,meta_path=str(bundle.raw_dir/f"full_{step}.pkl")); handle.wait(); full_physical=full_physical_bytes(ckpt); policy=CategoryAwarePolicy(current,args.block_size,configs(args)); base_step=step
            pending=policy.observe(current,step); frame,accounting=policy.pack(current)
            envelope=pack_ring_slot(frame,step,step,KIND_DELTA)
            stored=envelope+b"\0"*((4096-len(envelope)%4096)%4096)
            if len(stored)>args.delta_slot_bytes: raise RuntimeError("delta frame exceeds configured slot")
            slot=(step-1)%args.delta_slots; offset=DELTA_BASE+slot*args.delta_slot_bytes
            write_ms=write_frame(ckpt,stored,offset)
            metadata=pack_ring_metadata(step,step,base_step,base_step).ljust(4096,b"\0")
            metadata_ms=write_frame(ckpt,metadata,DELTA_META_A if step%2 else DELTA_META_B)
            policy.ack(pending["generation"])
            decoded=policy.reference()
            actual=len(stored)+4096+(full_physical if periodic_full else 0)
            row={"status":"pass","step":step,"base_step":base_step,"full_disk_step":full_disk_step,"generation":pending["generation"],"slot_generation":step,"offset":offset,"frame_bytes":len(frame),"stored_frame_bytes":len(stored),"metadata_commit_bytes":4096,"metadata_commit_ms":metadata_ms,"periodic_full":periodic_full,"periodic_full_bytes":full_physical if periodic_full else 0,"actual_checkpoint_bytes":actual,"full_state_bytes":full_bytes,"full_physical_bytes":full_physical,"write_ratio":actual/full_physical,"write_ms":write_ms,"train_ms":train_ms,"loss":loss,"decoded_state_sha256":state_digest(decoded),"approximation_nrmse":state_nrmse(current,decoded),**accounting}
            records.append(row); bundle.add_sample(row)
        manifest={"model":args.model_name,"pci":args.pci,"npu":args.npu,"seed":args.seed,"seq_len":args.seq_len,"chunk_size":args.chunk_size,"shm_id":args.shm_id,"keep_last_n":keep_last_n,"disk_step_base":disk_step_base,"records":records,"full_state_bytes":full_bytes,"strategy":{k:vars(v) for k,v in configs(args).items()}}
        (bundle.run_dir/"recovery_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    except BaseException as exc: bundle.add_failure({"error":repr(exc)})
    finally:
        if ckpt: ckpt.cleanup()
    ratios=[r["write_ratio"] for r in records]; actual=sum(r["actual_checkpoint_bytes"] for r in records); result=bundle.finalize(metrics={"model":args.model_name,"seed":args.seed,"mode":"category_delta","state_bytes":records[0]["full_state_bytes"] if records else None,"physical_bytes":actual,"nvme_bytes":actual,"nvme_submitted_bytes":actual,"nvme_bytes_measurement":"aligned host-submission accounting; not controller SMART/NAND media bytes","write_ratio":stats(ratios),"gate":{"p9_admission_max":.30,"final_go_max":.20,"p9_admitted":bool(ratios and np.mean(ratios)<=.30),"final_go":bool(ratios and np.mean(ratios)<.20)}},status="pass" if len(records)==args.steps and not bundle.failures else "fail")
    print(json.dumps({"run_id":result["run_id"],"status":result["status"],"manifest":str(bundle.run_dir/"recovery_manifest.json")},sort_keys=True))

def consumer(args):
    manifest=json.loads(Path(args.manifest).read_text()); args.model=manifest.get("model","gpt2_xl"); args.seed=manifest["seed"]; args.seq_len=manifest["seq_len"]; args.pci=manifest["pci"]; args.npu=manifest["npu"]
    bundle=EvidenceBundle("P9",{"manifest":str(args.manifest),"targets":args.targets,"continue_steps":args.continue_steps},root=args.output_root,repo_root=ROOT,environment=environment_snapshot(pci=args.pci,npu=str(args.npu),repo_root=ROOT,npu_info=command(["npu-smi","info"])))
    ms,model,opt,cell=build(args); ckpt=None; errors=[]
    try:
        ckpt=DirectCheckpoint(nvme_addr=args.pci,npu_device_id=args.npu,pipeline_depth=args.depth,requested_chunk_size=manifest["chunk_size"],spdk_shm_id=args.shm_id,keep_last_n=manifest["keep_last_n"],slot_size_gb=args.slot_size_gb)
        positions=[int(x) for x in args.targets.split(",")] if args.targets else [len(manifest["records"])]
        meta_a=read_frame(ckpt,4096,DELTA_META_A); meta_b=read_frame(ckpt,4096,DELTA_META_B); meta_name,visible=select_ab_metadata(meta_a,meta_b)
        if visible["head"]!=manifest["records"][-1]["step"]: raise AssertionError("on-disk Delta metadata does not publish latest step")
        for target in positions:
            chosen=manifest["records"][:target]; base=max(r["base_step"] for r in chosen); full_disk_step=chosen[-1]["full_disk_step"]; start=time.perf_counter_ns(); ckpt.load_state({"model":model,"optimizer":opt},step=full_disk_step); state=snapshot_state(model,opt,True)
            applicable=[r for r in chosen if r["base_step"]==base and r["step"]>base]
            last_generation=0
            for record in applicable:
                if record["generation"]<=last_generation: raise ValueError("non-monotonic generation")
                raw=read_frame(ckpt,record["stored_frame_bytes"],record["offset"]); envelope=unpack_ring_slot(raw)
                if envelope["slot_generation"]!=record["slot_generation"] or envelope["step_id"]!=record["step"]: raise ValueError("Delta ring envelope identity mismatch")
                state,parsed=apply_frame(state,envelope["frame"]); last_generation=parsed["generation"]
            expected_record=chosen[-1]; reconstructed_hash=state_digest(state)
            if reconstructed_hash!=expected_record["decoded_state_sha256"]: raise AssertionError("reconstructed state hash mismatch")
            assign_state(ms,model,opt,state); ms.hal.synchronize(); recovery_ms=(time.perf_counter_ns()-start)/1e6
            restored=snapshot_state(model,opt,True); num=den=0.0; maximum=0.0
            for name in state:
                diff=np.asarray(restored[name],dtype=np.float64)-np.asarray(state[name],dtype=np.float64); num+=float(np.vdot(diff,diff)); den+=float(np.vdot(np.asarray(state[name],dtype=np.float64),np.asarray(state[name],dtype=np.float64))); maximum=max(maximum,float(np.max(np.abs(diff),initial=0)))
            losses=[]
            target_step=int(expected_record["step"])
            for extra in range(args.continue_steps): losses.append(train_one(cell,ms,target_step+1+extra,args.seq_len)[0])
            source_next=next((r["loss"] for r in manifest["records"] if r["step"]==target_step+1),None)
            loss_deviation=(abs(losses[0]-source_next)/max(abs(source_next),1e-30) if source_next is not None else None)
            row={"status":"pass","target":target,"target_step":target_step,"base_step":base,"full_disk_step":full_disk_step,"delta_count":len(applicable),"visible_metadata_copy":meta_name,"visible_generation":visible["generation"],"assignment_nrmse":float(np.sqrt(num/max(den,1e-30))),"nrmse":expected_record["approximation_nrmse"],"max_error":maximum,"reconstructed_sha256":reconstructed_hash,"recovery_ms":recovery_ms,"continued_losses":losses,"source_next_loss":source_next,"loss_deviation":loss_deviation,"missing_or_stale":False}; errors.append(row); bundle.add_sample(row)
    except BaseException as exc: bundle.add_failure({"error":repr(exc)})
    finally:
        if ckpt: ckpt.cleanup()
    deviations=[r["loss_deviation"] for r in errors if r["loss_deviation"] is not None]
    passed=bool(errors and not bundle.failures and max(r["nrmse"] for r in errors)<=5e-3 and (not deviations or max(deviations)<=.01))
    result=bundle.finalize(metrics={"model":args.model,"seed":args.seed,"mode":"fresh_process_full_plus_delta","recovery_error":stats([r["nrmse"] for r in errors]),"loss_deviation":stats(deviations),"gate":{"nrmse_max":5e-3,"loss_deviation_max":.01,"no_stale":all(not r["missing_or_stale"] for r in errors),"hash_match":True}},status="pass" if passed else "fail")
    print(json.dumps({"run_id":result["run_id"],"status":result["status"]},sort_keys=True))

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="action",required=True)
    common=argparse.ArgumentParser(add_help=False); common.add_argument("--npu",type=int,default=7); common.add_argument("--pci",default="0000:83:00.0"); common.add_argument("--seed",type=int,default=41); common.add_argument("--seq-len",type=int,default=129); common.add_argument("--depth",type=int,default=4); common.add_argument("--chunk-size",type=int,default=4*1024**2); common.add_argument("--shm-id",type=int,default=9800); common.add_argument("--slot-size-gb",type=int,default=10); common.add_argument("--output-root",default=None)
    q=sub.add_parser("produce",parents=[common]); q.add_argument("--model-name",choices=("gpt2","gpt2_xl"),default="gpt2_xl"); q.add_argument("--steps",type=int,default=100); q.add_argument("--block-size",type=int,default=262144); q.add_argument("--model-fraction",type=float,default=.10); q.add_argument("--m-fraction",type=float,default=.20); q.add_argument("--m-encoding",choices=("raw","fp16","int8"),default="fp16"); q.add_argument("--v-encoding",choices=("raw","fp16","int8"),default="fp16"); q.add_argument("--v-refresh",type=int,default=4); q.add_argument("--full-interval",type=int,default=100); q.add_argument("--max-age",type=int,default=8); q.add_argument("--keep-last-n",type=int,default=None, help="override retained FULL metadata records (3 fits the 400 KiB metadata slot)"); q.add_argument("--delta-slots",type=int,default=128); q.add_argument("--delta-slot-bytes",type=int,default=2*1024**3)
    r=sub.add_parser("recover",parents=[common]); r.add_argument("--manifest",type=Path,required=True); r.add_argument("--targets",default="10,100"); r.add_argument("--continue-steps",type=int,default=10)
    args=p.parse_args(); producer(args) if args.action=="produce" else consumer(args)
if __name__=="__main__": main()
