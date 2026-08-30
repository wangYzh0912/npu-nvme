#!/usr/bin/env python3
"""Run the category-aware P8 grid and admit <=30% candidates to P9."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; DRIVER=ROOT/"experiments/benchmarks/p8_p9_incremental.py"

def run(cmd,log):
    proc=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,check=False)
    log.parent.mkdir(parents=True,exist_ok=True); log.write_text(proc.stdout+"\n[stderr]\n"+proc.stderr)
    payload=None
    for line in reversed(proc.stdout.splitlines()):
        try: payload=json.loads(line); break
        except json.JSONDecodeError: pass
    return proc.returncode,payload

def main():
    p=argparse.ArgumentParser(); p.add_argument("--npu",type=int,default=7); p.add_argument("--pci",default="0000:83:00.0"); p.add_argument("--seeds",nargs="+",type=int,default=(41,42,43)); p.add_argument("--full-intervals",nargs="+",type=int,default=(20,50,100)); p.add_argument("--max-ages",nargs="+",type=int,default=(4,8,16)); p.add_argument("--steps",type=int,default=100); p.add_argument("--output-root",type=Path,default=ROOT/"results/ppt-evidence-20260829"); args=p.parse_args()
    strategies=[("model5_m20",.05,.20,"fp16",4,20,4),
                ("model10_m50",.10,.50,"fp16",4,50,8),
                ("model20_m50_vint8",.20,.50,"int8",8,100,16)]
    records=[]; ordinal=0
    for seed in args.seeds:
      for name,mf,mf2,venc,vrefresh,interval,age in strategies:
       if interval not in args.full_intervals or age not in args.max_ages: continue
       for _selected in (0,):
         ordinal+=1; out=args.output_root/"P8_matrix"/f"{ordinal:03d}_{name}_s{seed}_f{interval}_a{age}"
         cmd=[sys.executable,str(DRIVER),"produce","--npu",str(args.npu),"--pci",args.pci,"--seed",str(seed),"--steps",str(args.steps),"--model-fraction",str(mf),"--m-fraction",str(mf2),"--v-encoding",venc,"--v-refresh",str(vrefresh),"--full-interval",str(interval),"--max-age",str(age),"--shm-id",str(12000+ordinal),"--output-root",str(out)]
         rc,payload=run(cmd,out/"produce.log"); record={"name":name,"seed":seed,"full_interval":interval,"max_age":age,"produce_rc":rc,"payload":payload}
         if rc==0 and payload and payload.get("manifest"):
            result_path=Path(payload["manifest"]).parent/"result.json"; result=json.loads(result_path.read_text()); admitted=bool(result.get("gate",{}).get("p9_admitted")); record["p9_admitted"]=admitted
            if admitted:
                recover=[sys.executable,str(DRIVER),"recover","--manifest",payload["manifest"],"--npu",str(args.npu),"--shm-id",str(14000+ordinal),"--output-root",str(out)]; rrc,rpayload=run(recover,out/"recover.log"); record.update({"recover_rc":rrc,"recover":rpayload})
         records.append(record); print(json.dumps(record,sort_keys=True),flush=True)
    summary={"records":records,"produced":sum(r["produce_rc"]==0 for r in records),"admitted":sum(bool(r.get("p9_admitted")) for r in records),"recovered":sum(r.get("recover_rc")==0 for r in records)}; (args.output_root/"P8_P9_matrix.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    if any(r["produce_rc"]!=0 or (r.get("p9_admitted") and r.get("recover_rc")!=0) for r in records): raise SystemExit(1)
if __name__=="__main__": main()
