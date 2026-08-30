#!/usr/bin/env python3
"""Resumable P1--P9 and reliability experiment orchestrator."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]; BENCH=ROOT/"experiments/benchmarks"

def run(argv,timeout=15):
    try:
        p=subprocess.run(argv,cwd=ROOT,text=True,capture_output=True,check=False,timeout=timeout)
        return {"argv":argv,"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
    except BaseException as exc: return {"argv":argv,"returncode":-1,"error":repr(exc)}

def preflight(args):
    driver=Path("/sys/bus/pci/devices/0000:83:00.0/driver")
    info={"timestamp":time.strftime("%FT%T%z"),"euid":os.geteuid(),"uio_nodes":[str(x) for x in Path("/dev").glob("uio*")],"driver_83":driver.resolve().name if driver.exists() else None,"models":run(["findmnt","-no","SOURCE,FSTYPE,OPTIONS","-T","/models"]),"lspci_83":run(["lspci","-s","83:00.0","-nnk"]),"lspci_84":run(["lspci","-s","84:00.0","-nnk"]),"npu_smi":run(["npu-smi","info"]),"sudo_probe":run(["sudo","-n","true"]),"sudo_stat":run(["stat","-c","uid=%u mode=%a","/usr/bin/sudo"]),"hugepages":Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else None,"library":str(ROOT/"build_out/lib/libnpu_nvme.so"),"library_exists":(ROOT/"build_out/lib/libnpu_nvme.so").exists()}
    mount=info["models"].get("stdout",""); blockers=[]
    if os.geteuid()!=0: blockers.append("hardware phases require root EUID")
    if info["sudo_probe"].get("returncode") != 0: blockers.append("sudo cannot elevate in this container")
    if not info["uio_nodes"]: blockers.append("no /dev/uio* device is visible")
    if info["driver_83"]!="uio_pci_generic": blockers.append("83:00.0 is not bound to uio_pci_generic")
    if "ro," in mount or mount.rstrip().endswith(",ro") or " ro," in mount: blockers.append("/models is mounted read-only")
    if not info["library_exists"]: blockers.append("installed libnpu_nvme.so is missing")
    info["blockers"]=blockers; info["status"]="pass" if not blockers else "blocked"; return info

def command_matrix(args):
    py=args.python; out=str(args.output_root)
    return {
      "P1_calibration":[str(ROOT/"scripts/calibrate_same_ssd_readonly.sh"),py,str(args.output_root/"P1"/"calibration.json")],
      "P1":[py,str(BENCH/"p1_fair_io.py"),"--path","all","--npu",str(args.npu)],
      "P1_aggregate":[py,str(BENCH/"p1_aggregate.py"),"--root",str(args.output_root/"P1"),"--output",str(args.output_root/"P1"/"summary.json")],
      "P2":[py,str(BENCH/"p2_stack_decompose.py"),"--npu",str(args.npu)],
      "P3":[py,str(BENCH/"p3_async_pipeline.py"),"--npu",str(args.npu),"--output-root",out],
      "P3_aggregate":[py,str(BENCH/"p3_aggregate.py"),"--root",str(args.output_root/"P3"),"--output",str(args.output_root/"P3"/"summary.json")],
      "P4":[py,str(BENCH/"p4_training_e2e.py"),"--npu",str(args.npu),"--output-root",out],
      "P4_aggregate":[py,str(BENCH/"p4_aggregate.py"),"--root",str(args.output_root/"P4"),"--output",str(args.output_root/"P4"/"summary.json")],
      "P5":[py,str(BENCH/"p5_ring_memory.py"),"--npu",str(args.npu),"--output-root",str(args.output_root/"P5")],
      "P6_profile":[py,str(ROOT/"experiments/microbench/vector_engine_profile.py"),"--model","gpt2_xl","--device-id",str(args.npu),"--seeds","41,42,43","--warmups","10","--steps","30","--output-dir",str(args.output_root/"P6"/"profile")],
      "P6_aux":[py,str(BENCH/"p6_aux_injection.py"),"--npu",str(args.npu),"--output-root",out],
      "P6_analyze":[py,str(BENCH/"p6_analyze_tree.py"),"--root",str(args.output_root/"P6"/"profile")],
      "P7_seed41":[py,str(BENCH/"s2_real_trajectory.py"),"--model","gpt2_xl","--seed","41","--steps","500","--sample-windows","1-30,236-265,471-500","--block-sizes","65536,262144,524288","--npu",str(args.npu),"--output-root",str(args.output_root/"P7"/"seed41")],
      "P7_seed42":[py,str(BENCH/"s2_real_trajectory.py"),"--model","gpt2_xl","--seed","42","--steps","500","--sample-windows","1-30,236-265,471-500","--block-sizes","65536,262144,524288","--npu",str(args.npu),"--output-root",str(args.output_root/"P7"/"seed42")],
      "P7_seed43":[py,str(BENCH/"s2_real_trajectory.py"),"--model","gpt2_xl","--seed","43","--steps","500","--sample-windows","1-30,236-265,471-500","--block-sizes","65536,262144,524288","--npu",str(args.npu),"--output-root",str(args.output_root/"P7"/"seed43")],
      "P7_analyze":[py,str(BENCH/"p7_analyze_tree.py"),"--root",str(args.output_root/"P7")],
      "P8_P9":[py,str(BENCH/"p8_p9_matrix.py"),"--npu",str(args.npu),"--output-root",out],
      "reliability":[py,str(BENCH/"reliability_gate.py"),"--npu",str(args.npu),"--output-root",str(args.output_root/"reliability")],
      "scale13b":[py,str(BENCH/"e5_model_owner_pressure.py"),"--model","gpt2_13b","--npu",str(args.npu),"--producers","1","--output-root",str(args.output_root/"scale13b")],
    }

def main():
    default_python=(Path("/home/user7/miniconda3/envs/ms_2.5/bin/python") if Path("/home/user7/miniconda3/envs/ms_2.5/bin/python").exists() else Path(sys.executable))
    p=argparse.ArgumentParser(); p.add_argument("--phases",nargs="+",default=("P1_calibration","P1","P1_aggregate","P2","P3","P3_aggregate","P4","P4_aggregate","P5","P6_profile","P6_analyze","P6_aux","P7_seed41","P7_seed42","P7_seed43","P7_analyze","P8_P9","reliability")); p.add_argument("--npu",type=int,default=7); p.add_argument("--python",default=str(default_python)); p.add_argument("--output-root",type=Path,default=ROOT/"results/ppt-evidence-20260829"); p.add_argument("--timeout",type=int,default=86400); p.add_argument("--dry-run",action="store_true"); p.add_argument("--allow-blocked",action="store_true"); p.add_argument("--rerun",action="store_true"); args=p.parse_args(); args.output_root.mkdir(parents=True,exist_ok=True)
    pre=preflight(args); (args.output_root/"preflight.json").write_text(json.dumps(pre,indent=2,sort_keys=True)+"\n")
    matrix=command_matrix(args); unknown=set(args.phases)-set(matrix)
    if unknown: raise SystemExit(f"unknown phases: {sorted(unknown)}")
    state_path=args.output_root/"execution_state.json"; state=json.loads(state_path.read_text()) if state_path.exists() else {"phases":{}}
    state["preflight"]=pre; state["requested_phases"]=args.phases
    if pre["blockers"] and not (args.allow_blocked or args.dry_run):
        state["status"]="blocked"; state_path.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n"); print(json.dumps(pre,sort_keys=True)); raise SystemExit(2)
    for name in args.phases:
        if not args.rerun and state["phases"].get(name,{}).get("status")=="pass": continue
        record={"argv":matrix[name],"started":time.strftime("%FT%T%z")}
        if args.dry_run: record.update({"status":"planned","returncode":None})
        else:
            log=args.output_root/"orchestrator_logs"/f"{name}.log"; log.parent.mkdir(parents=True,exist_ok=True)
            with log.open("w") as stream: proc=subprocess.run(matrix[name],cwd=ROOT,text=True,stdout=stream,stderr=subprocess.STDOUT,check=False,timeout=args.timeout)
            record.update({"returncode":proc.returncode,"status":"pass" if proc.returncode==0 else "fail","log":str(log)})
        record["ended"]=time.strftime("%FT%T%z"); state["phases"][name]=record; state_path.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n")
        print(json.dumps({"phase":name,**record},sort_keys=True),flush=True)
        if record["status"]=="fail": state["status"]="fail"; state_path.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n"); raise SystemExit(1)
    state["status"]="planned" if args.dry_run else "pass"; state_path.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
