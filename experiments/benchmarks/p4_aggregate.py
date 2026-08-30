#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); rows=[]
 for path in a.root.rglob("result.json"):
  r=json.loads(path.read_text())
  config_path=path.parent/"config.json"
  config=json.loads(config_path.read_text()) if config_path.exists() else {}
  if r.get("model")=="gpt2_xl" and r.get("mode") in ("none","sync","queue","async"):
   r={**r,"checkpoint_interval":r.get("checkpoint_interval",config.get("checkpoint_interval")),"seed":r.get("seed",config.get("seed"))}
   rows.append(r)
 baseline={r.get("seed"):r for r in rows if r["mode"]=="none"}; out=[]
 for r in rows:
  b=baseline.get(r.get("seed")); base_tp=b.get("training_throughput_steps_s") if b else None; throughput=r.get("training_throughput_steps_s"); overhead=r.get("step_overhead"); accepted=bool(r.get("restore_verified") and overhead is not None and overhead<=0.05); out.append({"seed":r.get("seed"),"mode":r["mode"],"interval":r.get("checkpoint_interval"),"throughput_steps_s":throughput,"throughput_overhead":1-throughput/base_tp if throughput and base_tp else None,"step_overhead_ratio":overhead,"step_overhead_percent":overhead*100 if overhead is not None else None,"acceptance_status":"pass" if accepted else "fail","checkpoint_step_ms":r.get("checkpoint_step_ms"),"foreground_wait":r.get("foreground_wait"),"restore_verified":r.get("restore_verified")})
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps({"rows":out},indent=2,sort_keys=True)+"\n")
if __name__=="__main__": main()
