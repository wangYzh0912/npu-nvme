#!/usr/bin/env python3
"""Aggregate P7 trajectory runs across seeds, block sizes and phases."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def load(path):
    path=Path(path)
    if path.is_dir(): sample_path=path/"samples.jsonl"; config_path=path/"config.json"
    elif path.name=="result.json": sample_path=path.parent/"samples.jsonl"; config_path=path.parent/"config.json"
    else: sample_path=path; config_path=path.parent/"config.json"
    config=json.loads(config_path.read_text()) if config_path.exists() else {}
    rows=[json.loads(line) for line in sample_path.read_text().splitlines() if line.strip()]
    return config,rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--inputs",nargs="+",type=Path,required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args()
    aggregate={}; jaccard={}; category_energy={}; counts={"early":0,"mid":0,"late":0}
    for source in args.inputs:
        config,rows=load(source); seed=str(config.get("seed","unknown"))
        for row in rows:
            step=int(row["step"]); phase="early" if step<=100 else ("mid" if step<=350 else "late"); counts[phase]+=1
            for block_size,data in row.get("block_sizes",{}).items():
                jaccard.setdefault((seed,block_size,phase),[]).append(float(data.get("selected_jaccard",0)))
                for fraction,record in data.get("coverage",{}).items():
                    aggregate.setdefault((seed,block_size,phase,fraction),[]).append(float(record["energy_fraction"]))
                for category,cdata in data.get("categories",{}).items():
                    category_energy.setdefault((seed,block_size,phase,category),[]).append(float(cdata.get("l2",0))**2)
    output={"inputs":[str(x) for x in args.inputs],"phase_counts":counts,
            "coverage":[{"seed":k[0],"block_size":int(k[1]),"phase":k[2],"top_percent":int(k[3]),"mean_energy_fraction":float(np.mean(v))} for k,v in sorted(aggregate.items())],
            "adjacent_jaccard":[{"seed":k[0],"block_size":int(k[1]),"phase":k[2],"mean":float(np.mean(v)),"p95":float(np.percentile(v,95))} for k,v in sorted(jaccard.items())],
            "category_change_energy":[{"seed":k[0],"block_size":int(k[1]),"phase":k[2],"category":k[3],"mean_energy":float(np.mean(v))} for k,v in sorted(category_energy.items())],
            "hypothesis":"energy concentration may coexist with dynamically migrating hot blocks"}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n"); print(json.dumps({"coverage_rows":len(output["coverage"]),"jaccard_rows":len(output["adjacent_jaccard"])},sort_keys=True))
if __name__=="__main__": main()
