#!/usr/bin/env python3
"""Join matching P1 path runs and calculate acceleration ratios."""
import argparse, json
from pathlib import Path
def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); args=p.parse_args(); rows=[]
    for path in args.root.rglob("result.json"):
        value=json.loads(path.read_text())
        if value.get("mode") not in ("buffered","odirect","spdk_host") or value.get("status")!="pass": continue
        rows.append({**value,"path":str(path)})
    groups={}
    for row in rows:
        modes=groups.setdefault((row.get("operation"),row.get("chunk_size"),row.get("pipeline_depth")),{})
        previous=modes.get(row["mode"])
        if previous is None or row.get("run_id","") > previous.get("run_id",""):
            modes[row["mode"]]=row
    joined=[]
    for key,modes in sorted(groups.items(),key=lambda x:str(x[0])):
        spdk_command_size=modes.get("spdk_host",{}).get("spdk_command_size")
        record={"operation":key[0],"block_size":key[1],"queue_depth":key[2],"physical_request_size_equivalent":not spdk_command_size or spdk_command_size==key[1],"paths":{name:{"run_id":value.get("run_id"),"latency_mean":value.get("latency_mean"),"latency_p50":value.get("latency_p50"),"latency_p95":value.get("latency_p95"),"throughput":value.get("throughput"),"cpu_seconds":value.get("cpu_seconds"),"spdk_command_size":value.get("spdk_command_size")} for name,value in modes.items()}}
        if "spdk_host" in modes:
            spdk=modes["spdk_host"].get("latency_mean")
            record["speedup"]={name:(value.get("latency_mean")/spdk if spdk and value.get("latency_mean") else None) for name,value in modes.items() if name!="spdk_host"}
        joined.append(record)
    output={"comparison":"same operation, logical bytes, configured depth and persistence boundary; physical request size differs when SPDK segments a logical block","groups":joined}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n"); print(json.dumps({"groups":len(joined)},sort_keys=True))
if __name__=="__main__": main()
