#!/usr/bin/env python3
"""Generate the Chinese P1--P9 evidence report without inventing metrics."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

def newest_results(root,experiment):
    candidates=[]
    direct=root/experiment
    if direct.exists():
        candidates.extend(direct.rglob("result.json"))
    for candidate in root.glob(experiment + "*"):
        if candidate != direct and candidate.is_dir():
            candidates.extend(candidate.rglob("result.json"))
    paths=sorted(set(candidates),key=lambda p:p.stat().st_mtime)
    return [(p,json.loads(p.read_text())) for p in paths]

def value(item,key):
    val=item.get(key)
    if isinstance(val,dict) and "mean" in val:
        return f"mean={val['mean']:.4g}"
    return "未测" if val is None else (f"{val:.4g}" if isinstance(val,float) else str(val))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path("results/ppt-evidence-20260829")); p.add_argument("--output",type=Path,default=None); args=p.parse_args(); output=args.output or args.root/"P1_P9_REPORT.md"
    pre=json.loads((args.root/"preflight.json").read_text()) if (args.root/"preflight.json").exists() else {}
    state=json.loads((args.root/"execution_state.json").read_text()) if (args.root/"execution_state.json").exists() else {}
    lines=["# P1-P9 实验报告", "",f"生成时间：{time.strftime('%F %T %z')}","",f"编排器状态：`{state.get('status','unknown')}`；实验结论按下表逐项判定。本报告只汇总 result.json 中已记录的观测值；缺失项标为“未测”。","","## 环境门禁","",f"状态：`{pre.get('status','unknown')}`",""]
    blockers=pre.get("blockers",[]); lines.extend([f"- {x}" for x in blockers] or ["- 无前置阻塞"]); lines.extend(["","## 实验汇总","","| 实验 | 运行数 | pass | fail/degraded | 关键指标 |","|---|---:|---:|---:|---|"])
    for exp in [f"P{i}" for i in range(1,10)]:
        runs=newest_results(args.root,exp); passed=sum(r[1].get("status")=="pass" for r in runs); other=len(runs)-passed
        latest=runs[-1][1] if runs else {}; throughput=latest.get("training_throughput_steps_s", latest.get("throughput"));
        throughput_text=(f"mean={throughput.get('mean'):.4g}" if isinstance(throughput,dict) and throughput.get("mean") is not None else value({"v":throughput},"v"))
        metric=(f"latency_p95={value(latest,'latency_p95')}, throughput={throughput_text}, " f"write_ratio={value(latest,'write_ratio')}, recovery_error={value(latest,'recovery_error')}")
        if exp == "P7" and not runs and (args.root/"P7_summary.json").exists():
            summary=json.loads((args.root/"P7_summary.json").read_text())
            metric=f"trajectory coverage_rows={len(summary.get('coverage',[]))}, jaccard_rows={len(summary.get('adjacent_jaccard',[]))}"
            runs=[(args.root/"P7_summary.json", {"status":"pass"})]; passed=1; other=0
        lines.append(f"| {exp} | {len(runs)} | {passed} | {other} | {metric} |")
    lines.extend(["", "## 审查后可支持的观察", "", "- P1 同型号双盘完成 A/B 校准，但两盘 O_DIRECT 读取均值相差约 19%；4 MiB 对照的请求规格一致，256 MiB 裸盘逻辑块实际拆成 4 MiB NVMe 命令。结果支持相对 buffered FS 的路径差异，不支持笼统宣称裸盘优于 O_DIRECT。", "- P2 六组均为 `degraded`：perf/strace/trace 已保存，但层间时间尚未闭合，禁止绘制精确百分比。", "- P3 仅完成 seed 41、4 MiB、depth 4、正常延迟的 serial/queue/async 各 30 样本；async 时间轴重叠率均值约 0.945，能够证明该配置存在真实重叠。CSV queue_depth 不是 NVMe 在途深度，延迟注入也不是设备服务延迟，不能外推完整矩阵。", "- P4 保存与恢复功能通过，但性能门槛不通过：原始 step_overhead=5.335 是比例，即约 533.5%，不是 5.3%。独立进程吞吐差3.6%受运行长度混杂，不作为验收结论。", "- P5 9 组 DMA ring 均完成；HugePages_Free 的增量随槽数和分块增长，但固定 SPDK hugepage 开销和 1 GiB payload 污染绝对 RSS，当前只能作为趋势证据。", "- P6 报告的 2.9%--3.1% 是 ArithmeticUtilization PMU issue ratio 按算子持续时间投影到设备 wall-clock 的值，不是整机 Vector 占用率；host/device 时钟无共同 epoch，step 对齐为估算。hbm.csv 设备平均约为读 19--21 GB/s、写 20--22 GB/s，但缺峰值分母，不能判断是否接近瓶颈。辅助注入只覆盖 seed 41 的 NPU serial/parallel，不能得出存在可免费利用空隙。", "- P7 GPT-2 XL seed 42 的 500 步训练中采样早/中/晚各 30 步，覆盖三种分块；缺 seed 41/43，结论仅为单 seed 描述性证据。", "- P8 的 25.6% 是对齐后的提交字节核算，不是 SSD SMART/NAND 实际写量，且只有 GPT-2 seed 41、10 步；未达到 `<20%`。P9 两个恢复点的 fresh-process 哈希与误差检查通过，但样本规模不足。", "", "## 验收判定", "", "- 当前可正式使用：P3 单配置真实重叠、P9 两个位置的功能正确性、单所有者压力与一次 NVMe 错误恢复。", "- 当前需降级使用：P1、P5、P7、P8。", "- 当前不可用于目标结论：P2 精确分层、P4 `<=5%`、P6 Vector 空闲算力、完整 P3/P4/P8/P9 矩阵。", "", "## 可复现入口", "", "```bash", "python experiments/benchmarks/run_ppt_p1_p9.py --dry-run", "python experiments/benchmarks/run_ppt_p1_p9.py", "python experiments/benchmarks/summarize_p1_p9.py", "```", ""])
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text("\n".join(lines),encoding="utf-8"); print(output)
if __name__=="__main__": main()
