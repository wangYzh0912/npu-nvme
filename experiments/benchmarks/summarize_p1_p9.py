#!/usr/bin/env python3
"""Summarize primary experiments separately from nested workloads and smoke runs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def collect_results(root):
    root = Path(root)
    records = {f"P{i}": {"primary": [], "auxiliary": []} for i in range(1, 10)}
    for path in sorted(root.rglob("result.json")):
        result = read_json(path)
        config = read_json(path.with_name("config.json"))
        experiment = (result.get("experiment_id") or config.get("experiment_id")
                      or str(result.get("run_id", "")).split("_", 1)[0])
        if experiment not in records:
            continue
        relative = path.relative_to(root)
        nested = any((parent / "result.json").is_file()
                     for parent in path.parent.parents
                     if parent != root and root in parent.parents)
        role = result.get("run_role", config.get("run_role", "primary"))
        parent_id = result.get("parent_run_id", config.get("parent_run_id"))
        auxiliary = (nested or bool(parent_id) or role != "primary"
                     or relative.parts[0] != experiment)
        record = {"path": str(relative), "result": result, "config": config,
                  "reason": "nested workload" if nested or parent_id else
                            "auxiliary directory or role" if auxiliary else None}
        records[experiment]["auxiliary" if auxiliary else "primary"].append(record)
    return records


def newest_results(root, experiment):
    """Compatibility helper: return only this experiment's primary records."""
    return [(Path(root) / row["path"], row["result"])
            for row in collect_results(root)[experiment]["primary"]]


def summarize(root):
    records = collect_results(root)
    summary = {}
    for experiment, groups in records.items():
        counts = Counter(row["result"].get("status", "unknown") for row in groups["primary"])
        summary[experiment] = {"runs": len(groups["primary"]), "statuses": dict(counts),
                               "auxiliary_runs": len(groups["auxiliary"]),
                               "primary_paths": [r["path"] for r in groups["primary"]],
                               "auxiliary": [{"path": r["path"], "reason": r["reason"]}
                                             for r in groups["auxiliary"]]}
    return records, summary


def render_report(root):
    root = Path(root)
    records, summary = summarize(root)
    pre = read_json(root / "preflight.json")
    state = read_json(root / "execution_state.json")
    gates = read_json(root / "gates_summary.json").get("gates", {})
    lines = ["# P1-P9 实验报告", "", f"生成时间：{time.strftime('%F %T %z')}", "",
             f"编排器状态：`{state.get('status', 'unknown')}`；环境门禁：`{pre.get('status', 'unknown')}`。",
             "主实验、嵌套负载和 smoke 分开统计；运行通过不代表完整实验矩阵或性能目标通过。",
             "不同配置的延迟和吞吐不合并为一个数值。", "", "## 主实验汇总", "",
             "| 实验 | 主运行数 | pass | degraded | fail/其他 | 辅助运行数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for experiment, row in summary.items():
        passed = row["statuses"].get("pass", 0)
        degraded = row["statuses"].get("degraded", 0)
        other = row["runs"] - passed - degraded
        lines.append(f"| {experiment} | {row['runs']} | {passed} | {degraded} | {other} | {row['auxiliary_runs']} |")
    lines.extend(["", "## 本轮证据与边界", ""])
    for blocker in pre.get("blockers", []):
        lines.append(f"- 环境阻塞：{blocker}")
    if not gates:
        lines.append("- 本目录没有独立正确性门禁汇总，不从历史结果推定 G0/G1/G2 通过。")
    for gate, record in sorted(gates.items()):
        lines.append(f"- {gate}：`{record.get('status', 'unknown')}`；{record.get('scope', '范围未记录')}。")
    if any(r["config"].get("cross_disk_calibration") for r in records["P1"]["primary"]):
        lines.append("- P1 包含跨物理盘配置，需按设备、请求规格和持久化边界解释；不直接推断严格软件加速比。")
    p2 = records["P2"]["primary"]
    if p2:
        closed = sum(r["result"].get("status") == "pass" and
                     r["result"].get("closure", {}).get("achieved") is True for r in p2)
        lines.append(f"- P2 主实验时间闭合通过 {closed}/{len(p2)}；嵌套 P1 不构成 P2 分层验收。")
        if closed != len(p2):
            lines.append("- P2 未闭合配置不得绘制精确分层百分比。")
    p3 = records["P3"]["primary"]
    if p3:
        modes = sorted({str(r["result"].get("mode") or r["config"].get("mode", "unknown")) for r in p3})
        phase = state.get("phases", {}).get("P3", {})
        lines.append(f"- P3 主运行 {len(p3)} 个，模式为 {', '.join(modes)}；阶段状态 `{phase.get('status', 'unknown')}`。")
        lines.append("- P3 仅报告已完成配置；须在相同配置的 serial/queue/async 对照和真实时间线完整后判断重叠与收益。")
    absent = [exp for exp in records if not records[exp]["primary"]]
    if absent:
        lines.append(f"- {', '.join(absent)} 无本轮主实验结果；不导入历史通过标志。")
    env_ids = sorted({str(r["result"].get("environment_id") or
                          r["config"].get("environment_id") or "legacy-unidentified")
                      for group in records.values() for r in group["primary"]})
    lines.append(f"- 环境标识：{', '.join(env_ids) or '无结果'}；不同环境分别解释，不混算性能。")
    lines.extend(["", "## 辅助结果归属", ""])
    for exp, row in summary.items():
        for record in row["auxiliary"]:
            lines.append(f"- {exp}: `{record['path']}`（{record['reason']}）。")
    if not any(row["auxiliary_runs"] for row in summary.values()):
        lines.append("无辅助结果。")
    return "\n".join(lines) + "\n", summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/ppt-evidence-20260829"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report, summary = render_report(args.root)
    output = args.output or args.root / "P1_P9_REPORT.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
