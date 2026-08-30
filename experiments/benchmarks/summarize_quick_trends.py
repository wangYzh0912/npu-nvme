#!/usr/bin/env python3
"""Write a compact, honest report for the two-hour quick trend round."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def mean_stat(value):
    if isinstance(value, dict):
        value = value.get("mean")
    return value if isinstance(value, (int, float)) else None


def mib(value):
    return f"{value / 1048576:.1f}" if isinstance(value, (int, float)) else "n/a"


def ms(value):
    return f"{value:.1f}" if isinstance(value, (int, float)) else "n/a"


def number(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "unavailable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    p1 = load(root / "P1/summary.json", {}).get("groups", [])
    p3 = load(root / "P3/summary.json", {}).get("groups", [])
    p4 = load(root / "P4/summary.json", {}).get("rows", [])
    p5 = load(root / "P5/summary.json", {}).get("records", [])
    p6 = load(root / "P6/summary.json", {}).get("records", [])
    p6_profile = load(root / "P6/profile_summary.json", {}).get("records", [])
    state = load(root / "quick_state.json", {})

    lines = [
        "# Quick Trend Round",
        "",
        "本轮是约两小时的方向性探测，不替代正式验收矩阵。正式样本量、模型规模和多 seed 覆盖不足时，只报告趋势，不报告最终性能结论。",
        "",
        f"状态：`{state.get('status', 'unknown')}`；编排记录数：{len(state.get('records', []))}。",
        "",
        "## P1 同规格路径",
        "",
        "4 MiB、256 MiB 总逻辑数据、QD 1/4 的快速样本；下表为 4 MiB 组平均值，吞吐按十进制 GB/s。",
        "",
        "| 操作/QD | buffered 延迟 ms | O_DIRECT 延迟 ms | SPDK host 延迟 ms | O_DIRECT/SPDK |",
        "|---|---:|---:|---:|---:|",
    ]
    p1_main = [x for x in p1 if x.get("block_size") == 4 * 1024 * 1024]
    for group in sorted(p1_main, key=lambda x: (x.get("operation", ""), x.get("queue_depth", 0))):
        paths = group.get("paths", {})
        direct = mean_stat(paths.get("odirect", {}).get("latency_mean"))
        spdk = mean_stat(paths.get("spdk_host", {}).get("latency_mean"))
        ratio = direct / spdk if direct is not None and spdk else None
        lines.append(
            f"| {group.get('operation')}/QD{group.get('queue_depth')} | "
            f"{ms(mean_stat(paths.get('buffered', {}).get('latency_mean')))} | "
            f"{ms(direct)} | {ms(spdk)} | {ratio:.3f} |"
            if ratio is not None else
            f"| {group.get('operation')}/QD{group.get('queue_depth')} | n/a | n/a | n/a | n/a |"
        )
    lines += [
        "",
        "观察：本轮 O_DIRECT 在四个 4 MiB 组合中均低于 SPDK host 延迟；因此不能把这轮快速数据写成‘裸盘路径胜过 O_DIRECT’，只能写成‘路径差异明显，需正式双盘校准’。",
        "",
        "## P3 DMA-NVMe 时间轴",
        "",
        "只把包含 serial/queue/async 三模式的 4 MiB 组视为完整组；早先中断留下的 1 MiB serial-only 记录保留但不参与结论。",
        "",
        "| depth | 注入延迟 ms | async overlap median | async/queue speedup |",
        "|---:|---:|---:|---:|",
    ]
    complete_p3 = [x for x in p3 if x.get("chunk_size") == 4 * 1024 * 1024 and
                   len(x.get("run_ids", {})) == 3]
    for group in sorted(complete_p3, key=lambda x: (x.get("depth", 0), x.get("delay_ms", 0))):
        overlap = group.get("overlap_rate") or {}
        lines.append(
            f"| {group.get('depth')} | {group.get('delay_ms')} | "
            f"{overlap.get('median', 'n/a'):.3f} | "
            f"{group.get('async_speedup_vs_queue', 'n/a'):.3f} |"
            if isinstance(overlap.get("median"), (int, float)) and
            isinstance(group.get("async_speedup_vs_queue"), (int, float)) else
            f"| {group.get('depth')} | {group.get('delay_ms')} | n/a | n/a |"
        )
    lines += [
        "",
        "趋势：depth=1 的 overlap 为 0；depth=4 的两组 overlap median 约 0.943/0.953，但 async 相对已有 queue 仅约 1.00x/1.04x。说明确实存在时间交叠，尚未证明新 async 实现带来稳定端到端收益。",
        "",
        "## P4 训练影响",
        "",
        "| 模式 | 吞吐 step/s | 吞吐下降 | checkpoint/普通 step - 1 | 前台等待均值 ms | 恢复校验 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in sorted(p4, key=lambda x: x.get("mode", "")):
        lines.append(
            f"| {row.get('mode')} | {row.get('throughput_steps_s', 'n/a'):.4f} | "
            f"{(row.get('throughput_overhead') * 100):.2f}% | "
            f"{row.get('step_overhead_percent', 'n/a'):.2f}% | "
            f"{ms(mean_stat(row.get('foreground_wait')))} | "
            f"{row.get('restore_verified')} |"
        )
    lines += [
        "",
        "注意：sync/async 都出现 parameter checksum mismatch，故恢复校验为 false；该失败本身是本轮的有效趋势/阻塞证据，不能把 P4 记为通过。",
        "",
        "## P5 环形缓冲 RSS",
        "",
        "| 槽数 | 分块 MiB | 期望池 MiB | 增量 RSS MiB | host 峰值 RSS MiB | HugePage delta MiB |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(p5, key=lambda x: (x.get("slots", 0), x.get("chunk_size", 0))):
        lines.append(
            f"| {row.get('slots')} | {mib(row.get('chunk_size'))} | {mib(row.get('expected_pool_bytes'))} | "
            f"{mib(row.get('incremental_rss'))} | {mib(row.get('host_rss_peak'))} | {mib(row.get('pinned_dram_peak'))} |"
        )
    lines += [
        "",
        "观察：期望池大小严格按 slots×chunk 变化；进程 RSS 还包含 Python/NPU 基线，不能用 RSS 绝对值直接等同于 ring 大小。`pinned_dram_peak` 是 HugeTLB 可用页差值字段，报告中不将其称为已分配 pinned bytes。",
        "",
        "## P6 辅助任务与真实利用率",
        "",
        "| 模式 | 辅助任务 | 总延迟均值 ms | 前台等待均值 ms | 状态 |",
        "|---|---|---:|---:|---|",
    ]
    for row in p6:
        lines.append(
            f"| {row.get('mode')} | {row.get('auxiliary')} | {ms(row.get('latency_mean_ms'))} | "
            f"{ms(row.get('foreground_wait_mean_ms'))} | {row.get('status')} |"
        )
    lines += ["", "辅助 diff 在本轮约增加 61 ms；npu_parallel 与 npu_serial 基本相同，不能仅凭该注入实验声称并行已生效。", ""]
    if p6_profile:
        lines += [
            "真实 msprof 导出摘要：",
            "",
            "| 指标组 | Vector 时间线均值 | HBM 设备读 GB/s | HBM 设备写 GB/s |",
            "|---|---:|---:|---:|",
        ]
        for row in p6_profile:
            summary = row.get("timeline_summary") or {}
            lines.append(
                f"| {row.get('metric_group')} | {number(summary.get('vector_mean'))} | "
                f"{number(summary.get('hbm_device_average_read_gb_s'))} | "
                f"{number(summary.get('hbm_device_average_write_gb_s'))} |"
            )
        lines += [
            "",
            "Vector 值仅在导出的 PMU 字段可用时解释为时间线投影，不能直接当作整颗 NPU 的百分比占用；HBM 列是设备平均带宽，也不是 HBM 利用率百分比。",
            "",
        ]
    else:
        lines += ["真实 msprof 时间线摘要：尚未生成。", ""]
    lines += [
        "## 结论边界",
        "",
        "- 本轮目标是快速拿到变化趋势，样本数为 3--8，单 seed，GPT-2；不能替代 GPT-2 XL、多 seed、30 样本正式实验。",
        "- 可保留的趋势：P3 depth=4 出现稳定 DMA/NVMe 时间交叠；P5 池容量随 slots×chunk 配置变化；P4 checkpoint 仍是主要前台代价；P1 三路径性能排序需要重新校准。",
        "- 必须继续修复/复测：P4 checksum mismatch、P6 并行辅助任务无收益，以及 P1 双盘校准和正式置信区间。",
        "",
        "原始运行目录：`/tmp/npu-nvme-quick-trend-20260830`；本目录仅提交紧凑摘要。",
        "",
    ]
    output = root / "QUICK_TREND_REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
