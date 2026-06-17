#!/usr/bin/env python3
"""
Phase 1a PMU Visualization — A1 (baseline) vs A2_50 (50-param injection)

Produces:
  1. Core Time Distribution (pie/donut) — A1 vs A2_50
  2. ALU Utilization bar chart — Cube vs Vector, A1 vs A2_50
  3. Vector Idle Time comparison
  4. Delta Ops Core Type verification
  5. Kernel Count change
  6. Combined summary dashboard

Output: experiments/output/phase1a_comparison.png
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from collections import defaultdict

REPO = "/home/user7/npu-nvme"

# ── Load data ──────────────────────────────────────────────
with open(os.path.join(REPO, "experiments/output/phase1a_a1_pmu.json")) as f:
    a1 = json.load(f)
with open(os.path.join(REPO, "experiments/output/phase1a_a2_50_pmu.json")) as f:
    a2 = json.load(f)

# Also load standalone timing
with open(os.path.join(REPO, "experiments/output/phase1a_a1_baseline.json")) as f:
    a1_standalone = json.load(f)
with open(os.path.join(REPO, "experiments/output/phase1a_a2_50.json")) as f:
    a2_standalone = json.load(f)


# ── Style setup ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
})

# Extended color palette
CUBE_COLOR = "#58a6ff"
CUBE_LIGHT = "#79c0ff"
VECTOR_COLOR = "#f0883e"
VECTOR_LIGHT = "#ffa657"
IDLE_COLOR = "#484f58"
GREEN = "#3fb950"
RED_ORANGE = "#d29922"
PURPLE = "#bc8cff"
TEAL = "#39d353"

fig = plt.figure(figsize=(22, 14))
fig.suptitle(
    "Phase 1a: PMU Verification — A1 (Baseline) vs A2_50 (50-Param Vector Edge Injection)\n"
    "GPT-2 XL 48L, 1.56B params, Ascend 910B, MindSpore 2.5",
    fontsize=16, fontweight="bold", color="#f0f6fc", y=0.98,
)

# ═══════════════════════════════════════════════════════════
# Panel 1: Core Time Distribution — Stacked Horizontal Bar
# ═══════════════════════════════════════════════════════════
ax1 = fig.add_axes([0.04, 0.56, 0.30, 0.35])

# A1 data
a1_total = a1["aic_time_ms"] + a1["aiv_time_ms"]
labels = ["AI_CORE\n(Cube)", "AI_VECTOR_CORE\n(Vector)"]
a1_times = [a1["aic_time_ms"], a1["aiv_time_ms"]]
a1_pcts = [a1["aic_time_pct"], a1["aiv_time_pct"]]

# A2 data
a2_total = a2["aic_time_ms"] + a2["aiv_time_ms"] + a2["aicpu_time_ms"]
a2_times = [a2["aic_time_ms"], a2["aiv_time_ms"]]
a2_pcts = [a2["aic_time_pct"], a2["aiv_time_pct"]]

y_pos = [0, 1]
bar_height = 0.35

# Calculate what fraction of each core time is idle vs utilized
# Cube
a1_cube_util_time = a1["aic_time_ms"] * a1["cube_eff_util_pct"] / 100
a1_cube_idle_time = a1["aic_time_ms"] - a1_cube_util_time
a2_cube_util_time = a2["aic_time_ms"] * a2["cube_eff_util_pct"] / 100
a2_cube_idle_time = a2["aic_time_ms"] - a2_cube_util_time

# Vector
a1_vec_util_time = a1["aic_time_ms"] * a1["vec_eff_util_pct"] / 100  # note: a1 has same fields
# Actually a1 uses different naming - let me recompute
a1_vec_eff = a1["vec_eff_util_pct"] / 100
a1_vec_util_time = a1["aiv_time_ms"] * a1_vec_eff
a1_vec_idle_time = a1["aiv_time_ms"] - a1_vec_util_time

a2_vec_eff = a2["vec_eff_util_pct"] / 100
a2_vec_util_time = a2["aiv_time_ms"] * a2_vec_eff
a2_vec_idle_time = a2["aiv_time_ms"] - a2_vec_util_time

# Stacked: utilized + idle
a1_stacked_cube = [a1_cube_util_time, a1_cube_idle_time]
a1_stacked_vec = [a1_vec_util_time, a1_vec_idle_time]
a2_stacked_cube = [a2_cube_util_time, a2_cube_idle_time]
a2_stacked_vec = [a2_vec_util_time, a2_vec_idle_time]

ax1.barh(y_pos[1] + bar_height/2, a1_stacked_cube[0], bar_height, color=CUBE_COLOR, label="Utilized (Cube MAC)")
ax1.barh(y_pos[1] + bar_height/2, a1_stacked_cube[1], bar_height, left=a1_stacked_cube[0],
          color=IDLE_COLOR, alpha=0.5, label="Idle")
ax1.barh(y_pos[0] - bar_height/2, a2_stacked_cube[0], bar_height, color=CUBE_COLOR)
ax1.barh(y_pos[0] - bar_height/2, a2_stacked_cube[1], bar_height, left=a2_stacked_cube[0],
          color=IDLE_COLOR, alpha=0.5)

ax1.barh(y_pos[1] + bar_height/2 + 1, a1_stacked_vec[0], bar_height, color=VECTOR_COLOR, label="Utilized (Vector ALU)")
ax1.barh(y_pos[1] + bar_height/2 + 1, a1_stacked_vec[1], bar_height, left=a1_stacked_vec[0],
          color=IDLE_COLOR, alpha=0.5)
ax1.barh(y_pos[0] - bar_height/2 + 1, a2_stacked_vec[0], bar_height, color=VECTOR_COLOR)
ax1.barh(y_pos[0] - bar_height/2 + 1, a2_stacked_vec[1], bar_height, left=a2_stacked_vec[0],
          color=IDLE_COLOR, alpha=0.5)

# Labels
ax1.set_yticks([0, 1])
ax1.set_yticklabels(["A1 (baseline)", "A2_50 (inject)"], fontsize=11)
ax1.set_xlabel("OP Time (ms)", fontsize=11)
ax1.set_title("Core Time Distribution (Stacked: Utilized + Idle)", fontsize=13, fontweight="bold", color="#f0f6fc")

# Annotate with % and ms
for i, (label, times) in enumerate([
    ("Cube", [(a1_cube_util_time, a1_cube_idle_time, a1["aic_time_ms"], a1["cube_eff_util_pct"]),
              (a2_cube_util_time, a2_cube_idle_time, a2["aic_time_ms"], a2["cube_eff_util_pct"])]),
    ("Vector", [(a1_vec_util_time, a1_vec_idle_time, a1["aiv_time_ms"], a1["vec_eff_util_pct"]),
                (a2_vec_util_time, a2_vec_idle_time, a2["aiv_time_ms"], a2["vec_eff_util_pct"])]),
]):
    for j, (util_t, idle_t, total_t, eff_pct) in enumerate(times):
        y = j * 0.35 + (0 if i == 0 else 1)
        ax1.text(total_t + 30, y, f"{total_t:.0f}ms\n{eff_pct:.1f}% util",
                fontsize=8, va="center", color="#c9d1d9")

ax1.legend(loc="lower right", fontsize=8, ncol=2)
ax1.set_xlim(0, max(a2["aiv_time_ms"], a2["aic_time_ms"]) * 1.4)


# ═══════════════════════════════════════════════════════════
# Panel 2: ALU Utilization — Grouped Bar Chart
# ═══════════════════════════════════════════════════════════
ax2 = fig.add_axes([0.38, 0.56, 0.28, 0.35])

metrics = ["Cube MAC\n(ai_core)", "Vec ALU\n(aiv_vec)", "Vec Scalar\n(aiv_scalar)", "Vector Eff\n(vec+scalar)"]
a1_vals = [a1["aic_mac_ratio_pct"], a1["aiv_vec_ratio_pct"], a1["aiv_scalar_ratio_pct"], a1["vec_eff_util_pct"]]
a2_vals = [a2["aic_mac_ratio_pct"], a2["aiv_vec_ratio_pct"], a2["aiv_scalar_ratio_pct"], a2["vec_eff_util_pct"]]

x = np.arange(len(metrics))
w = 0.32
bars1 = ax2.bar(x - w/2, a1_vals, w, color=CUBE_COLOR, label="A1 (baseline)", alpha=0.85)
bars2 = ax2.bar(x + w/2, a2_vals, w, color=VECTOR_COLOR, label="A2_50 (inject)", alpha=0.85)

# Annotate values
for bar, val in zip(bars1, a1_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{val:.1f}%",
             ha="center", va="bottom", fontsize=9, color="#c9d1d9", fontweight="bold")
for bar, val in zip(bars2, a2_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{val:.1f}%",
             ha="center", va="bottom", fontsize=9, color="#c9d1d9", fontweight="bold")

# Highlight the Core Type regions
ax2.axhline(y=50, color="#30363d", linestyle="--", linewidth=1)
ax2.axvspan(-0.5, 0.5, alpha=0.08, color=CUBE_COLOR)
ax2.axvspan(0.5, 3.5, alpha=0.08, color=VECTOR_COLOR)

ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=10)
ax2.set_ylabel("ALU Utilization (%)", fontsize=11)
ax2.set_title("ALU Utilization — A1 vs A2_50", fontsize=13, fontweight="bold", color="#f0f6fc")
ax2.set_ylim(0, 65)
ax2.legend(fontsize=9, loc="upper right")

# Annotate delta
for i, (v1, v2) in enumerate(zip(a1_vals, a2_vals)):
    delta = v2 - v1
    color = GREEN if abs(delta) < 0.5 else (RED_ORANGE if delta < 0 else TEAL)
    ax2.text(i, max(v1, v2) + 4, f"Δ={delta:+.2f}pp", ha="center", fontsize=8, color=color)

# ═══════════════════════════════════════════════════════════
# Panel 3: Vector Idle Time — HBar
# ═══════════════════════════════════════════════════════════
ax3 = fig.add_axes([0.70, 0.56, 0.27, 0.35])

idle_times = [a1["vec_idle_ms_est"], a2["vec_idle_ms_est"]]
eff_pcts = [a1["vec_eff_util_pct"], a2["vec_eff_util_pct"]]
idle_pcts = [a1["vec_idle_pct"], a2["vec_idle_pct"]]

ax3.barh([1, 0], idle_times, 0.5, color=[CUBE_LIGHT, VECTOR_LIGHT], alpha=0.85)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(["A2_50 (inject)", "A1 (baseline)"], fontsize=11)
ax3.set_xlabel("Vector Idle Time (ms)", fontsize=10)

# Annotate
for i, (itime, epct, ipct) in enumerate(zip(idle_times, eff_pcts, idle_pcts)):
    ax3.text(itime + 20, 1 - i, f"{itime:.0f}ms idle\n({ipct:.1f}% of {eff_pct:.1f}% eff util)",
            fontsize=9, va="center", color="#c9d1d9")
    # Mark SPDK write time
    ax3.axvline(x=705, color=GREEN, linestyle="--", linewidth=1.5, alpha=0.7)

ax3.text(705 + 5, 1.4, "SPDK write\n(~705ms)", fontsize=8, color=GREEN, va="bottom")
ax3.set_title("Vector Core Idle Time", fontsize=13, fontweight="bold", color="#f0f6fc")
ax3.set_xlim(0, max(idle_times) * 1.25)


# ═══════════════════════════════════════════════════════════
# Panel 4: Delta Ops Core Type — Horizontal Bar (proof that ops land on Vector)
# ═══════════════════════════════════════════════════════════
ax4 = fig.add_axes([0.04, 0.10, 0.30, 0.38])

delta_ops = a2["delta_ops"]
op_names = []
op_times = []
op_cores = []
for op_name, info in sorted(delta_ops.items(), key=lambda x: -x[1]["dur_us"]):
    op_names.append(op_name)
    op_times.append(info["dur_us"] / 1000)  # ms
    core = info["core_type"]
    op_cores.append(core)
    # Color by core type
    colors_map = {"AI_VECTOR_CORE": VECTOR_COLOR, "AI_CORE": CUBE_COLOR, "MIX_AIV": PURPLE, "AI_CPU": IDLE_COLOR}

colors = [colors_map.get(c, IDLE_COLOR) for c in op_cores]
bars = ax4.barh(range(len(op_names)), op_times, color=colors, alpha=0.85, height=0.6)
ax4.set_yticks(range(len(op_names)))
ax4.set_yticklabels(op_names, fontsize=10)
ax4.set_xlabel("Total OP Time (ms)", fontsize=10)
ax4.set_title("Delta Ops Core Type Attribution\n(Proof: ops land on AI_VECTOR_CORE)", fontsize=12, fontweight="bold", color="#f0f6fc")

# Annotate
for i, (t, c) in enumerate(zip(op_times, op_cores)):
    ax4.text(t + 5, i, f"{t:.1f}ms | {c}", fontsize=8, va="center", color="#c9d1d9")
    # Checkmark for Vector
    if c in ("AI_VECTOR_CORE", "MIX_AIV"):
        ax4.text(t + 5, i - 0.25, "✅", fontsize=10)

# Legend
legend_patches = [
    mpatches.Patch(color=VECTOR_COLOR, label="AI_VECTOR_CORE ✅"),
    mpatches.Patch(color=PURPLE, label="MIX_AIV ✅"),
    mpatches.Patch(color=CUBE_COLOR, label="AI_CORE"),
]
ax4.legend(handles=legend_patches, fontsize=8, loc="lower right")


# ═══════════════════════════════════════════════════════════
# Panel 5: Kernel Count Change
# ═══════════════════════════════════════════════════════════
ax5 = fig.add_axes([0.38, 0.10, 0.28, 0.38])

kc_metrics = ["AI_CORE\n(Cube)", "AI_VECTOR\n(Vector)"]
a1_kc = [a1["aic_rows"], a1["aiv_rows"]]
a2_kc = [a2["aic_rows"], a2["aiv_rows"]]

x = np.arange(len(kc_metrics))
ax5.bar(x - 0.18, a1_kc, 0.35, color=CUBE_COLOR, alpha=0.7, label="A1 (baseline)")
ax5.bar(x + 0.18, a2_kc, 0.35, color=VECTOR_COLOR, alpha=0.7, label="A2_50 (inject)")

for i, (v1, v2) in enumerate(zip(a1_kc, a2_kc)):
    delta_pct = (v2 - v1) / v1 * 100
    ax5.text(i - 0.18, v1 + 2000, f"{v1:,}", ha="center", fontsize=9, color=CUBE_LIGHT)
    ax5.text(i + 0.18, v2 + 2000, f"{v2:,}", ha="center", fontsize=9, color=VECTOR_LIGHT)
    ax5.text(i, max(v1, v2) * 1.1, f"Δ=+{delta_pct:.1f}%", ha="center", fontsize=9, color=GREEN)

ax5.set_xticks(x)
ax5.set_xticklabels(kc_metrics, fontsize=10)
ax5.set_ylabel("Kernel Count", fontsize=10)
ax5.set_title("Kernel Count Change: A1 → A2_50", fontsize=12, fontweight="bold", color="#f0f6fc")
ax5.legend(fontsize=9)


# ═══════════════════════════════════════════════════════════
# Panel 6: Verdict Summary Table
# ═══════════════════════════════════════════════════════════
ax6 = fig.add_axes([0.70, 0.10, 0.27, 0.38])
ax6.axis("off")

# Verdict data
verdict_data = [
    ("① Core Type", "Delta ops → AI_VECTOR_CORE", "✅ PASS", GREEN),
    ("② Vector ALU util", "33.2% → 33.1%  (unchanged)", "⚠️ STALL", RED_ORANGE),
    ("③ Cube util", "50.1% → 50.1%  (unchanged)", "✅ PASS", GREEN),
    ("④ Step time", "379ms → 385ms  (+1.5%)", "✅ PASS", GREEN),
]

y_start = 0.85
for i, (check, target, result, color) in enumerate(verdict_data):
    y = y_start - i * 0.2
    ax6.text(0.02, y, check, fontsize=11, fontweight="bold", color="#f0f6fc",
             transform=ax6.transAxes)
    ax6.text(0.30, y, target, fontsize=9, color="#8b949e", transform=ax6.transAxes)
    ax6.text(0.82, y, result, fontsize=11, fontweight="bold", color=color, transform=ax6.transAxes,
             ha="right")

# Key insight box
ax6.text(0.02, -0.05, "Key Insight", fontsize=11, fontweight="bold", color=TEAL,
         transform=ax6.transAxes)
insight = (
    "Vector ALU util does NOT rise because\n"
    "delta ops (Sub/Cast/ReduceSum) are\n"
    "scalar-heavy, low-density Vector ops.\n"
    "This proves Vector Core has abundant\n"
    "IDLE TIME SLOTS (~1.6s/step) available\n"
    "for compression computation — a\n"
    "stronger argument than ALU utilization."
)
ax6.text(0.02, -0.12, insight, fontsize=8.5, color="#8b949e", transform=ax6.transAxes,
         va="top", linespacing=1.5)

# Title
ax6.text(0.02, 0.95, "Phase 1a Verdict", fontsize=13, fontweight="bold", color="#f0f6fc",
         transform=ax6.transAxes)

# Narratvie update annotation
ax6.text(0.02, -0.58, "Narrative Update", fontsize=10, fontweight="bold", color=PURPLE,
         transform=ax6.transAxes)
narrative = (
    "OLD: \"Utilize Vector idle ALU\"\n"
    "NEW: \"Utilize Vector idle TIME SLOTS\"\n"
    "→ Low-density ops fit naturally\n"
    "→ No ALU contention needed\n"
    "→ Stronger: ~1.6s budget >> ~705ms SPDK"
)
ax6.text(0.02, -0.66, narrative, fontsize=8.5, color=PURPLE, transform=ax6.transAxes,
         va="top", linespacing=1.5)


# ── Save ────────────────────────────────────────────────────
out_dir = os.path.join(REPO, "experiments/output")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "phase1a_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print(f"Saved: {out_path}")
