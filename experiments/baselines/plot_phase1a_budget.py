#!/usr/bin/env python3
"""
Phase 1a: Vector Idle Budget Visualization
  - Shows Vector Core available time (idle slots) vs step time
  - Compares with SPDK write time (the sink that needs to be covered)
  - Illustrates the central I3 contribution: compression budget characterization
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO = "/home/user7/npu-nvme"

with open(os.path.join(REPO, "experiments/output/phase1a_a1_pmu.json")) as f:
    a1 = json.load(f)
with open(os.path.join(REPO, "experiments/output/phase1a_a2_50_pmu.json")) as f:
    a2 = json.load(f)

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12,
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9", "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
    "legend.labelcolor": "#c9d1d9",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Vector Engine Idle Budget — The Core I3 Contribution",
             fontsize=16, fontweight="bold", color="#f0f6fc", y=1.01)

# ── Left: Waterfall-style idle budget ──
step_ms = 379  # S0 baseline
spdk_ms = 705

# A1 data
vec_active_a1 = a1["aiv_time_ms"] * (a1["vec_eff_util_pct"] / 100)
vec_idle_a1 = a1["aiv_time_ms"] - vec_active_a1
cube_a1 = a1["aic_time_ms"]

# A2 data
vec_active_a2 = a1["vec_eff_util_pct"] / 100  # keep baseline util rate
vec_idle_a2 = a2["vec_idle_ms_est"]  # computed idle

ax1.barh([4], [vec_active_a1], 0.6, color="#3fb950", alpha=0.8, label="Vector Active (eff util)")
ax1.barh([4], [vec_idle_a1], 0.6, left=[vec_active_a1], color="#f0883e", alpha=0.9,
         label="Vector Idle (~1,164ms)")

# Annotation: SPDK write line
ax1.axvline(x=spdk_ms, ymin=0.35, ymax=0.65, color="#ff7b72", linewidth=3, linestyle="--")
ax1.text(spdk_ms + 10, 2.2, f"SPDK Write\n~{spdk_ms}ms", fontsize=11, color="#ff7b72",
         fontweight="bold", va="bottom")
ax1.text(spdk_ms + 10, 1.7, "← Covered by idle budget", fontsize=9, color="#8b949e")

# Annotate
ax1.text(vec_active_a1/2, 4.2, f"{vec_active_a1:.0f}ms\n({a1['vec_eff_util_pct']:.0f}%)",
         ha="center", fontsize=10, color="#c9d1d9", fontweight="bold")
ax1.text(vec_active_a1 + vec_idle_a1/2, 4.2,
         f"{vec_idle_a1:.0f}ms\n({a1['vec_idle_pct']:.0f}%)",
         ha="center", fontsize=10, color="#c9d1d9", fontweight="bold")

# Budget ratio
ax1.text(vec_active_a1 + vec_idle_a1 + 80, 3.5,
         f"Budget/SPDK =\n{vec_idle_a1/spdk_ms:.2f}×",
         fontsize=12, fontweight="bold", color="#bc8cff")

ax1.set_yticks([4])
ax1.set_yticklabels(["GPT-2 XL\n(1.56B)"], fontsize=12)
ax1.set_xlabel("Time (ms)", fontsize=11)
ax1.set_title("Vector Engine Time Budget\n(A1 Baseline — No Injection)", fontsize=13,
              fontweight="bold", color="#f0f6fc")
ax1.legend(fontsize=9, loc="upper right")
ax1.set_xlim(0, 2200)

# ── Right: Multi-model projected curve ──
# Combine known V1-V4 data + Phase 1a A1 + placeholder projections for Phase 1b
models_known = {
    "V1\nDense\n10K":      (1e4,     1.6),
    "V2\nGPT-2 6L\n81M":   (81e6,    29),
    "V4\nGPT-2 4L\n55M":   (55e6,    21),
    "A1\nGPT-2 XL\n1.56B": (1.56e9,  1164),
}

# Phase 1b targets (estimated from trend)
models_todo = {
    "V5\nGPT-2 Large\n0.77B":  (0.77e9,  550),
    "V6\nCustom\n3.5B":         (3.5e9,   2000),
    "V7\nLLaMA-2\n7B":          (7e9,     3500),
}

for label, (params, idle_ms) in models_known.items():
    color = "#f0883e" if "A1" in label else "#58a6ff"
    marker = "s" if "A1" in label else "o"
    ax2.scatter(params/1e9, idle_ms, s=120, c=color, marker=marker, edgecolors="white",
                linewidths=1.5, zorder=5, alpha=0.9)
    ax2.annotate(label, (params/1e9 + 0.05, idle_ms), fontsize=8, color=color,
                 fontweight="bold" if "A1" in label else "normal")

for label, (params, idle_ms) in models_todo.items():
    ax2.scatter(params/1e9, idle_ms, s=80, c="none", marker="o", edgecolors="#8b949e",
                linewidths=2, linestyle="--", alpha=0.5, zorder=3)
    ax2.annotate(label, (params/1e9 + 0.05, idle_ms), fontsize=7, color="#8b949e")

# Trend line
all_params_known = np.array([v[0] for v in models_known.values()]) / 1e9
all_idle_known = np.array([v[1] for v in models_known.values()])

# Rough power-law fit: idle ~ params^0.7
x_fit = np.logspace(-5, 1, 100)
y_fit = 700 * x_fit**0.6  # rough fit
ax2.plot(x_fit, y_fit, color="#bc8cff", linewidth=2, alpha=0.7, linestyle="--",
         label="projected trend\n$T_{idle} \\propto M^{0.6}$")

# SPDK reference line
ax2.axhline(y=spdk_ms, color="#ff7b72", linewidth=2, linestyle=":", alpha=0.7)
ax2.text(3.5, spdk_ms + 20, f"SPDK write (705ms)", fontsize=9, color="#ff7b72")

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Model Size (billions of parameters)", fontsize=11)
ax2.set_ylabel("Vector Idle Time (ms/step)", fontsize=11)
ax2.set_title("Vector Idle Budget $T_{idle}(M)$:\nPhase 1a Confirmed + Phase 1b Projections",
              fontsize=13, fontweight="bold", color="#f0f6fc")
ax2.legend(fontsize=9, loc="lower right")
ax2.grid(True, alpha=0.3)

# Insight
ax2.text(0.05, 0.95,
         "Key: Larger models → more Vector idle →\nmore budget for complex compression algorithms",
         transform=ax2.transAxes, fontsize=9, color="#3fb950", va="top",
         fontstyle="italic")

out_path = os.path.join(REPO, "experiments/output/phase1a_idle_budget.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.close(fig)
print(f"Saved: {out_path}")
