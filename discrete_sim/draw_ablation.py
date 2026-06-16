"""
draw_ablation.py — Standalone figure renderer for ablation study results.
=========================================================================
Reads  discrete_sim/results/ablation_results.json  (produced by run_ablation.py)
and writes  discrete_sim/results/ablation_combined.png.

Run from the project root:
    python discrete_sim/draw_ablation.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
JSON_PATH   = os.path.join(RESULTS_DIR, "ablation_results.json")
OUT_PATH    = os.path.join(RESULTS_DIR, "ablation_combined.png")

# ── Variant metadata ───────────────────────────────────────────────────────────
VARIANTS = [
    ("ours",         "Ours\n(Full)"),
    ("no_semantic",  "w/o\nSemantic"),
    ("no_provision", "w/o\nProvision"),
    ("no_merge",     "w/o\nMerge"),
    ("no_autoscale", "w/o\nAutoScale"),
]

SCALES = {
    "1c_2n": "1 Cluster (2 Nodes)",
    "2c_3n": "2 Clusters (3 Nodes)",
    "3c_5n": "3 Clusters (15 Nodes)",
}

COST_RPS_LIST = [5, 8, 10, 12, 14, 16, 20]

VARIANT_COLORS = {
    "ours":         "#1a6ea8",
    "no_semantic":  "#e07b39",
    "no_provision": "#2aab60",
    "no_merge":     "#9b59b6",
    "no_autoscale": "#c0392b",
}
VARIANT_LINESTYLES = {
    "ours":         "-",
    "no_semantic":  "--",
    "no_provision": "-.",
    "no_merge":     ":",
    "no_autoscale": (0, (3, 1, 1, 1)),
}
VARIANT_MARKERS = {
    "ours":         "o",
    "no_semantic":  "s",
    "no_provision": "^",
    "no_merge":     "D",
    "no_autoscale": "v",
}

SCALE_COLORS = ["#4C72B0", "#DD8452", "#55A868"]


# ── Load data ──────────────────────────────────────────────────────────────────
def load_data(path: str):
    with open(path) as f:
        raw = json.load(f)

    # cost_vs_rps keys come out of JSON as strings → convert to int
    cost = {}
    for vk, rmap in raw["cost_vs_rps"].items():
        cost[vk] = {int(k): v for k, v in rmap.items()}

    return raw["max_throughput"], cost


# ── Draw ───────────────────────────────────────────────────────────────────────
def draw(summary_tput, summary_cost, out_path: str):
    keys      = [v for v, _ in VARIANTS]
    xlabels   = [lbl.replace("\n", " ") for _, lbl in VARIANTS]
    labels    = {v: lbl.replace("\n", " ") for v, lbl in VARIANTS}
    scale_ids = list(SCALES.keys())
    rps_list  = sorted(COST_RPS_LIST)
    n_v       = len(keys)
    n_s       = len(scale_ids)

    # ── Build throughput matrix ────────────────────────────────────────────────
    tput_mat = np.zeros((n_v, n_s))
    for vi, vk in enumerate(keys):
        for si, sk in enumerate(scale_ids):
            e = summary_tput.get(vk, {}).get(sk, {})
            tput_mat[vi, si] = e.get("max_throughput", 0.0)

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), dpi=150)
    fig.suptitle("Ablation Study: System Component Contributions",
                 fontsize=15, fontweight="bold", y=1.02)

    # ══════════════════════════════════════════════════════════════════════════
    # (a) Max Stable Throughput — grouped bar chart
    # ══════════════════════════════════════════════════════════════════════════
    ax = axes[0]
    bw      = 0.22
    x       = np.arange(n_v)
    offsets = np.linspace(-(n_s - 1) / 2.0, (n_s - 1) / 2.0, n_s) * bw

    bar_handles = []
    for si, (sk, col) in enumerate(zip(scale_ids, SCALE_COLORS)):
        bars = ax.bar(x + offsets[si], tput_mat[:, si], bw,
                      label=SCALES[sk], color=col,
                      edgecolor="black", linewidth=0.6)
        bar_handles.append(bars[0])          # save one patch per scale for legend
        for bar, val in zip(bars, tput_mat[:, si]):
            if val > 0:
                ax.annotate(f"{val:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, val),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9.5)
    ax.set_ylabel("Max Stable Throughput (Req/s)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Ablation Variant", fontsize=11, fontweight="bold")
    ax.set_title("(a) Max Stable Throughput", fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(handles=bar_handles,
              labels=[SCALES[sk] for sk in scale_ids],
              title="Federation Scale",
              fontsize=8.5, title_fontsize=9.5,
              loc="center right")
    ax.text(0.97, 0.97, "Higher is better", transform=ax.transAxes,
            fontsize=8, ha="right", va="top", color="gray", style="italic")

    # ══════════════════════════════════════════════════════════════════════════
    # (b) Total Cost vs RPS — line chart
    # ══════════════════════════════════════════════════════════════════════════
    ax = axes[1]
    for vk in keys:
        xs, ys = [], []
        for rps in rps_list:
            entry = summary_cost.get(vk, {}).get(rps, {})
            val   = entry.get("total_cost_per_req")
            if val is not None:
                xs.append(rps)
                ys.append(val)
        if xs:
            ax.plot(xs, ys,
                    label=labels[vk],
                    color=VARIANT_COLORS[vk],
                    linestyle=VARIANT_LINESTYLES[vk],
                    marker=VARIANT_MARKERS[vk],
                    linewidth=2.2, markersize=7)

    ax.set_xlabel("Global Requests Per Second (RPS)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Avg Cost / Request (network + instance + drop penalty)",
                  fontsize=11, fontweight="bold")
    ax.set_title("(b) Total Cost vs Load", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper left")
    ax.text(0.97, 0.97, "Lower is better", transform=ax.transAxes,
            fontsize=8, ha="right", va="top", color="gray", style="italic")

    # ══════════════════════════════════════════════════════════════════════════
    # (c) GPU Active Time vs RPS — line chart
    # ══════════════════════════════════════════════════════════════════════════
    ax = axes[2]
    for vk in keys:
        xs, ys = [], []
        for rps in rps_list:
            entry = summary_cost.get(vk, {}).get(rps, {})
            val   = entry.get("gpu_active_ms_per_req")
            if val is not None:
                xs.append(rps)
                ys.append(val)
        if xs:
            ax.plot(xs, ys,
                    label=labels[vk],
                    color=VARIANT_COLORS[vk],
                    linestyle=VARIANT_LINESTYLES[vk],
                    marker=VARIANT_MARKERS[vk],
                    linewidth=2.2, markersize=7)

    ax.set_xlabel("Global Requests Per Second (RPS)", fontsize=11, fontweight="bold")
    ax.set_ylabel("GPU Active Time / Request (ms/req)", fontsize=11, fontweight="bold")
    ax.set_title("(c) GPU Resource Consumption vs Load", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.97, 0.97, "Lower is better", transform=ax.transAxes,
            fontsize=8, ha="right", va="top", color="gray", style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[draw_ablation] Saved → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[draw_ablation] Loading {JSON_PATH}")
    summary_tput, summary_cost = load_data(JSON_PATH)
    draw(summary_tput, summary_cost, OUT_PATH)
