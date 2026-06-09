import json
import matplotlib.pyplot as plt

RESULTS = [
    {"v_val": 0.1, "p95_ttft": 0.73700, "avg_cost": 0.00521},
    {"v_val": 0.2, "p95_ttft": 0.73805, "avg_cost": 0.00521},
    {"v_val": 0.3, "p95_ttft": 0.73600, "avg_cost": 0.00520},
    {"v_val": 0.5, "p95_ttft": 0.74300, "avg_cost": 0.00519},
    {"v_val": 0.7, "p95_ttft": 0.74300, "avg_cost": 0.00520},
    {"v_val": 1.0, "p95_ttft": 0.81700, "avg_cost": 0.00510},
    {"v_val": 1.5, "p95_ttft": 1.19220, "avg_cost": 0.00505},
    {"v_val": 2.0, "p95_ttft": 1.53100, "avg_cost": 0.00502},
    {"v_val": 3.0, "p95_ttft": 2.37410, "avg_cost": 0.00496},
    {"v_val": 5.0, "p95_ttft": 3.96005, "avg_cost": 0.00489},
    {"v_val": 7.0, "p95_ttft": 5.56740, "avg_cost": 0.00484},
    {"v_val": 10.0, "p95_ttft": 8.28340, "avg_cost": 0.00483},
    {"v_val": 15.0, "p95_ttft": 11.58600, "avg_cost": 0.00482},
    {"v_val": 20.0, "p95_ttft": 15.67120, "avg_cost": 0.00481},
    {"v_val": 30.0, "p95_ttft": 24.23765, "avg_cost": 0.00481},
    {"v_val": 40.0, "p95_ttft": 28.28355, "avg_cost": 0.00480},
    {"v_val": 50.0, "p95_ttft": 35.83400, "avg_cost": 0.00480},
    {"v_val": 60.0, "p95_ttft": 38.61010, "avg_cost": 0.00479},
    {"v_val": 70.0, "p95_ttft": 46.53430, "avg_cost": 0.00479},
    {"v_val": 80.0, "p95_ttft": 53.49765, "avg_cost": 0.00479},
    {"v_val": 90.0, "p95_ttft": 50.42485, "avg_cost": 0.00478},
    {"v_val": 100.0, "p95_ttft": 52.20410, "avg_cost": 0.00478}
]
OUTPUT_FILE = "lyapunov_v_tradeoff_optimized.png"

def main():
    results = RESULTS
    results.sort(key=lambda x: x["v_val"])
    
    plt.rcParams.update({'font.size': 20})
    x_vals = [r["v_val"] for r in results]
    p95_ttfts = [r["p95_ttft"] for r in results]
    avg_costs = [r["avg_cost"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

    color1 = "#d62728"
    ax1.set_xlabel("Lyapunov Control Parameter $V$ (Log Scale)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("P95 TTFT (Seconds)", color=color1, fontsize=13, fontweight="bold")
    
    line1, = ax1.plot(
        x_vals,
        p95_ttfts,
        marker="o",
        markersize=6,
        color=color1,
        linewidth=2.5,
        label="P95 TTFT",
    )
    
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")
    ax1.grid(True, which="major", ls="-", alpha=0.3)
    ax1.grid(True, which="minor", ls="--", alpha=0.1)

    ax2 = ax1.twinx()
    color2 = "#1f77b4"
    line2, = ax2.plot(
        x_vals,
        avg_costs,
        marker="s",
        markersize=6,
        color=color2,
        linewidth=2.5,
        label="Avg Operating Cost",
    )
    ax2.set_ylabel("Time-Average Operating Cost (NTD)", color=color2, fontsize=13, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)

    # 補回第一版原有的 SLO 目標線
    slo_line = ax1.axhline(
        y=6.0,
        color='gray',
        linestyle='dashdot',
        linewidth=2,
        label='SLO Target (6.0s)'
    )

    lines = [line1, line2, slo_line]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=11, frameon=True, shadow=True)

    fig.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"圖表已儲存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()