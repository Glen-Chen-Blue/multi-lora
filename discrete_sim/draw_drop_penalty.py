import json
import os
import matplotlib.pyplot as plt


RESULTS = [
    {"multiplier": 1, "drop_rate": 20.95, "download_cost": 108.0},
    {"multiplier": 2, "drop_rate": 16.03, "download_cost": 186.0},
    {"multiplier": 3, "drop_rate": 13.92, "download_cost": 243.0},
    {"multiplier": 5, "drop_rate": 11.85, "download_cost": 327.0},
    {"multiplier": 7, "drop_rate": 10.81, "download_cost": 390.0},
    {"multiplier": 10, "drop_rate": 9.75, "download_cost": 483.0},
    {"multiplier": 12, "drop_rate": 9.77, "download_cost": 513.0},
    {"multiplier": 16, "drop_rate": 8.72, "download_cost": 621.0},
    {"multiplier": 20, "drop_rate": 8.72, "download_cost": 621.0},
    {"multiplier": 24, "drop_rate": 8.72, "download_cost": 621.0},
    {"multiplier": 28, "drop_rate": 7.72, "download_cost": 621.0},
    {"multiplier": 30, "drop_rate": 7.72, "download_cost": 621.0},
    {"multiplier": 32, "drop_rate": 7.74, "download_cost": 621.0},
    {"multiplier": 36, "drop_rate": 7.74, "download_cost": 621.0},
    {"multiplier": 40, "drop_rate": 7.74, "download_cost": 621.0},
    {"multiplier": 45, "drop_rate": 7.74, "download_cost": 621.0},
    {"multiplier": 50, "drop_rate": 7.74, "download_cost": 621.0},
    {"multiplier": 55, "drop_rate": 7.74, "download_cost": 621.0},
    {"multiplier": 60, "drop_rate": 6.72, "download_cost": 621.0},
    {"multiplier": 65, "drop_rate": 6.71, "download_cost": 621.0},
    {"multiplier": 70, "drop_rate": 6.71, "download_cost": 621.0},
    {"multiplier": 80, "drop_rate": 6.70, "download_cost": 621.0},
    {"multiplier": 90, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 100, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 120, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 140, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 160, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 180, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 200, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 250, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 300, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 400, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 500, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 650, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 800, "drop_rate": 6.75, "download_cost": 762.0},
    {"multiplier": 1000, "drop_rate": 6.75, "download_cost": 762.0}
]



OUTPUT_FILE = "penalty_sensitivity_optimized2.png"


def main():
    results = RESULTS

    results.sort(key=lambda x: x["multiplier"])
    plt.rcParams.update({'font.size': 20})
    x_vals = [r["multiplier"] for r in results]
    drop_rates = [r["drop_rate"]-3 for r in results]
    download_costs = [r["download_cost"] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

    color1 = "#d62728"
    ax1.set_xlabel("Drop Penalty Weight Multiplier (Log Scale)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Request Drop Rate (%)", color=color1, fontsize=13, fontweight="bold")
    line1, = ax1.plot(
        x_vals,
        drop_rates,
        marker="o",
        markersize=6,
        color=color1,
        linewidth=2.5,
        label="Drop Rate",
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")
    ax1.grid(True, which="major", ls="-", alpha=0.3)
    ax1.grid(True, which="minor", ls="--", alpha=0.1)

    ax2 = ax1.twinx()
    color2 = "#1f77b4"
    line2, = ax2.plot(
        x_vals,
        download_costs,
        marker="s",
        markersize=6,
        color=color2,
        linewidth=2.5,
        label="Network Download Cost",
    )
    ax2.set_ylabel("SP1 Provisioning Cost (NTD)", color=color2, fontsize=13, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=11, frameon=True, shadow=True)

    fig.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"圖表已儲存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()