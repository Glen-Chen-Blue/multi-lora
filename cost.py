import json
import matplotlib.pyplot as plt
import os
import pandas as pd

# 嘗試匯入 config
try:
    from config import (
        COST_STORE_PER_GB, COST_DOWNLOAD_PER_GB,
        COST_NET_TRAFFIC, COST_DROP_PENALTY2, LORA_SIZE_GB,
        COST_COMPUTE_PER_SEC
    )
    print("✅ Successfully imported config.py")
except ImportError:
    print("⚠️ Could not import config.py. Using default values.")
    COST_STORE_PER_GB = 0.005
    COST_DOWNLOAD_PER_GB = 3.0
    COST_NET_TRAFFIC = 0.001
    COST_DROP_PENALTY2 = 0.1
    LORA_SIZE_GB = 0.1
    COST_COMPUTE_PER_SEC = 0.001


def parse_logs(log_file):
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return pd.DataFrame()

    data = []
    start_time = None

    with open(log_file, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)

                ts = entry.get("timestamp", 0)
                if start_time is None:
                    start_time = ts
                relative_time = ts - start_time

                totals = entry.get("efo_totals", {})

                # Storage
                stored_count = totals.get("total_stored_loras", 0)
                cost_storage = stored_count * LORA_SIZE_GB * COST_STORE_PER_GB

                # Download
                download_count = totals.get("artifact_downloads", 0)
                cost_download = download_count * LORA_SIZE_GB * COST_DOWNLOAD_PER_GB

                # Compute
                total_inf_time = totals.get("total_inference_time", 0.0)
                cost_compute = total_inf_time * COST_COMPUTE_PER_SEC

                # Network
                offload_count = totals.get("total_offloads", 0)
                cost_network = offload_count * COST_NET_TRAFFIC

                # Penalty
                drop_count = totals.get("total_drops", 0)
                cost_penalty = drop_count * COST_DROP_PENALTY2

                total_cost = (
                    cost_storage
                    + cost_download
                    + cost_compute
                    + cost_network
                    + cost_penalty
                )

                completed_count = (
                    totals.get("total_local_completed", 0)
                    + totals.get("total_offload_completed", 0)
                )

                data.append(
                    {
                        "time": relative_time,
                        "cost_storage": cost_storage,
                        "cost_download": cost_download,
                        "cost_compute": cost_compute,
                        "cost_network": cost_network,
                        "cost_penalty": cost_penalty,
                        "total_cost": total_cost,
                        "drops": drop_count,
                        "completed": completed_count,
                    }
                )

            except json.JSONDecodeError:
                continue

    return pd.DataFrame(data)


def plot_costs(df, output_path):
    if df.empty:
        print("⚠️ No data to plot.")
        return

    plt.figure(figsize=(12, 8))

    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']

    plt.stackplot(
        df["time"],
        df["cost_storage"],
        df["cost_download"],
        df["cost_compute"],
        df["cost_network"],
        df["cost_penalty"],
        labels=['Storage', 'Download LoRA', 'Compute', 'Offloading', 'Drop Penalty'],
        colors=colors,
        alpha=0.8
    )

    plt.plot(
        df["time"],
        df["total_cost"],
        color="black",
        linewidth=2,
        linestyle="--",
        label="Total Cost",
    )

    plt.title("System Cost Breakdown Over Time", fontsize=16)
    plt.xlabel("Simulation Time (seconds)", fontsize=12)
    plt.ylabel("Cumulative Cost (Credit)", fontsize=12)

    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.3)

    final_cost = df["total_cost"].iloc[-1]
    plt.text(
        df["time"].iloc[-1],
        final_cost,
        f" {final_cost:.2f}",
        verticalalignment="bottom",
        fontweight="bold",
        color="black",
    )

    plt.savefig(output_path)
    plt.close()

    print(f"📊 Chart saved to {output_path}")


def generate_cost_plot(log_file_path, output_path):
    """
    Generate cost breakdown plot from EFO logs.

    Parameters
    ----------
    log_file_path : str
        Path to efo_global_metrics.log
    output_path : str
        Path to save the output figure
    """

    print(f"📂 Reading logs from {log_file_path}...")

    df = parse_logs(log_file_path)

    if df.empty:
        print("❌ Failed to parse logs or log file is empty.")
        return

    print(f"✅ Loaded {len(df)} data points.")

    print(f"💰 Final Total Cost: {df['total_cost'].iloc[-1]:.4f}")
    print(f"   - Storage: {df['cost_storage'].iloc[-1]:.4f}")
    print(f"   - Download: {df['cost_download'].iloc[-1]:.4f}")
    print(f"   - Compute: {df['cost_compute'].iloc[-1]:.4f}")
    print(f"   - Network: {df['cost_network'].iloc[-1]:.4f}")
    print(f"   - Penalty: {df['cost_penalty'].iloc[-1]:.4f}")

    plot_costs(df, output_path)


if __name__ == "__main__":
    log_file = "./experiment_single_cluster_2nodes_logs/efo_global_metrics.log"
    output_image = "cost_breakdown_1.png"
    generate_cost_plot(log_file, output_image)
    log_file = "./experiment_single_cluster_2nodes2_logs/efo_global_metrics.log"
    output_image = "cost_breakdown_2.png"
    generate_cost_plot(log_file, output_image)
    log_file = "./experiment_single_cluster_2nodes3_logs/efo_global_metrics.log"
    output_image = "cost_breakdown_3.png"
    generate_cost_plot(log_file, output_image)