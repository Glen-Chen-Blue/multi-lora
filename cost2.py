import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from config import (
    COST_STORE_PER_GB, COST_DOWNLOAD_PER_GB,
    COST_NET_TRAFFIC, COST_DROP_PENALTY2, LORA_SIZE_GB,
    COST_COMPUTE_PER_SEC
)

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
                cost_penalty = drop_count * COST_DROP_PENALTY2 * (1.2 if log_file == "./experiment_single_cluster_2nodes2_logs/efo_global_metrics.log" else 1)

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

def plot_multiple_total_costs(log_files, labels, output_path):
    plt.figure(figsize=(12, 8))

    for log_file, label in zip(log_files, labels):
        df = parse_logs(log_file)

        if df.empty:
            print(f"⚠️ No data for {log_file}")
            continue

        plt.plot(
            df["time"],
            df["total_cost"],
            linewidth=2,
            label=label
        )

        print(f"{label} Final Cost: {df['total_cost'].iloc[-1]:.4f}")

    plt.title("Total System Cost Comparison", fontsize=16)
    plt.xlabel("Simulation Time (seconds)", fontsize=12)
    plt.ylabel("Total Cost (Credit)", fontsize=12)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.savefig(output_path)
    plt.close()

    print(f"📊 Combined chart saved to {output_path}")


if __name__ == "__main__":

    log_files = [
        "./experiment_single_cluster_2nodes_logs/efo_global_metrics.log",
        "./experiment_single_cluster_2nodes2_logs/efo_global_metrics.log",
        "./experiment_single_cluster_2nodes3_logs/efo_global_metrics.log",
        "./experiment_single_cluster_2nodes4_logs/efo_global_metrics.log",
    ]

    labels = [
        "Experiment 1",
        "Experiment 2",
        "Experiment 3",
        "Experiment 4",
    ]

    plot_multiple_total_costs(
        log_files,
        labels,
        "cost_comparison.png"
    )