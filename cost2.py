import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import numpy as np

from config import (
    COST_STORE_PER_GB, COST_DOWNLOAD_PER_GB,
    COST_NET_TRAFFIC, COST_DROP_PENALTY2, COST_DROP_PENALTY, LORA_SIZE_GB,
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

                stored_count = totals.get("total_stored_loras", 0)
                cost_storage = stored_count * LORA_SIZE_GB * COST_STORE_PER_GB

                download_count = totals.get("artifact_downloads", 0)
                cost_download = download_count * LORA_SIZE_GB * COST_DOWNLOAD_PER_GB

                total_inf_time = totals.get("total_inference_time", 0.0)
                cost_compute = total_inf_time * COST_COMPUTE_PER_SEC

                offload_count = totals.get("total_offloads", 0)
                cost_network = offload_count * COST_NET_TRAFFIC

                drop_count = totals.get("total_drops", 0) 
                cost_penalty = drop_count * COST_DROP_PENALTY 

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

                total_requests = totals.get("total_requests", completed_count + drop_count)
                
                # 計算單筆 request 的平均成本 (避免除以零)
                cost_per_request = total_cost / total_requests if total_requests > 0 else 0.0

                data.append(
                    {
                        "time": relative_time,
                        "cost_storage": cost_storage,
                        "cost_download": cost_download,
                        "cost_compute": cost_compute,
                        "cost_network": cost_network,
                        "cost_penalty": cost_penalty,
                        "total_cost": total_cost,
                        "cost_per_request": cost_per_request,
                        "drops": drop_count,
                        "completed": completed_count,
                        "total_requests": total_requests,
                    }
                )

            except json.JSONDecodeError:
                continue

    return pd.DataFrame(data)


def build_request_rate_from_simulation_csv(
    csv_path="./simulation_data.csv",
    target_clusters=None,
    speed_rate=2.0,
    start_offset_days=2,
    duration_hours=8,
    bin_minutes=5,
):
    """
    從 simulation_data.csv 建立 request rate 曲線
    回傳:
        bg_df: columns = ['time', 'request_rate']
               time 單位是 seconds，方便直接和 cost plot 的 x 軸對齊
    """
    if target_clusters is None:
        target_clusters = ["cluster_1", "cluster_2", "cluster_3"]

    if not os.path.exists(csv_path):
        print(f"❌ Simulation CSV not found: {csv_path}")
        return pd.DataFrame()

    print(f"📥 Loading simulation CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    START_OFFSET = 86400 * start_offset_days
    BIN_SECONDS = bin_minutes * 60
    PLOT_DURATION_SECONDS = duration_hours * 3600
    REQUIRED_RAW_DURATION = PLOT_DURATION_SECONDS * speed_rate
    RAW_END_TIME = START_OFFSET + REQUIRED_RAW_DURATION

    if "arrive_timestamp" in df.columns:
        time_col = "arrive_timestamp"
    elif "arrival_sec" in df.columns:
        time_col = "arrival_sec"
    else:
        print("❌ Error: neither 'arrive_timestamp' nor 'arrival_sec' exists in CSV.")
        return pd.DataFrame()

    df[time_col] = df[time_col].astype(float)

    mask = (df[time_col] >= START_OFFSET) & (df[time_col] < RAW_END_TIME)
    df_filtered = df[mask].copy()

    if "cluster" not in df_filtered.columns:
        print("❌ Error: column 'cluster' not found in CSV.")
        return pd.DataFrame()

    df_filtered = df_filtered[df_filtered["cluster"].isin(target_clusters)].copy()

    if df_filtered.empty:
        print("⚠️ No simulation data in selected range/clusters.")
        return pd.DataFrame()

    df_filtered["rel_time"] = df_filtered[time_col] - START_OFFSET
    df_filtered["sim_time"] = df_filtered["rel_time"] / speed_rate

    bins = np.arange(0, PLOT_DURATION_SECONDS + BIN_SECONDS, BIN_SECONDS)
    df_filtered["bin_idx"] = pd.cut(
        df_filtered["sim_time"],
        bins=bins,
        labels=False,
        include_lowest=True
    )

    counts = df_filtered.groupby("bin_idx").size()
    full_index = np.arange(len(bins) - 1)
    counts = counts.reindex(full_index, fill_value=0)

    rates = counts / BIN_SECONDS

    # 用 bin 中心點，單位轉回 seconds，直接對齊 cost plot x 軸
    bin_centers_seconds = bins[:-1] + BIN_SECONDS / 2

    bg_df = pd.DataFrame({
        "time": bin_centers_seconds,
        "request_rate": rates.values
    })

    return bg_df


def plot_multiple_total_costs_with_simulation_bg(
    log_files,
    labels,
    output_path,
    csv_path="./simulation_data.csv",
    target_clusters=None,
    speed_rate=2.0,
    start_offset_days=2,
    duration_hours=8,
    bin_minutes=5,
):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    # 先畫背景 request-rate 線
    bg_df = build_request_rate_from_simulation_csv(
        csv_path=csv_path,
        target_clusters=target_clusters,
        speed_rate=speed_rate,
        start_offset_days=start_offset_days,
        duration_hours=duration_hours,
        bin_minutes=bin_minutes,
    )

    if not bg_df.empty:
        ax2.fill_between(
            bg_df["time"],
            0,
            bg_df["request_rate"],
            color="gray",
            alpha=0.12,
            label="Request Rate (Background)"
        )
        ax2.plot(
            bg_df["time"],
            bg_df["request_rate"],
            color="gray",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Request Rate"
        )

    # 再畫 cost_per_request 曲線
    for log_file, label in zip(log_files, labels):
        df = parse_logs(log_file)

        if df.empty:
            print(f"⚠️ No data for {log_file}")
            continue

        # 這裡改繪製 cost_per_request
        ax1.plot(
            df["time"],
            df["cost_per_request"],
            linewidth=2.5,
            label=label
        )

        print(f"{label} Final Cost per Request: {df['cost_per_request'].iloc[-1]:.4f}")

    # 更新圖表標題與 Y 軸標籤
    ax1.set_title("Cost per Request Comparison with Request Rate Background", fontsize=16)
    ax1.set_xlabel("Simulation Time (seconds)", fontsize=12)
    ax1.set_ylabel("Total Cost / Total Request (Credit/req)", fontsize=12)
    
    # Set log scale for cost/request y-axis
    # ax1.set_yscale("log")

    ax2.set_ylabel("Request Rate (req/s)", fontsize=12, color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax2.set_ylim(bottom=0)

    # 視情況設定上限
    if not bg_df.empty:
        ax2.set_ylim(0, max(bg_df["request_rate"].max() * 1.15, 1))

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path + str(start_offset_days) + ".png", dpi=300)
    plt.close()

    print(f"📊 Combined chart saved to {output_path}")


if __name__ == "__main__":
    log_files = [
        "experiment_single_cluster_2nodes1_logs/efo_global_metrics.log",
        "experiment_single_cluster_2nodes2_logs/efo_global_metrics.log",
        "experiment_single_cluster_2nodes3_logs/efo_global_metrics.log",
        "experiment_single_cluster_2nodes4_logs/efo_global_metrics.log",
        "experiment_single_cluster_2nodes5_logs/efo_global_metrics.log",
        "experiment_single_cluster_2nodes6_logs/efo_global_metrics.log"
    ]
    folder_name = "./"

    log_files = [folder_name + f for f in log_files]


    labels = [
        "Experiment 1 (SP1+SP2)",
        "Experiment 2 (SP1+SP2 w/o semantic)",
        "Experiment 3 (SP1+Random)",
        "Experiment 4 (LRU+Random)",
        "Experiment 5 (Dlora)",
        "Experiment 6 (Slora)"
    ]

    plot_multiple_total_costs_with_simulation_bg(
        log_files=log_files,
        labels=labels,
        output_path="cost_per_request",  # 更改輸出檔名以避免覆寫原檔
        csv_path="./information/simulation_data.csv",
        target_clusters=["cluster_1"],
        speed_rate=1.0,
        start_offset_days=4,
        duration_hours=8,
        bin_minutes=5,
    )