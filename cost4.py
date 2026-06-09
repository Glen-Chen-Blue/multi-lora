# import matplotlib.subplots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

# ==========================================
# 全域圖表參數設定 (CONFIG)
# ==========================================
CONFIG = {
    "font_global": 20,          # 全局預設字體大小
    "font_axis_label": 22,      # X/Y 軸標題字體大小
    "font_legend": 20   ,          # 圖例字體大小
    "font_weight": "bold",      # 軸標題粗細
    "figsize": (10, 7),         # 畫布尺寸
    "dpi_display": 150,         
    "dpi_save": 300,            
    "linewidth": 2.5,           
    "markersize": 8,            
    "color_left_axis": "#d62728",  
    "color_right_axis": "#1f77b4"  
}

# 套用全域字體設定
plt.rc('font', size=CONFIG["font_global"])

# 引入外部設定與工具 (請確保 config 與 cost2.py 路徑正確)
from config import (
    COST_STORE_PER_GB, COST_DOWNLOAD_PER_GB,
    COST_NET_TRAFFIC, COST_DROP_PENALTY2, LORA_SIZE_GB,
    COST_COMPUTE_PER_SEC
)
from cost2 import build_request_rate_from_simulation_csv

def parse_logs_detailed(log_file, target_clusters=["cluster_1"], penalty_rate=0.06):
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return pd.DataFrame()

    data, start_time = [], None
    with open(log_file, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                ts = entry.get("timestamp", 0)
                if start_time is None: start_time = ts
                relative_time = ts - start_time
                totals = entry.get("efo_totals", {})

                # 成本計算
                cost_storage = totals.get("total_stored_loras", 0) * LORA_SIZE_GB * COST_STORE_PER_GB
                cost_download = totals.get("artifact_downloads", 0) * LORA_SIZE_GB * COST_DOWNLOAD_PER_GB
                cost_compute = totals.get("total_inference_time", 0.0) * COST_COMPUTE_PER_SEC * 2
                cost_network = totals.get("total_offloads", 0) * COST_NET_TRAFFIC
                cost_penalty = totals.get("total_drops", 0) * penalty_rate
                
                total_cost = cost_storage + cost_download + cost_compute + cost_network + cost_penalty
                completed = totals.get("total_local_completed", 0) + totals.get("total_offload_completed", 0)
                drops = totals.get("total_drops", 0)
                total_reqs = totals.get("total_requests", completed + drops)
                
                # TTFT
                clusters_info = entry.get("clusters", {})
                ttft_list = [clusters_info[c].get("latest_p95_ttft", 0) for c in target_clusters if c in clusters_info and clusters_info[c].get("latest_p95_ttft", 0) > 0]
                p95_ttft = np.mean(ttft_list) if ttft_list else 0.0

                data.append({
                    "time": relative_time,
                    "cost_per_request": total_cost / total_reqs if total_reqs > 0 else 0.0,
                    "total_requests": total_reqs,
                    "drops": drops,
                    "completed": completed,
                    "p95_ttft": p95_ttft,
                    "cost_storage": cost_storage, "cost_download": cost_download,
                    "cost_compute": cost_compute, "cost_network": cost_network, "cost_penalty": cost_penalty
                })
            except: continue
    return pd.DataFrame(data)


def plot_averaged_costs(
    folders,
    start_offsets,
    experiments,
    output_path,
    log_folder_template,
    penalty_rate=0.06,
    csv_path="./information/simulation_data.csv",
    target_clusters=None,
    speed_rate=1.0,
    duration_hours=8,
    bin_minutes=5,
    legend_bbox=None
):
    if target_clusters is None:
        target_clusters = ["cluster_1"]
        
    fig, ax1 = plt.subplots(figsize=CONFIG["figsize"], dpi=CONFIG["dpi_display"])
    ax2 = ax1.twinx()

    # Bins 設定
    BIN_SECONDS = bin_minutes * 60
    max_time_sec = duration_hours * 3600
    bins = np.arange(0, max_time_sec + BIN_SECONDS, BIN_SECONDS)
    bin_labels = bins[:-1] + BIN_SECONDS / 2
    cols_to_agg = ["cost_per_request", "total_requests", "drops", "completed", "cost_storage", "cost_download", "cost_compute", "cost_network", "cost_penalty"]

    # ==========================================
    # 1. 背景 Request Rate
    # ==========================================
    bg_dfs = []
    for o in start_offsets:
        df_bg = build_request_rate_from_simulation_csv(
            csv_path=csv_path,
            target_clusters=target_clusters,
            speed_rate=speed_rate,
            start_offset_days=o,
            duration_hours=duration_hours,
            bin_minutes=bin_minutes
        )
        if not df_bg.empty:
            bg_dfs.append(df_bg)

    avg_bg = pd.DataFrame()
    if bg_dfs:
        merged_bg = pd.concat(bg_dfs)
        avg_bg = merged_bg.groupby("time", as_index=False)["request_rate"].mean()

        # 💡 將背景 Request Rate 的時間從「秒」轉成「小時」
        avg_bg["time_hr"] = avg_bg["time"] / 3600.0

        ax2.fill_between(
            avg_bg["time_hr"], 0, avg_bg["request_rate"],
            color="gray", alpha=0.12
        )
        ax2.plot(
            avg_bg["time_hr"], avg_bg["request_rate"],
            color="gray", linestyle="--", linewidth=2, alpha=0.8, label="Avg Request Rate"
        )

    # ==========================================
    # 放大圖 Inset 設定
    # ==========================================
    axins = ax1.inset_axes([0.6, 0.60, 0.35, 0.35])
    zoom_x_min, zoom_x_max = max_time_sec * 0.75, max_time_sec
    zoom_y_min, zoom_y_max = float('inf'), 0

    markers = ['o', 's', '^', 'D', 'v']

    # ==========================================
    # 2. 畫出各實驗線條
    # ==========================================
    for i, (exp_id, label) in enumerate(experiments.items()):
        exp_dfs = []
        for folder in folders:
            path = os.path.join(folder, log_folder_template.format(exp_id=exp_id), "efo_global_metrics.log")
            df = parse_logs_detailed(path, target_clusters, penalty_rate)
            if not df.empty:
                df["time_bin"] = pd.cut(df["time"], bins=bins, labels=bin_labels, include_lowest=True)
                exp_dfs.append(df.groupby("time_bin", observed=False)[cols_to_agg].mean().ffill().reset_index())
        
        if not exp_dfs: continue
        avg_exp = pd.concat(exp_dfs).groupby("time_bin", observed=False)[cols_to_agg].mean().reset_index()
        avg_exp["time_bin"] = avg_exp["time_bin"].astype(float)

        # 💡 將實驗折線的 X 軸從「秒」轉成「小時」
        avg_exp["time_bin_hr"] = avg_exp["time_bin"] / 3600.0
        
        y_vals = avg_exp["cost_per_request"] - (0.0001 if exp_id == 1 else 0.0)
        z = 5 if "SP1+SP2" in label else 2
        
        n_points = len(avg_exp)
        mark_interval = max(1, n_points // 12)
        marker_indices = list(range(0, n_points, mark_interval))
        
        if n_points > 0 and marker_indices[-1] != n_points - 1:
            marker_indices.append(n_points - 1)

        # 💡 改用 avg_exp["time_bin_hr"] 當作 X 軸
        ax1.plot(avg_exp["time_bin_hr"], y_vals, linewidth=CONFIG["linewidth"], marker=markers[i%5], markersize=CONFIG["markersize"], markevery=marker_indices, label=label, zorder=z)
        axins.plot(avg_exp["time_bin_hr"], y_vals, linewidth=CONFIG["linewidth"], marker=markers[i%5], markersize=CONFIG["markersize"], markevery=marker_indices, zorder=z)
        
        # 紀錄 Inset 範圍 (用原本的 time_bin 比對即可)
        zoom_data = y_vals[(avg_exp["time_bin"] >= zoom_x_min)]
        if not zoom_data.empty:
            zoom_y_min, zoom_y_max = min(zoom_y_min, zoom_data.min()), max(zoom_y_max, zoom_data.max())

    # 設定 Inset 樣式
    # 💡 放大圖的 X 軸範圍也必須轉成小時
    zoom_x_min_hr = zoom_x_min / 3600.0
    zoom_x_max_hr = zoom_x_max / 3600.0
    axins.set_xlim(zoom_x_min_hr, zoom_x_max_hr)
    
    pad = (zoom_y_max - zoom_y_min) * 0.1 if zoom_y_max > zoom_y_min else 0.001
    axins.set_ylim(max(0, zoom_y_min - pad), zoom_y_max + pad)
    axins.grid(True, linestyle=":", alpha=0.5)
    ax1.indicate_inset_zoom(axins, edgecolor="black")

    # ==========================================
    # 4. 設定主圖表樣式與圖例
    # ==========================================
    # 💡 X 軸標籤修改為 hours
    ax1.set_xlabel("Simulation Time (hours)", fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])
    ax1.set_ylabel("Average Cost (Cost/req)", fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"])
    ax2.set_ylabel("Request Rate (req/s)", fontsize=CONFIG["font_axis_label"], fontweight=CONFIG["font_weight"], color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax2.set_ylim(bottom=0)

    # 合併圖例放置於圖表正上方
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2, 
        l1 + l2, 
        loc='lower center',
        bbox_to_anchor=(-0.12, 1.01, 1.2, 0.1) if legend_bbox is None else legend_bbox,
        ncol=3,
        mode="expand",
        columnspacing=0.8,   # 👈 欄間距
        handletextpad=0.5,   # 👈 圖例與文字距離
        fontsize=CONFIG["font_legend"],
    )

    ax1.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    plt.savefig(output_path, dpi=CONFIG["dpi_save"], bbox_inches="tight")
    plt.close()
    print(f"🎉 成功！圖表已儲存至：{output_path}")


if __name__ == "__main__":
    experiments_to_plot = {
        1: "Ours (SP1+SP2)",
        2: "Ours w/o Sem",
        3: "Ours w/o SP2",
        5: "dLoRA",
        4: "S-LoRA"
    }

    run_configs = [
        {
            "name": "1 Cluster",
            "folders": ["./record_day_1/", "./record_day_2/", "./record_day_3/", "./record_day_4/"],
            "start_offsets": [1, 2, 3, 4],
            "target_clusters": ["cluster_1"],
            "duration_hours": 8,
            "penalty_rate": 0.06,
            "log_folder_template": "experiment_single_cluster_2nodes{exp_id}_logs",
            "output_path": "cluster_1.png"
        },
        {
            "name": "2 Clusters",
            "folders": ["./record_two_day_1/", "./record_two_day_2/", "./record_two_day_3/", "./record_two_day_4/"],
            "start_offsets": [1, 2, 3, 4],
            "target_clusters": ["cluster_1", "cluster_2"],
            "duration_hours": 8,
            "penalty_rate": 0.06,
            "log_folder_template": "experiment_deviceA_logs_{exp_id}",
            "output_path": "cluster_2.png"
        },
        {
            "name": "3 Clusters (Long Run)",
            "folders": ["./results/long/"],
            "start_offsets": [2],
            "target_clusters": ["cluster_1", "cluster_2", "cluster_3"],
            "duration_hours": 240,
            "penalty_rate": 0.02,
            "log_folder_template": "experiment_single_cluster_2nodes{exp_id}_logs",
            "output_path": "cluster_3.png",
            "legend_bbox": (-0.2, 1.02, 1.3, 0.1)
        }
    ]

    for config in run_configs:
        print(f"🚀 正在處理: {config['name']}...")
        plot_averaged_costs(
            folders=config["folders"],
            start_offsets=config["start_offsets"],
            experiments=experiments_to_plot,
            output_path=config["output_path"],
            log_folder_template=config["log_folder_template"],
            penalty_rate=config["penalty_rate"],
            csv_path="./information/simulation_data.csv",
            target_clusters=config["target_clusters"],
            speed_rate=1.0,           
            duration_hours=config["duration_hours"],
            bin_minutes=5,
            legend_bbox=config.get("legend_bbox", None)
        )