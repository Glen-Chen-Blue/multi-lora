import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from cost2 import parse_logs, build_request_rate_from_simulation_csv

def plot_averaged_costs(
    folders,
    start_offsets,
    labels,
    output_path,
    csv_path="./information/simulation_data.csv",
    target_clusters=None,
    speed_rate=1.0,
    duration_hours=8,
    bin_minutes=5,
):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    # 定義統一的時間分桶 (例如每 5 分鐘 = 300 秒)
    BIN_SECONDS = bin_minutes * 60
    max_time_sec = duration_hours * 3600
    bins = np.arange(0, max_time_sec + BIN_SECONDS, BIN_SECONDS)
    bin_labels = bins[:-1] + BIN_SECONDS / 2  # 取區間中心點作為 X 軸

    # ==========================================
    # 1. 處理並平均背景的 Request Rate
    # ==========================================
    bg_dfs = []
    for offset in start_offsets:
        df_bg = build_request_rate_from_simulation_csv(
            csv_path=csv_path,
            target_clusters=target_clusters,
            speed_rate=speed_rate,
            start_offset_days=offset,
            duration_hours=duration_hours,
            bin_minutes=bin_minutes,
        )
        if not df_bg.empty:
            bg_dfs.append(df_bg)
    
    if bg_dfs:
        merged_bg = pd.concat(bg_dfs)
        avg_bg = merged_bg.groupby("time", as_index=False)["request_rate"].mean()

        ax2.fill_between(
            avg_bg["time"], 0, avg_bg["request_rate"],
            color="gray", alpha=0.12, label="Avg Request Rate (Bg)"
        )
        ax2.plot(
            avg_bg["time"], avg_bg["request_rate"],
            color="gray", linestyle="--", linewidth=2, alpha=0.8, label="Avg Request Rate"
        )

    # ==========================================
    # 2. 處理並平均 6 種 Baseline 的 Cost (修復對齊問題)
    # ==========================================
    for i, label in enumerate(labels, start=1):
        exp_dfs = []
        for folder in folders:
            log_file = os.path.join(folder, f"experiment_single_cluster_2nodes{i}_logs", "efo_global_metrics.log")
            df = parse_logs(log_file)
            
            if not df.empty:
                # 把每筆 log 歸類到對應的時間桶子裡
                df['time_bin'] = pd.cut(df['time'], bins=bins, labels=bin_labels, include_lowest=True)
                
                # 算出「這一天」在每個桶子內的平均 Cost
                day_binned = df.groupby('time_bin', observed=False)['cost_per_request'].mean().reset_index()
                
                # 向前填充 (ffill)：如果某個 5 分鐘內完全沒有 log，就延續上一個桶子的 Cost 狀態
                day_binned['cost_per_request'] = day_binned['cost_per_request'].ffill()
                
                exp_dfs.append(day_binned)
        
        if not exp_dfs:
            print(f"⚠️ No data found for {label}")
            continue
            
        # 將 4 天的 Binned 數據合併，再算一次跨天數的平均
        merged_exp = pd.concat(exp_dfs)
        avg_exp = merged_exp.groupby('time_bin', observed=False)['cost_per_request'].mean().reset_index()
        
        # 移除空值並轉回數值型態以便繪圖
        avg_exp = avg_exp.dropna(subset=['cost_per_request'])
        avg_exp['time_bin'] = avg_exp['time_bin'].astype(float)

        # 畫出平滑的平均曲線
        ax1.plot(
            avg_exp['time_bin'],
            avg_exp['cost_per_request'],
            linewidth=2.5,
            label=label
        )
        print(f"✅ {label} (Day 1-4 Avg) Final Cost: {avg_exp['cost_per_request'].iloc[-1]:.4f}")

    # ==========================================
    # 3. 圖表樣式設定
    # ==========================================
    ax1.set_title("Average Cost per Request Comparison (Day 1 to Day 4)", fontsize=16, fontweight='bold')
    ax1.set_xlabel("Simulation Time (seconds)", fontsize=12)
    ax1.set_ylabel("Total Cost / Total Request (Credit/req)", fontsize=12)

    ax2.set_ylabel("Request Rate (req/s)", fontsize=12, color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax2.set_ylim(bottom=0)
    
    if bg_dfs and not avg_bg.empty:
        ax2.set_ylim(0, max(avg_bg["request_rate"].max() * 1.15, 1))

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"\n🎉 成功！合併平均後的平滑圖表已儲存至：{output_path}")

if __name__ == "__main__":
    folders_to_average = [
        "./record_day_1/",
        "./record_day_2/",
        "./record_day_3/",
        "./record_day_4/"
    ]
    start_offsets = [1, 2, 3, 4]  
    
    labels = [
        "Experiment 1 (SP1+SP2)",
        "Experiment 2 (SP1+SP2 w/o semantic)",
        "Experiment 3 (SP1+Random)",
        "Experiment 4 (LRU+Random)",
        "Experiment 5 (Dlora)",
        "Experiment 6 (Slora)"
    ]

    plot_averaged_costs(
        folders=folders_to_average,
        start_offsets=start_offsets,
        labels=labels,
        output_path="cost_per_request_avg_day1_to_4.png",
        csv_path="./information/simulation_data.csv",
        target_clusters=["cluster_1"],
        speed_rate=1.0,
        duration_hours=8,
        bin_minutes=5,
    )