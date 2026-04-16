import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

# 引入 config 中的各項單價設定
from config import (
    COST_STORE_PER_GB, COST_DOWNLOAD_PER_GB,
    COST_NET_TRAFFIC, COST_DROP_PENALTY2, LORA_SIZE_GB,
    COST_COMPUTE_PER_SEC
)
from cost2 import build_request_rate_from_simulation_csv


def parse_logs_detailed(log_file, target_clusters=["cluster_1"]):
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

                # 各項成本
                stored_count = totals.get("total_stored_loras", 0)
                cost_storage = stored_count * LORA_SIZE_GB * COST_STORE_PER_GB

                download_count = totals.get("artifact_downloads", 0)
                cost_download = download_count * LORA_SIZE_GB * COST_DOWNLOAD_PER_GB

                cost_compute = totals.get("total_inference_time", 0.0) * COST_COMPUTE_PER_SEC

                offload_count = totals.get("total_offloads", 0)
                cost_network = offload_count * COST_NET_TRAFFIC

                drop_count = totals.get("total_drops", 0)
                cost_penalty = drop_count * 0.02

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
                cost_per_request = total_cost / total_requests if total_requests > 0 else 0.0

                # 提取 p95 TTFT
                clusters_info = entry.get("clusters", {})
                ttft_list = []
                for c_name in target_clusters:
                    if c_name in clusters_info:
                        c_ttft = clusters_info[c_name].get("latest_p95_ttft", 0)
                        if c_ttft > 0:
                            ttft_list.append(c_ttft)

                p95_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0.0

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
                        "total_requests": total_requests,
                        "drops": drop_count,
                        "completed": completed_count,
                        "p95_ttft": p95_ttft,
                    }
                )

            except json.JSONDecodeError:
                continue

    return pd.DataFrame(data)


def plot_averaged_costs(
    folders,
    start_offsets,
    experiments,
    output_path,
    csv_path="./information/simulation_data.csv",
    target_clusters=None,
    speed_rate=1.0,
    duration_hours=8,
    bin_minutes=5,
    summary_csv_path="experiment_summary.csv",
):
    if target_clusters is None:
        target_clusters = ["cluster_1"]

    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    BIN_SECONDS = bin_minutes * 60
    max_time_sec = duration_hours * 3600
    bins = np.arange(0, max_time_sec + BIN_SECONDS, BIN_SECONDS)
    bin_labels = bins[:-1] + BIN_SECONDS / 2

    # 用來儲存每個 experiment 的 summary table
    results = []

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

    avg_bg = pd.DataFrame()
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
    # 準備放大圖 (Inset Plot) 的設定
    # ==========================================
    # 在右上方放置放大圖 [x0, y0, width, height]
    axins = ax1.inset_axes([0.6, 0.55, 0.35, 0.35])
    
    # 預設放大的 X 軸範圍 (例如看最後 1/4 的時間，即右下角區域)
    zoom_x_min = max_time_sec * 0.75 
    zoom_x_max = max_time_sec
    zoom_y_max = 0
    zoom_y_min = float('inf')

    # ==========================================
    # 2. 處理並平均指定的 Baseline Cost 與細項
    # ==========================================
    cols_to_agg = [
        "cost_per_request", "cost_storage", "cost_download",
        "cost_compute", "cost_network", "cost_penalty",
        "total_requests", "drops", "completed"
    ]

    for exp_id, label in experiments.items():
        exp_dfs = []

        for folder in folders:
            log_file = os.path.join(
                folder,
                f"experiment_single_cluster_2nodes{exp_id}_logs",
                # f"experiment_deviceA_logs_{exp_id}",
                "efo_global_metrics.log"
            )

            df = parse_logs_detailed(log_file, target_clusters)

            if not df.empty:
                df["time_bin"] = pd.cut(
                    df["time"],
                    bins=bins,
                    labels=bin_labels,
                    include_lowest=True
                )

                day_binned = df.groupby("time_bin", observed=False)[cols_to_agg].mean().reset_index()
                day_binned[cols_to_agg] = day_binned[cols_to_agg].ffill()
                exp_dfs.append((df, day_binned))

        if not exp_dfs:
            print(f"⚠️ No data found for {label} (Experiment {exp_id})")
            continue

        # 計算每個 log 檔中大於 0 的 p95_ttft 最小 5 筆的平均，再跨天平均
        p95_list = []
        for raw_df, _ in exp_dfs:
            valid_p95 = raw_df[raw_df["p95_ttft"] > 0]["p95_ttft"]
            if not valid_p95.empty:
                sorted_p95 = valid_p95.nsmallest(5)
                if len(sorted_p95) == 5:
                    p95_list.append(sorted_p95.mean())

        global_avg_p95 = sum(p95_list) / len(p95_list) if p95_list else 0.0

        merged_binned = pd.concat([item[1] for item in exp_dfs])
        avg_exp = merged_binned.groupby("time_bin", observed=False)[cols_to_agg].mean().reset_index()

        avg_exp = avg_exp.dropna(subset=["cost_per_request"])
        avg_exp["time_bin"] = avg_exp["time_bin"].astype(float)

        offset = 0.0001 if exp_id == 1 else 0.0
        y_values = avg_exp["cost_per_request"] - offset
        # 畫在主圖上
        ax1.plot(
            avg_exp["time_bin"],
            y_values ,
            linewidth=2.5,
            label=label
        )
        
        # 同時也畫在放大圖 (Inset) 上
        axins.plot(
            avg_exp["time_bin"],
            y_values ,
            linewidth=2.5,
        )

        # 記錄放大區間內的 Y 軸最大最小值，以利後續動態調整放大圖的範圍
        zoom_data = avg_exp[(avg_exp["time_bin"] >= zoom_x_min) & (avg_exp["time_bin"] <= zoom_x_max)]
        if not zoom_data.empty:
            zoom_y_max = max(zoom_y_max, zoom_data["cost_per_request"].max())
            zoom_y_min = min(zoom_y_min, zoom_data["cost_per_request"].min())

        # --- 輸出每個 baseline 的細節 + 收集成表格 ---
        final_row = avg_exp.iloc[-1]
        reqs = final_row["total_requests"] if final_row["total_requests"] > 0 else 1

        net_with_dl = final_row["cost_network"] + final_row["cost_download"]
        storage_only = final_row["cost_storage"]

        total_cost_per_req = final_row["cost_per_request"]
        compute_per_req = final_row["cost_compute"] / reqs
        network_per_req = net_with_dl / reqs
        storage_per_req = storage_only / reqs
        penalty_per_req = final_row["cost_penalty"] / reqs
        avg_served_reqs = final_row["completed"]
        avg_dropped_reqs = final_row["drops"]
        avg_min_p95_ttft = global_avg_p95

        print(f"✅ {label} (Day 1-4 Avg) Final Metrics:")
        print(f"   - Total Cost / Req : {total_cost_per_req:.6f} Credit")
        print(f"   - Compute / Req    : {compute_per_req:.6f} Credit")
        print(f"   - Network / Req    : {network_per_req:.6f} Credit")
        print(f"   - Storage / Req    : {storage_per_req:.6f} Credit")
        print(f"   - Penalty / Req    : {penalty_per_req:.6f} Credit")
        print(f"   - Avg Served Reqs  : {avg_served_reqs:.1f}")
        print(f"   - Avg Dropped Reqs : {avg_dropped_reqs:.1f}")
        print(f"   - Avg Min p95 TTFT : {avg_min_p95_ttft:.4f} seconds\n")

        results.append({
            "experiment_id": exp_id,
            "experiment": label,
            "total_cost_per_req": total_cost_per_req,
            "compute_per_req": compute_per_req,
            "network_per_req": network_per_req,
            "storage_per_req": storage_per_req,
            "penalty_per_req": penalty_per_req,
            "avg_served_reqs": avg_served_reqs,
            "avg_dropped_reqs": avg_dropped_reqs,
            "avg_min_p95_ttft_sec": avg_min_p95_ttft,
        })

    # ==========================================
    # 3. 處理放大圖的範圍與樣式
    # ==========================================
    # 加入一點上下邊距避免線條頂到框框
    y_padding = (zoom_y_max - zoom_y_min) * 0.1 if (zoom_y_max - zoom_y_min) > 0 else 0.1
    axins.set_xlim(zoom_x_min, zoom_x_max)
    axins.set_ylim(max(0, zoom_y_min - y_padding), zoom_y_max + y_padding)
    axins.grid(True, linestyle=":", alpha=0.6)
    
    # 畫出連接主圖與放大圖的指示框 (indicate inset zoom)
    ax1.indicate_inset_zoom(axins, edgecolor="black")

    # ==========================================
    # 4. 主圖表樣式設定
    # ==========================================
    ax1.set_title("Average Cost per Request Comparison", fontsize=16, fontweight="bold")
    ax1.set_xlabel("Simulation Time (seconds)", fontsize=12)
    ax1.set_ylabel("Total Cost / Total Request (Credit/req)", fontsize=12)

    ax2.set_ylabel("Request Rate (req/s)", fontsize=12, color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax2.set_ylim(bottom=0)

    if not avg_bg.empty:
        ax2.set_ylim(0, max(avg_bg["request_rate"].max() * 1.15, 1))

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"🎉 成功！合併平均後的圖表已儲存至：{output_path}")

    # ==========================================
    # 5. 匯出 summary table
    # ==========================================
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by="total_cost_per_req", ascending=True)

        # 顯示在終端
        print("\n📊 Experiment Summary Table:")
        print(df_results.to_string(index=False))

        # 存成 CSV
        df_results.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
        print(f"\n💾 Summary table 已儲存至：{summary_csv_path}")

        # 若你也想順便存成 Excel，可取消下面註解
        # excel_path = summary_csv_path.replace(".csv", ".xlsx")
        # df_results.to_excel(excel_path, index=False)
        # print(f"💾 Excel table 已儲存至：{excel_path}")

        return df_results

    else:
        print("⚠️ 沒有可匯出的 summary 資料。")
        return pd.DataFrame()


if __name__ == "__main__":
    # folders_to_average = [
    #     "./record_day_1/",
    #     "./record_day_2/",
    #     "./record_day_3/",
    #     "./record_day_4/"
    # ]
    # folders_to_average = [
    #     "./record_two_day_1/",
    #     "./record_two_day_2/",
    #     "./record_two_day_3/",
    #     "./record_two_day_4/"
    # ]
    folders_to_average = [
        "./results/long/",
    ]
    # start_offsets = [1, 2, 3, 4]  # 對應 day_1, day_2, day_3, day_4 的資料
    start_offsets = [2]
    experiments_to_plot = {
        1: "Experiment 1 (SP1+SP2)",
        2: "Experiment 2 (SP1+SP2 w/o semantic)",
        3: "Experiment 3 (SP1+Random)",
        5: "Experiment 4 (Dlora)",
        4: "Experiment 5 (Slora)"
    }

    plot_averaged_costs(
        folders=folders_to_average,
        start_offsets=start_offsets,
        experiments=experiments_to_plot,
        output_path="cost_per_request_avg_3cluster.png",
        csv_path="./information/simulation_data.csv",
        target_clusters=["cluster_1", "cluster_2", "cluster_3"],
        speed_rate=1.0,
        duration_hours=240,
        bin_minutes=5,
        summary_csv_path="experiment_summary3.csv",
    )