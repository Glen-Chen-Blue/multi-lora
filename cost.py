import json
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

# 嘗試匯入 config，若失敗則使用預設值
try:
    from config import (
        COST_STORE_PER_GB, COST_DOWNLOAD_PER_GB, 
        COST_NET_TRAFFIC, COST_DROP_PENALTY, LORA_SIZE_GB,
        COST_COMPUTE_PER_SEC  # [新增] 匯入新的算力計價參數
    )
    print("✅ Successfully imported config.py")
except ImportError:
    print("⚠️ Could not import config.py. Using default values.")
    COST_STORE_PER_GB = 0.005
    COST_DOWNLOAD_PER_GB = 3.0
    COST_NET_TRAFFIC = 0.001
    COST_DROP_PENALTY = 0.1
    LORA_SIZE_GB = 0.1
    COST_COMPUTE_PER_SEC = 0.001 # 預設值
except ImportError:
    # Fallback 如果 config 裡還沒加這個參數，避免報錯
    print("⚠️ COST_COMPUTE_PER_SEC not found in config. Using default 0.001")
    COST_COMPUTE_PER_SEC = 0.001

LOG_FILE = "logs/efo_global_metrics.log"

def parse_logs(log_file):
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        return []

    data = []
    start_time = None

    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                
                # 取得時間戳記並正規化起始時間
                ts = entry.get("timestamp", 0)
                if start_time is None:
                    start_time = ts
                relative_time = ts - start_time

                totals = entry.get("efo_totals", {})
                
                # 1. 存儲成本 (Storage Cost)
                # cumulative_stored_loras 是累計的 (LoRA * Step)，代表存儲資源的使用積分
                stored_count = totals.get("total_stored_loras", 0)
                cost_storage = stored_count * LORA_SIZE_GB * COST_STORE_PER_GB

                # 2. 下載成本 (Download Cost)
                download_count = totals.get("artifact_downloads", 0)
                cost_download = download_count * LORA_SIZE_GB * COST_DOWNLOAD_PER_GB

                # 3. [修改] 算力成本 (Compute Cost)
                # 使用 total_inference_time (累積秒數) * 每秒單價
                total_inf_time = totals.get("total_inference_time", 0.0)
                cost_compute = total_inf_time * COST_COMPUTE_PER_SEC

                # 4. 網路流量成本 (Network Traffic Cost)
                offload_count = totals.get("total_offloads", 0)
                cost_network = offload_count * COST_NET_TRAFFIC

                # 5. 丟棄懲罰 (Drop Penalty)
                drop_count = totals.get("total_drops", 0)
                cost_penalty = drop_count * COST_DROP_PENALTY

                # 總成本
                total_cost = (cost_storage + cost_download + cost_compute + 
                              cost_network + cost_penalty)

                # 統計用：總完成請求數 (僅供參考，不參與成本計算)
                completed_count = totals.get("total_local_completed", 0) + totals.get("total_offload_completed", 0)

                data.append({
                    "time": relative_time,
                    "cost_storage": cost_storage,
                    "cost_download": cost_download,
                    "cost_compute": cost_compute,
                    "cost_network": cost_network,
                    "cost_penalty": cost_penalty,
                    "total_cost": total_cost,
                    "drops": drop_count,
                    "completed": completed_count
                })

            except json.JSONDecodeError:
                continue
    
    return pd.DataFrame(data)

def plot_costs(df):
    if df.empty:
        print("⚠️ No data to plot.")
        return

    plt.figure(figsize=(12, 8))
    
    # 繪製堆疊區域圖 (Stacked Area Chart)
    # 顏色選擇柔和一點的配色
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
    
    plt.stackplot(df["time"], 
                  df["cost_storage"], 
                  df["cost_download"], 
                  df["cost_compute"], 
                  df["cost_network"], 
                  df["cost_penalty"],
                  labels=['Storage', 'Download', 'Compute', 'Network', 'Drop Penalty'],
                  colors=colors,
                  alpha=0.8)
    
    # 繪製總成本線
    plt.plot(df["time"], df["total_cost"], color='black', linewidth=2, linestyle='--', label='Total Cost')

    plt.title("System Cost Breakdown Over Time", fontsize=16)
    plt.xlabel("Simulation Time (seconds)", fontsize=12)
    plt.ylabel("Cumulative Cost (Credit)", fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 在圖表右側標註最終成本數值
    final_cost = df["total_cost"].iloc[-1]
    plt.text(df["time"].iloc[-1], final_cost, f" {final_cost:.2f}", 
             verticalalignment='bottom', fontweight='bold', color='black')

    output_file = "system_cost_over_time.png"
    plt.savefig(output_file)
    print(f"📊 Chart saved to {output_file}")

if __name__ == "__main__":
    print(f"📂 Reading logs from {LOG_FILE}...")
    df = parse_logs(LOG_FILE)
    if not df.empty:
        print(f"✅ Loaded {len(df)} data points.")
        print(f"💰 Final Total Cost: {df['total_cost'].iloc[-1]:.4f}")
        print(f"   - Storage: {df['cost_storage'].iloc[-1]:.4f}")
        print(f"   - Download: {df['cost_download'].iloc[-1]:.4f}")
        print(f"   - Compute: {df['cost_compute'].iloc[-1]:.4f}")
        print(f"   - Network: {df['cost_network'].iloc[-1]:.4f}")
        print(f"   - Penalty: {df['cost_penalty'].iloc[-1]:.4f}")
        plot_costs(df)
    else:
        print("❌ Failed to parse logs or log file is empty.")